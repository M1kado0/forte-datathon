"""Stand-in for the tracks of #32, so the heuristics (#33) can be tuned before the real tracker exists.

Two steps, both local:
  detect  D-FINE-S (person, cup, cell phone, bottle, book) and OWLv2 (cash and document queries) on every
          frame at 4 fps in native resolution. Boxes are stored with low scores, so score sweeps do not
          need the GPU again. One parquet per video in data/processed/standin/.
  track   a greedy IoU tracker over those boxes, and the TRACKS_SCHEMA frame the heuristics read.

Run: python -m scripts.standin_tracks detect [video_id ...]
Throw this away when #32 lands; the sweep in scripts/heuristics_lovo.py then runs on the real tracks.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, VIDEOS_DIR, WEIGHTS
from src.data.video import DEFAULT_FPS, iter_frames

RAW_DIR = DATA_PROCESSED / "standin"
DFINE_CLASSES = ("person", "cup", "cell phone", "bottle", "book")
DFINE_MIN = 0.3  # lowest score kept; the sweep raises it
OWL_MIN = 0.08  # lowest OWLv2 score kept; the sweep raises it
OWL_QUERIES = {
    "cash": ["a photo of banknotes", "a stack of cash", "a pile of banknotes", "money"],
    "document": ["a passport", "an identity card", "a document", "a sheet of paper"],
}
OWL_NMS_IOU = 0.5
OWL_PER_QUERY = 15


def _load():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import (
        AutoImageProcessor,
        AutoModelForObjectDetection,
        AutoProcessor,
        Owlv2ForObjectDetection,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dproc = AutoImageProcessor.from_pretrained(WEIGHTS / "dfine-small")
    dfine = AutoModelForObjectDetection.from_pretrained(WEIGHTS / "dfine-small").to(device).eval()
    oproc = AutoProcessor.from_pretrained(WEIGHTS / "owlv2-base")
    owl = Owlv2ForObjectDetection.from_pretrained(WEIGHTS / "owlv2-base").to(device).eval()
    return torch, device, dproc, dfine, oproc, owl


def detect_video(path: Path, models, fps: float = DEFAULT_FPS) -> pd.DataFrame:
    """Raw boxes of one video: t, source, kind, label, query, x1..y2, conf. No tracking yet."""
    import torch
    from PIL import Image
    from torchvision.ops import nms

    _, device, dproc, dfine, oproc, owl = models
    names = {i: n for i, n in dfine.config.id2label.items() if n in DFINE_CLASSES}
    queries = [(kind, q) for kind, qs in OWL_QUERIES.items() for q in qs]
    texts = [[q for _, q in queries]]
    rows = []
    for t, bgr in iter_frames(path, fps):
        image = Image.fromarray(bgr[:, :, ::-1])
        w, h = image.size
        with torch.no_grad():
            inputs = dproc(images=image, return_tensors="pt").to(device)
            res = dproc.post_process_object_detection(
                dfine(**inputs), target_sizes=[(h, w)], threshold=DFINE_MIN
            )[0]
            for score, label, box in zip(
                res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist(), strict=True
            ):
                if label in names:
                    kind = "person" if names[label] == "person" else "object"
                    rows.append((t, "dfine", kind, names[label], "", *box, score))

            inputs = oproc(text=texts, images=image, return_tensors="pt").to(device)
            out = owl(**inputs)
            scores = out.logits[0].sigmoid()  # boxes x queries
            side = max(h, w)  # OWLv2 pads to a square at the bottom/right
            cx, cy, bw, bh = (out.pred_boxes[0] * side).unbind(-1)
            boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1)
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
            for qi, (kind, q) in enumerate(queries):
                keep = (scores[:, qi] >= OWL_MIN).nonzero().flatten()
                if not len(keep):
                    continue
                keep = keep[nms(boxes[keep], scores[keep, qi], OWL_NMS_IOU)][:OWL_PER_QUERY]
                for i in keep.tolist():
                    rows.append((t, "owl", kind, kind, q, *boxes[i].tolist(), float(scores[i, qi])))
    cols = ["t", "source", "kind", "label", "query", "x1", "y1", "x2", "y2", "conf"]
    return pd.DataFrame(rows, columns=cols)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix of boxes (n, 4) and (m, 4) in x1 y1 x2 y2."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def greedy_iou_track(
    boxes: pd.DataFrame, fps: float = DEFAULT_FPS, min_iou: float = 0.3, max_age: int = 8, first_id: int = 0
) -> pd.DataFrame:
    """Add track_id: each frame, pairs of (live track, box) are taken best IoU first when IoU >= min_iou;
    a track missing for more than max_age frames is closed. Boxes left over start new tracks."""
    boxes = boxes.assign(frame=(boxes["t"] * fps).round().astype(int)).sort_values(["frame", "conf"])
    ids = np.full(len(boxes), -1, dtype=np.int64)
    live: dict[int, tuple[int, np.ndarray]] = {}  # id -> (last frame, last box)
    next_id = first_id
    position = 0
    for frame, group in boxes.groupby("frame", sort=True):
        cur = group[["x1", "y1", "x2", "y2"]].to_numpy()
        live = {i: v for i, v in live.items() if frame - v[0] <= max_age}
        assigned = np.full(len(group), -1)
        if live and len(group):
            keys = list(live)
            m = iou(np.stack([live[k][1] for k in keys]), cur)
            pairs = sorted(zip(*np.nonzero(m >= min_iou), strict=True), key=lambda p: -m[p])
            used_t, used_b = set(), set()
            for ti, bi in pairs:
                if ti in used_t or bi in used_b:
                    continue
                used_t.add(ti)
                used_b.add(bi)
                assigned[bi] = keys[ti]
        for bi in range(len(group)):
            if assigned[bi] < 0:
                assigned[bi] = next_id
                next_id += 1
            live[int(assigned[bi])] = (frame, cur[bi])
        ids[position : position + len(group)] = assigned
        position += len(group)
    return boxes.assign(track_id=ids).drop(columns="frame")


def build_tracks(
    raw: pd.DataFrame,
    person_min: float = 0.5,
    object_min: dict[str, float] | None = None,
    cash_min: float = 0.10,
    cash_queries: tuple[str, ...] | None = None,
    doc_min: float = 0.10,
    doc_queries: tuple[str, ...] | None = None,
    min_iou: float = 0.3,
    max_age: int = 8,
) -> pd.DataFrame:
    """TRACKS_SCHEMA rows from raw boxes. The best query per OWLv2 box region wins by NMS across queries."""
    object_min = object_min or {}
    keep = pd.Series(False, index=raw.index)
    keep |= (raw["kind"] == "person") & (raw["conf"] >= person_min)
    keep |= (raw["kind"] == "object") & (raw["conf"] >= raw["label"].map(object_min).fillna(DFINE_MIN))
    cash = raw["kind"] == "cash"
    if cash_queries is not None:
        cash &= raw["query"].isin(cash_queries)
    keep |= cash & (raw["conf"] >= cash_min)
    doc = raw["kind"] == "document"
    if doc_queries is not None:
        doc &= raw["query"].isin(doc_queries)
    keep |= doc & (raw["conf"] >= doc_min)
    boxes = raw[keep]
    out, first_id = [], 0
    for _, group in boxes.groupby(["kind", "label"], sort=True):
        tracked = greedy_iou_track(group, min_iou=min_iou, max_age=max_age, first_id=first_id)
        first_id = int(tracked["track_id"].max()) + 1
        out.append(tracked)
    if not out:
        return pd.DataFrame(columns=["track_id", "kind", "label", "t", "x1", "y1", "x2", "y2", "conf"])
    tracks = pd.concat(out)
    # one row per (track, t): the best box when two boxes of a track land in one frame
    tracks = tracks.sort_values("conf", ascending=False).drop_duplicates(["track_id", "t"])
    return tracks[["track_id", "kind", "label", "t", "x1", "y1", "x2", "y2", "conf"]].sort_values(
        ["t", "track_id"], ignore_index=True
    )


def main(argv: list[str]) -> None:
    cmd, wanted = argv[0], set(argv[1:])
    if cmd != "detect":
        raise SystemExit(__doc__)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    models = _load()
    for path in sorted(VIDEOS_DIR.glob("*.mov")):
        stem = path.stem.replace("Х", "X")  # the files use a Cyrillic Х
        out = RAW_DIR / f"{stem}.parquet"
        if (wanted and stem not in wanted) or out.exists():
            continue
        df = detect_video(path, models)
        df.to_parquet(out)
        print(f"{stem}: {df['t'].nunique()} frames, {len(df)} boxes", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
