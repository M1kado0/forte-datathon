"""Tracks of people, objects, cash and documents for branch B (#32), in the TRACKS_SCHEMA format.

    uv run python -m src.features.tracks [--force] [--no-log] [video ...]

Without arguments every video in data/raw/videos is tracked and cached in data/processed/tracks/<setting>/,
where <setting> is a short hash of the classes, queries and rates, so changing any of them never reads a stale
cache. Only boxes of people are kept: no faces, no identities (TZ-F.1).
"""

import csv
import datetime as dt
import hashlib
import itertools
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from contracts.data import TRACKS_SCHEMA, has_area
from src.config import EXPERIMENTS, TRACKS_DIR, VIDEOS_DIR
from src.data.labels import load_labels, normalize_name
from src.data.video import DEFAULT_FPS, iter_frames
from src.features import detectors
from src.features.bytetrack import ByteTracker
from src.features.detectors import Box, Detector

REF_VIDEO = "X-14"  # the 145 s clip the time budget is checked on
BATCH = 8  # frames per model call
OWL_EVERY = (
    2  # OWLv2 sees every second decoded frame (2/s): it is the slow model, and the heuristics bridge 1 s
)
HIGH_CONF = {"person": 0.5, "object": 0.4, "cash": 0.1, "document": 0.1}  # a box above this may start a track
LOW_CONF = {"person": 0.2, "object": 0.2, "cash": 0.1, "document": 0.1}  # a box above this may continue one
COLUMNS = list(TRACKS_SCHEMA.columns)


def setting_id(fps: float = DEFAULT_FPS) -> str:
    """Short hash of everything that changes what a track cache holds."""
    setting = {
        "objects": detectors.OBJECT_CLASSES,
        "cash": detectors.CASH_QUERIES,
        "document": detectors.DOCUMENT_QUERIES,
        "extra": detectors.EXTRA_QUERIES,
        "fps": fps,
        "owl_every": OWL_EVERY,
        "coco_min": detectors.COCO_MIN_CONF,
        "owl_min": detectors.OWL_MIN_CONF,
        "high": HIGH_CONF,
        "low": LOW_CONF,
    }
    return hashlib.sha1(json.dumps(setting, sort_keys=True).encode()).hexdigest()[:8]


def build_tracks(path: str | Path, detector: Detector, fps: float = DEFAULT_FPS) -> pd.DataFrame:
    """Decode the video, detect in batches, track every kind and return the validated long table."""
    ids = itertools.count()
    trackers: dict[tuple[str, str], ByteTracker] = {}
    rows: list[tuple] = []
    vocab_keys = set(detectors.owl_groups())  # (kind, label) found by OWLv2, seen on every OWL_EVERY-th frame

    def add(t: float, found: Sequence[Box], vocab_frame: bool) -> None:
        by_class: dict[tuple[str, str], list[Box]] = {}
        for box in found:
            by_class.setdefault((box.kind, box.label), []).append(box)
        # a tracker also sees the frames where its class is absent, so its lost counter keeps time; the OWLv2
        # classes skip the frames OWLv2 did not look at
        for key in trackers.keys() | by_class.keys():
            if key in vocab_keys and not vocab_frame:
                continue
            lost = int(fps) if key in vocab_keys else int(2 * fps)
            trackers.setdefault(
                key, ByteTracker(ids, high=HIGH_CONF[key[0]], low=LOW_CONF[key[0]], max_lost=lost)
            )
            group = by_class.get(key, [])
            xyxy = np.array([[b.x1, b.y1, b.x2, b.y2] for b in group]).reshape(-1, 4)
            conf = np.array([b.conf for b in group])
            for box, track_id in zip(group, trackers[key].update(xyxy, conf), strict=True):
                if track_id >= 0:
                    rows.append(
                        (int(track_id), box.kind, box.label, t, box.x1, box.y1, box.x2, box.y2, box.conf)
                    )

    batch: list[tuple[float, np.ndarray]] = []
    seen = 0  # frames already flushed

    def flush() -> None:
        nonlocal seen
        if not batch:
            return
        times, frames = [t for t, _ in batch], [f for _, f in batch]
        coco = detector.coco(frames)
        wide = [i for i in range(len(frames)) if (seen + i) % OWL_EVERY == 0]
        vocab = dict(zip(wide, detector.open_vocab([frames[i] for i in wide]), strict=True)) if wide else {}
        for i, t in enumerate(times):
            add(t, [*coco[i], *vocab.get(i, [])], i in vocab)
        seen += len(batch)
        batch.clear()

    for t, frame in iter_frames(path, fps):
        batch.append((t, frame))
        if len(batch) == BATCH:
            flush()
    flush()
    table = pd.DataFrame(rows, columns=COLUMNS)
    table = table[has_area(table)].sort_values(["t", "track_id"]).reset_index(drop=True)
    return TRACKS_SCHEMA.validate(table)


def video_id(path: str | Path) -> str:
    return normalize_name(Path(path).stem)


def cache_path(path: str | Path, fps: float = DEFAULT_FPS) -> Path:
    return TRACKS_DIR / setting_id(fps) / f"{video_id(path)}.parquet"


def load_tracks(path: str | Path, fps: float = DEFAULT_FPS) -> pd.DataFrame | None:
    """The cached tracks of a video, or None when there are none for the current setting."""
    cached = cache_path(path, fps)
    return TRACKS_SCHEMA.validate(pd.read_parquet(cached)) if cached.is_file() else None


def get_tracks(
    path: str | Path, detector: Detector, fps: float = DEFAULT_FPS, force: bool = False
) -> pd.DataFrame:
    """Cached tracks of the video; built, validated and cached on a miss."""
    cached = None if force else load_tracks(path, fps)
    if cached is not None:
        return cached
    table = build_tracks(path, detector, fps)
    target = cache_path(path, fps)
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(target, index=False)
    return table


def log_experiment(summary: dict) -> None:
    """Append one row keyed by the file's own header; the numbers go to params."""
    with EXPERIMENTS.open(newline="") as f:
        header = next(csv.reader(f))
    row = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "note": "tracks-dfine-bytetrack-owlv2",
        "params": json.dumps(summary, ensure_ascii=False),
    }
    with EXPERIMENTS.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=header, restval="").writerow(row)


def run(files: Sequence[Path], force: bool) -> dict:
    detector = detectors.LocalDetector()
    timings: dict[str, float] = {}
    ref = None
    for path in files:
        start = time.perf_counter()
        # the reference video is always rebuilt: its time is the number the budget is checked against
        table = get_tracks(path, detector, force=force or video_id(path) == REF_VIDEO)
        timings[video_id(path)] = time.perf_counter() - start
        if video_id(path) == REF_VIDEO:
            ref = {"video_id": REF_VIDEO, "duration": float(table["t"].max()), "seconds": timings[REF_VIDEO]}
        kinds = table.groupby("kind")["track_id"].nunique().to_dict()
        print(
            f"  {video_id(path)}: {len(table)} rows, tracks {kinds}, {timings[video_id(path)]:.1f} s",
            flush=True,
        )
    return {
        "setting": setting_id(),
        "n_videos": len(files),
        "detect_seconds_ref_video": ref,  # decode + D-FINE + OWLv2 + tracking; branch A adds its own, 45 s together
    }


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    # the files named in labels.csv: X-47 and X-48 are copies of X-4 and X-5 and are left out
    files = [Path(a) for a in args] or [VIDEOS_DIR / f for f in load_labels()["file"].unique()]
    summary = run(files, force="--force" in sys.argv)
    if "--no-log" not in sys.argv:
        log_experiment(summary)
