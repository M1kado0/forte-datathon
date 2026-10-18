"""Check (TZ-5.2, #70): which new ADR 0005 items does the detector see from above?

D-FINE classes laptop, cup, cell phone, bottle, book and OWLv2 queries "key", "open safe", "open door" are scored on
the frames around the labels they are meant for and on frames of the videos without violations, as in #19.
Frames are decoded and scored locally with the production `LocalDetector`; nothing leaves the machine. Run:
    uv run python -m src.features.adr5_check [--no-log]
"""

import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from src.config import VIDEOS_DIR
from src.data.video import iter_frames
from src.features import detectors
from src.features.cash_check import (
    CLEAN_FRAMES,
    FPS,
    frame_seconds,
    in_window,
    load_label_table,
    pick_frame_indices,
)
from src.features.experiments import append_row
from src.heuristics.thresholds import Thresholds

OWL_THRESHOLD = detectors.OWL_MIN_CONF
# OWLv2 floors tried, to see whether any separates label frames from clean ones
OWL_SWEEP = (0.1, 0.15, 0.2, 0.25)
# floor of the classes thresholds.py has none for (laptop) or keeps off (book)
LOOSE_CONF = 0.3
WINDOW_S = 3.0  # frames from [second - WINDOW_S, second + WINDOW_S] around a violation label
BATCH = 8
LABEL_GOAL = 0.70  # an item passes when it is found on at least this share of its target labels...
CLEAN_RATIO = 0.5  # ...and its share of clean frames is under this fraction of its share of target frames
NOTE = "adr5-check-dfine-owlv2"


@dataclass(frozen=True)
class Item:
    name: str  # a COCO class, an OWLv2 query, or a union of classes (see members)
    source: Literal["coco", "owl"]
    targets: tuple[str, ...]  # the labels of ADR 0005 whose scenario the item should show
    # confidence at which the item counts as found; COCO classes use the floor of the heuristics
    threshold: float = OWL_THRESHOLD
    # a union item is found when any member clears its own threshold (the form "foreign object" has in #32)
    members: tuple[str, ...] = ()


FOREIGN_OBJECT_LABELS = ("X-15", "X-40", "X-46")
_th = Thresholds()
ITEMS = (
    Item("laptop", "coco", ("X-28",), LOOSE_CONF),
    Item("cup", "coco", FOREIGN_OBJECT_LABELS, _th.cup_conf),
    Item("cell phone", "coco", FOREIGN_OBJECT_LABELS, _th.phone_conf),
    Item("bottle", "coco", FOREIGN_OBJECT_LABELS, _th.bottle_conf),
    Item("foreign object", "coco", FOREIGN_OBJECT_LABELS, 1.0, ("cup", "cell phone", "bottle")),
    Item("book", "coco", ("X-24",), LOOSE_CONF),
    Item("key", "owl", ("X-11", "X-22")),
    Item("open safe", "owl", ("X-20",)),
    Item("open door", "owl", ("X-32", "X-33")),
)


@dataclass(frozen=True)
class ItemSummary:
    # frames with the item on all labels of the item's source (the ticket's label set)
    violation_frames_all: float
    violation_frames_target: float  # the same on the item's own target labels
    target_labels_hit: float  # target labels with the item in at least one window frame
    clean_frames: float  # frames of videos without violations with the item
    clean_videos_hit: float  # such videos with the item in at least one sampled frame
    passed: bool


def _share(scores: list[float], threshold: float) -> float:
    return sum(s >= threshold for s in scores) / len(scores) if scores else 0.0


def _pool(per_label: dict[str, list[float]]) -> list[float]:
    return [s for scores in per_label.values() for s in scores]


def label_set(source: str) -> list[str]:
    """Video ids of all labels the items of one source are judged on, in ADR order."""
    return list(dict.fromkeys(v for i in ITEMS if i.source == source for v in i.targets))


def summarize_item(
    item: Item, violation: dict[str, list[float]], clean: dict[str, list[float]]
) -> ItemSummary:
    """violation: best score of the item per window frame, by label's video id; clean: the same by video id."""
    t = item.threshold
    target = {v: s for v, s in violation.items() if v in item.targets}
    in_set = {v: s for v, s in violation.items() if v in label_set(item.source)}
    hit = sum(any(s >= t for s in target.get(v, [])) for v in item.targets) / len(item.targets)
    target_frames, clean_frames = _share(_pool(target), t), _share(_pool(clean), t)
    return ItemSummary(
        violation_frames_all=_share(_pool(in_set), t),
        violation_frames_target=target_frames,
        target_labels_hit=hit,
        clean_frames=clean_frames,
        clean_videos_hit=sum(any(s >= t for s in v) for v in clean.values()) / len(clean) if clean else 0.0,
        passed=bool(hit >= LABEL_GOAL and clean_frames < CLEAN_RATIO * target_frames),
    )


def matching_clean(
    item: Item, clean: dict[str, list[float]], arrangement: dict[str, str]
) -> dict[str, list[float]]:
    """Clean videos shot from the camera position of the item's target labels, so the camera is not the difference."""
    wanted = {arrangement[v] for v in item.targets}
    return {v: s for v, s in clean.items() if arrangement[v] in wanted}


def sweep(
    item: Item,
    violation: dict[str, list[float]],
    clean: dict[str, list[float]],
    thresholds: tuple[float, ...],
) -> list[tuple[float, ItemSummary]]:
    return [(t, summarize_item(replace(item, threshold=t), violation, clean)) for t in thresholds]


def union_score(item: Item, best: dict[str, float]) -> float:
    """Score of a union item: its members' best scores, each divided by its own floor (>= 1 means found)."""
    floors = {i.name: i.threshold for i in ITEMS}
    return max(best[m] / floors[m] for m in item.members)


def frame_scores(det: detectors.Detector, frames: list) -> list[dict[str, float]]:
    """Best score per item for each BGR frame of one size."""
    coco, owl = det.coco(frames), det.open_vocab(frames)
    out = []
    for c_boxes, o_boxes in zip(coco, owl, strict=True):
        best = dict.fromkeys((i.name for i in ITEMS), 0.0)
        for b in (*c_boxes, *o_boxes):
            if b.label in best:
                best[b.label] = max(best[b.label], b.conf)
        for item in ITEMS:
            if item.members:
                best[item.name] = union_score(item, best)
        out.append(best)
    return out


def score_video(det: detectors.Detector, path: Path, wanted: set[int]) -> dict[int, dict[str, float]]:
    """Scores of the sampled frames whose index is in wanted; frames are batched and dropped, so memory stays flat."""
    scores: dict[int, dict[str, float]] = {}
    batch: list[tuple[int, object]] = []

    def flush() -> None:
        for (i, _), s in zip(batch, frame_scores(det, [f for _, f in batch]), strict=True):
            scores[i] = s
        batch.clear()

    for i, (_, frame) in enumerate(iter_frames(path, FPS)):
        if i in wanted:
            batch.append((i, frame))
            if len(batch) == BATCH:
                flush()
    if batch:
        flush()
    return scores


def run() -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, list[float]]], dict[str, str]]:
    """(violation, clean, arrangement): item name -> video id -> best score per frame; video id -> camera position."""
    labels = load_label_table()
    det = detectors.LocalDetector(extra_queries=[i.name for i in ITEMS if i.source == "owl"])
    names = [i.name for i in ITEMS]
    violation: dict[str, dict[str, list[float]]] = {n: {} for n in names}
    clean: dict[str, dict[str, list[float]]] = {n: {} for n in names}
    wanted_labels = {v for source in ("coco", "owl") for v in label_set(source)}
    for (video_id, file), group in labels.groupby(["video_id", "file"], sort=False):
        is_clean = group["verdict"].eq("no_violation").all()
        if not is_clean and video_id not in wanted_labels:
            continue
        path = VIDEOS_DIR / file
        seconds = frame_seconds(path)
        if is_clean:
            wanted = {None: pick_frame_indices(len(seconds), CLEAN_FRAMES)}
        else:
            wanted = {int(s): in_window(seconds, int(s), WINDOW_S) for s in group["second"].dropna()}
        scores = score_video(det, path, {i for idx in wanted.values() for i in idx})
        dest = clean if is_clean else violation
        for idx in wanted.values():
            for n in names:
                dest[n].setdefault(video_id, []).extend(scores[i][n] for i in idx)
        print(f"  {video_id}: {len(seconds)} frames sampled, {len(scores)} scored", flush=True)
    arrangement = dict(
        labels.drop_duplicates("video_id")[["video_id", "arrangement"]].itertuples(index=False)
    )
    return violation, clean, arrangement


def report(violation: dict, clean: dict, arrangement: dict[str, str]) -> dict[str, dict]:
    matched = {i.name: matching_clean(i, clean[i.name], arrangement) for i in ITEMS}
    summaries = {i.name: summarize_item(i, violation[i.name], matched[i.name]) for i in ITEMS}

    def pct(x: float) -> str:
        return f"{x:.0%}"

    print(
        f"\n{'признак':<15}{'метки':<16}{'порог':>6}{'кадры наруш.':>13}{'на метках':>10}{'попад.':>7}"
        f"{'кадры чист.':>12}{'видео чист.':>12}{'passed':>8}"
    )
    for i in ITEMS:
        s = summaries[i.name]
        print(
            f"{i.name:<15}{','.join(i.targets):<16}{i.threshold:>6.2f}{pct(s.violation_frames_all):>13}"
            f"{pct(s.violation_frames_target):>10}{pct(s.target_labels_hit):>7}"
            f"{pct(s.clean_frames):>12}{pct(s.clean_videos_hit):>12}{str(s.passed):>8}"
        )
    print(
        "\nпо меткам: доля кадров окна с находкой (макс. уверенность; у foreign object в долях пола класса)"
    )
    for i in ITEMS:
        cells = []
        for v in i.targets:
            scores = violation[i.name].get(v, [])
            top = f"{max(scores, default=0.0):.2f}" + (" пола" if i.members else "")
            cells.append(f"{v} {_share(scores, i.threshold):.0%} ({top})")
        print(f"  {i.name:<15}" + "  ".join(cells))
    sweeps: dict[str, list[dict]] = {}
    print("\nпороги OWLv2: доля кадров меток / доля чистых кадров / макс. на чистых кадрах / passed")
    for i in ITEMS:
        if i.source != "owl":
            continue
        top = max(_pool(matched[i.name]), default=0.0)
        rows = sweep(i, violation[i.name], matched[i.name], OWL_SWEEP)
        sweeps[i.name] = [{"threshold": t} | asdict(s) for t, s in rows]
        cells = [
            f"{t:.2f}: {pct(s.violation_frames_target)}/{pct(s.clean_frames)}/{s.passed}" for t, s in rows
        ]
        print(f"  {i.name:<15}макс. {top:.2f}  " + "  ".join(cells))
    print(
        f"\nпорог находки: у COCO-классов пол эвристик (laptop и book {LOOSE_CONF}), у OWLv2 {OWL_THRESHOLD}; "
        "foreign object найден, когда любой из cup, cell phone, bottle выше своего пола. "
        f"Чистые видео только с той же раскладкой камеры, что у целевых меток. "
        f"passed: найден на ≥ {LABEL_GOAL:.0%} целевых меток и доля чистых кадров < {CLEAN_RATIO:.0%} доли кадров меток."
    )
    items = {}
    for i in ITEMS:
        any_arrangement = summarize_item(i, violation[i.name], clean[i.name])
        items[i.name] = asdict(summaries[i.name]) | {
            "threshold": i.threshold,
            "targets": list(i.targets),
            "n_target_frames": len(_pool({v: violation[i.name].get(v, []) for v in i.targets})),
            "n_clean_frames": len(_pool(matched[i.name])),
            "n_clean_videos": len(matched[i.name]),
            "clean_frames_any_arrangement": any_arrangement.clean_frames,
        }
    return {"items": items, "owl_sweep": sweeps}


if __name__ == "__main__":
    violation, clean, arrangement = run()
    params = report(violation, clean, arrangement)
    if "--no-log" not in sys.argv:
        append_row(
            NOTE,
            params
            | {
                "label_goal": LABEL_GOAL,
                "clean_ratio": CLEAN_RATIO,
                "window_s": WINDOW_S,
                "fps": FPS,
                "clean_frames_per_video": CLEAN_FRAMES,
            },
        )
