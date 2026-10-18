"""Check (TZ-5.2, #70): which new ADR 0005 items does the detector see from above?

D-FINE classes laptop, cup, cell phone, bottle, book and OWLv2 queries "key", "open safe", "open door" are scored on
the frames around the labels they are meant for and on frames of the videos without violations, as in #19.
Frames are decoded and scored locally with the production `LocalDetector`; nothing leaves the machine. Run:
    uv run python -m src.features.adr5_check [--no-log]
"""

import sys
from dataclasses import asdict, dataclass

from src.config import VIDEOS_DIR
from src.data.video import iter_frames
from src.features import detectors
from src.features.cash_check import CLEAN_FRAMES, FPS, in_window, load_label_table, pick_frame_indices
from src.features.experiments import append_row
from src.heuristics.thresholds import Thresholds

OWL_THRESHOLD = detectors.OWL_MIN_CONF
WINDOW_S = 3.0  # frames from [second - WINDOW_S, second + WINDOW_S] around a violation label
BATCH = 8
LABEL_GOAL = 0.70  # an item passes when it is found on at least this share of its target labels...
CLEAN_RATIO = 0.5  # ...and its share of clean frames is under this fraction of its share of target frames
NOTE = "adr5-check-dfine-owlv2"


@dataclass(frozen=True)
class Item:
    name: str  # a COCO class or an OWLv2 query
    source: str  # "coco" or "owl"
    targets: tuple[str, ...]  # the labels of ADR 0005 whose scenario the item should show
    threshold: float = (
        OWL_THRESHOLD  # confidence at which the item counts as found; COCO classes: the heuristics'
    )


_FOREIGN = ("X-15", "X-40", "X-46")
ITEMS = (
    Item("laptop", "coco", ("X-28",), 0.3),  # no floor in thresholds.py: the same as bottle's
    Item("cup", "coco", _FOREIGN, Thresholds().cup_conf),
    Item("cell phone", "coco", _FOREIGN, Thresholds().phone_conf),
    Item("bottle", "coco", _FOREIGN, Thresholds().bottle_conf),
    Item(
        "book", "coco", ("X-24",), 0.3
    ),  # book is off in thresholds.py; 0.3 is the low end of its clutter range
    Item("key", "owl", ("X-11", "X-22")),
    Item("open safe", "owl", ("X-20",)),
    Item("open door", "owl", ("X-32", "X-33")),
)


@dataclass(frozen=True)
class ItemSummary:
    violation_frames_all: (
        float  # frames with the item on all labels of the item's source (the ticket's label set)
    )
    violation_frames_target: float  # the same on the item's own target labels
    target_labels_hit: float  # target labels with the item in at least one window frame
    clean_frames: float  # frames of videos without violations with the item
    clean_videos_hit: float  # such videos with the item in at least one sampled frame
    passed: bool


def _share(scores: list[float], threshold: float) -> float:
    return sum(s >= threshold for s in scores) / len(scores) if scores else 0.0


def _pool(per_label: dict[str, list[float]]) -> list[float]:
    return [s for scores in per_label.values() for s in scores]


def summarize_item(
    item: Item, violation: dict[str, list[float]], clean: dict[str, list[float]]
) -> ItemSummary:
    """violation: best score of the item per window frame, by label's video id; clean: the same by video id."""
    t = item.threshold
    target = {v: s for v, s in violation.items() if v in item.targets}
    hit = sum(any(s >= t for s in target.get(v, [])) for v in item.targets) / len(item.targets)
    target_frames, clean_frames = _share(_pool(target), t), _share(_pool(clean), t)
    return ItemSummary(
        violation_frames_all=_share(
            _pool({v: s for v, s in violation.items() if v in label_set(item.source)}), t
        ),
        violation_frames_target=target_frames,
        target_labels_hit=hit,
        clean_frames=clean_frames,
        clean_videos_hit=sum(any(s >= t for s in v) for v in clean.values()) / len(clean) if clean else 0.0,
        passed=bool(hit >= LABEL_GOAL and clean_frames < CLEAN_RATIO * target_frames),
    )


def label_set(source: str) -> list[str]:
    """Video ids of all labels the items of one source are judged on, in ADR order."""
    return list(dict.fromkeys(v for i in ITEMS if i.source == source for v in i.targets))


def frame_scores(det, frames: list) -> list[dict[str, float]]:
    """Best score per item for each BGR frame of one size."""
    coco, owl = det.coco(frames), det.open_vocab(frames)
    out = []
    for c_boxes, o_boxes in zip(coco, owl, strict=True):
        best = dict.fromkeys((i.name for i in ITEMS), 0.0)
        for b in (*c_boxes, *o_boxes):
            if b.label in best:
                best[b.label] = max(best[b.label], b.conf)
        out.append(best)
    return out


def score_video(det, path, wanted: set[int]) -> dict[int, dict[str, float]]:
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


def run() -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, list[float]]]]:
    """(violation, clean): item name -> video id -> best score per frame."""
    labels = load_label_table()
    # the queries are read when the detector loads, so they go in before it
    detectors.EXTRA_QUERIES = tuple(i.name for i in ITEMS if i.source == "owl")
    det = detectors.LocalDetector()
    names = [i.name for i in ITEMS]
    violation: dict[str, dict[str, list[float]]] = {n: {} for n in names}
    clean: dict[str, dict[str, list[float]]] = {n: {} for n in names}
    wanted_labels = {v for source in ("coco", "owl") for v in label_set(source)}
    for (video_id, file), group in labels.groupby(["video_id", "file"], sort=False):
        is_clean = group["verdict"].eq("no_violation").all()
        if not is_clean and video_id not in wanted_labels:
            continue
        path = VIDEOS_DIR / file
        seconds = [sec for sec, _ in iter_frames(path, FPS)]
        if is_clean:
            wanted = {None: pick_frame_indices(len(seconds), CLEAN_FRAMES)}
        else:
            wanted = {int(s): in_window(seconds, int(s), WINDOW_S) for s in group["second"].dropna()}
        scores = score_video(det, path, {i for idx in wanted.values() for i in idx})
        target = clean if is_clean else violation
        for idx in wanted.values():
            for n in names:
                target[n].setdefault(video_id, []).extend(scores[i][n] for i in idx)
        print(f"  {video_id}: {len(seconds)} frames sampled, {len(scores)} scored", flush=True)
    return violation, clean


def report(violation: dict, clean: dict) -> dict[str, dict]:
    summaries = {i.name: summarize_item(i, violation[i.name], clean[i.name]) for i in ITEMS}
    pct = lambda x: f"{x:.0%}"  # noqa: E731
    print(
        f"\n{'признак':<12}{'метки':<16}{'порог':>6}{'кадры наруш.':>13}{'на метках':>10}{'попад.':>7}"
        f"{'кадры чист.':>12}{'видео чист.':>12}{'passed':>8}"
    )
    for i in ITEMS:
        s = summaries[i.name]
        print(
            f"{i.name:<12}{','.join(i.targets):<16}{i.threshold:>6.1f}{pct(s.violation_frames_all):>13}"
            f"{pct(s.violation_frames_target):>10}{pct(s.target_labels_hit):>7}"
            f"{pct(s.clean_frames):>12}{pct(s.clean_videos_hit):>12}{str(s.passed):>8}"
        )
    print("\nпо меткам: доля кадров окна с находкой (макс. уверенность)")
    for i in ITEMS:
        cells = []
        for v in i.targets:
            scores = violation[i.name].get(v, [])
            cells.append(f"{v} {_share(scores, i.threshold):.0%} ({max(scores, default=0.0):.2f})")
        print(f"  {i.name:<12}" + "  ".join(cells))
    print(
        f"\nпорог находки: у COCO-классов пол эвристик (laptop и book 0.3), у OWLv2 {OWL_THRESHOLD}. "
        f"passed: найден на ≥ {LABEL_GOAL:.0%} целевых меток и доля чистых кадров < {CLEAN_RATIO:.0%} доли кадров меток."
    )
    return {
        i.name: asdict(summaries[i.name])
        | {
            "threshold": i.threshold,
            "targets": list(i.targets),
            "n_target_frames": len(_pool({v: violation[i.name].get(v, []) for v in i.targets})),
            "n_clean_frames": len(_pool(clean[i.name])),
        }
        for i in ITEMS
    }


if __name__ == "__main__":
    violation, clean = run()
    items = report(violation, clean)
    if "--no-log" not in sys.argv:
        append_row(
            NOTE,
            {
                "items": items,
                "label_goal": LABEL_GOAL,
                "clean_ratio": CLEAN_RATIO,
                "window_s": WINDOW_S,
                "fps": FPS,
                "clean_frames_per_video": CLEAN_FRAMES,
                "n_clean_videos": len(clean[ITEMS[0].name]),
            },
        )
