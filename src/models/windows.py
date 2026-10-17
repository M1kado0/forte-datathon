"""Window labelling, leave-one-video-out folds, threshold (ADR 0002) and metrics. Pure functions, no model."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from statistics import median

import numpy as np
import pandas as pd

from src.data.labels import DUPLICATES

WINDOW_SEC = 2.0
STEP_SEC = 0.5
POS_BEFORE = 1.0  # a window overlapping [N - POS_BEFORE, N + POS_AFTER] is positive
POS_AFTER = 4.0
NEG_BEFORE = 3.0  # a window ending at or before N - NEG_BEFORE is negative

POSITIVE, NEGATIVE, UNLABELED = 1, 0, -1


def make_windows(duration: float, labels: Sequence[float] = ()) -> pd.DataFrame:
    """Sliding windows over one video: columns start, end, label (POSITIVE / NEGATIVE / UNLABELED).

    Positive: overlaps [N-1, N+4] of any label N. Negative: ends at or before N-3 of every label N
    (so also nothing after the first label's N+4: lasting states are not trusted as negatives).
    No labels: negative over the full length. Everything else is UNLABELED.
    """
    n = int((duration - WINDOW_SEC) / STEP_SEC + 1e-9) + 1 if duration >= WINDOW_SEC else 0
    start = np.arange(n) * STEP_SEC
    end = start + WINDOW_SEC
    positive = np.zeros(n, dtype=bool)
    negative = np.ones(n, dtype=bool)
    for second in labels:
        positive |= (start < second + POS_AFTER) & (end > second - POS_BEFORE)
        negative &= end <= second - NEG_BEFORE
    label = np.where(positive, POSITIVE, np.where(negative, NEGATIVE, UNLABELED))
    return pd.DataFrame({"start": start, "end": end, "label": label})


def labels_by_video(labels: pd.DataFrame) -> dict[str, list[int]]:
    """data/labels.csv rows -> {video_id: [label seconds]}; videos without violation map to []."""
    out: dict[str, list[int]] = defaultdict(list)
    for video_id, second in zip(labels["video_id"], labels["second"], strict=True):
        out.setdefault(video_id, [])
        if not pd.isna(second):
            out[video_id].append(int(second))
    return dict(out)


def group_of(video_id: str) -> str:
    """CV group of a clip: merged duplicates (X-47 -> X-4) share one group."""
    return DUPLICATES.get(video_id, video_id)


def lovo_folds(video_ids: Sequence[str]) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-video-out: yield (train_idx, val_idx) per group; one group never sits on both sides."""
    groups = np.array([group_of(v) for v in video_ids])
    for g in sorted(set(groups)):
        val = groups == g
        yield np.flatnonzero(~val), np.flatnonzero(val)


def video_score(window_scores: Sequence[float]) -> float:
    """A video is as suspicious as its worst window."""
    return float(np.max(window_scores))


def choose_threshold(scores: Sequence[float], is_violation: Sequence[bool]) -> float:
    """ADR 0002: recall 1.0 on violation videos first, then the fewest false positives.

    Verdict is `score >= threshold`. False positives only fall as the threshold rises, so the
    highest threshold that still flags every violation video is the lowest violation video score.
    """
    s = np.asarray(scores, dtype=float)
    v = np.asarray(is_violation, dtype=bool)
    if not v.any():
        raise ValueError("need at least one violation video to choose a threshold")
    return float(s[v].min())


def timecode(
    starts: Sequence[float], scores: Sequence[float], threshold: float, mode: str = "onset"
) -> float | None:
    """Reported second: 'onset' = start of the first above-threshold window, 'peak' = centre of the best one."""
    s = np.asarray(scores, dtype=float)
    t = np.asarray(starts, dtype=float)
    above = np.flatnonzero(s >= threshold)
    if above.size == 0:
        return None
    if mode == "onset":
        return float(t[above[0]])
    if mode == "peak":
        return float(t[int(np.argmax(s))] + WINDOW_SEC / 2)
    raise ValueError(f"unknown mode: {mode!r}")


def video_metrics(
    scores: Sequence[float],
    is_violation: Sequence[bool],
    threshold: float,
    timecodes: Sequence[float | None],
    label_seconds: Sequence[Sequence[float]],
) -> dict:
    """Per-video lists -> video recall, FP rate on clean videos, median |Δt| to the nearest label.

    |Δt| is taken over flagged violation videos with a timecode only.
    """
    s = np.asarray(scores, dtype=float)
    v = np.asarray(is_violation, dtype=bool)
    flagged = s >= threshold
    deltas = [
        min(abs(tc - n) for n in ns)
        for tc, ns, bad, hit in zip(timecodes, label_seconds, v, flagged, strict=True)
        if bad and hit and tc is not None and ns
    ]
    return {
        "video_recall": float(flagged[v].mean()) if v.any() else None,
        "fp_rate": float(flagged[~v].mean()) if (~v).any() else None,
        "median_abs_dt": float(median(deltas)) if deltas else None,
    }
