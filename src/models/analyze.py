"""Branch A end to end: video path -> violations (TZ-4.4). The verdict is `violation` iff the list is non-empty.

    uv run python -m src.models.analyze data/raw/videos/X-14.mov

Window scores come from the saved classifier (src/models/train.py). Each violation is a peak of the score
above the threshold; at most MAX_VIOLATIONS, at least MIN_GAP_SEC apart. Until branch B types it, every
violation gets the fallback type, its description from the rules reference and frames around the peak.
"""

from __future__ import annotations

import functools
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from contracts.schemas import Violation
from src.data.evidence import evidence_frames
from src.features.embeddings import EMBED_FPS, FINGERPRINT, Embedder, embed_video, window_features
from src.models.train import load_model
from src.models.windows import WINDOW_SEC, make_windows, timecode
from src.rules.describe import describe_violation

MAX_VIOLATIONS = 2
MIN_GAP_SEC = 5.0
FALLBACK_TYPE = "other"  # rules.yaml `fallback: true`; branch B replaces it with a heuristic type
EVIDENCE_KIND = "action"  # the timecode is a score peak, so frames are the peak, 1 s before and 1 s after


@functools.cache
def default_embedder() -> Embedder:
    return Embedder()


@functools.cache
def default_model() -> dict:
    """Saved classifier + threshold, refused when it was trained on other embeddings or windows."""
    model = load_model()
    if model.get("embedding") != FINGERPRINT or model.get("window_sec") != WINDOW_SEC:
        raise RuntimeError(
            f"models/window_clf.pkl was trained on {model.get('embedding')!r} with "
            f"{model.get('window_sec')} s windows, expected {FINGERPRINT!r} with {WINDOW_SEC} s: "
            "rerun `uv run python -m src.models.train`"
        )
    return model


def analyze(
    path: str | Path,
    *,
    embedder: Callable[[list[np.ndarray]], np.ndarray] | None = None,
    model: dict | None = None,
) -> list[Violation]:
    """Violations in one video, strongest first; [] means no violation. The upload is read, never cached."""
    path = Path(path)
    model = model or default_model()
    seconds, emb, _ = embed_video(path, embedder or default_embedder())
    starts, scores = score_windows(seconds, emb, model["model"])
    violations: list[Violation] = []
    for tc in pick_timecodes(starts, scores, model["threshold"]):
        frames = evidence_frames(path, tc, EVIDENCE_KIND)
        # report the second of the frame actually decoded, so timecode and first frame are equal
        second = frames[0].timestamp_sec
        if any(abs(second - v.timecode_sec) < MIN_GAP_SEC for v in violations):
            continue  # snapping to a frame pulled two peaks closer than the gap
        name, description = describe_violation(FALLBACK_TYPE, {})
        violations.append(
            Violation(type=name, timecode_sec=second, description=description, evidence_frames=frames)
        )
    return violations


def score_windows(seconds: np.ndarray, emb: np.ndarray, clf) -> tuple[np.ndarray, np.ndarray]:
    """(window starts, violation probability) over the video, windows built as in training."""
    duration = float(seconds[-1]) + 1.0 / EMBED_FPS if len(seconds) else 0.0
    windows = make_windows(duration)
    feats, kept = window_features(seconds, emb, windows)
    starts = windows["start"].to_numpy()[kept]
    scores = clf.predict_proba(feats[kept])[:, 1] if kept.any() else np.zeros(0)
    return starts, scores


def pick_timecodes(
    starts: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    max_n: int = MAX_VIOLATIONS,
    min_gap: float = MIN_GAP_SEC,
) -> list[float]:
    """Peak seconds of above-threshold windows, best first; each next peak is min_gap from those before."""
    starts = np.asarray(starts, dtype=float)
    left = np.asarray(scores, dtype=float).copy()
    centres = starts + WINDOW_SEC / 2
    out: list[float] = []
    while len(out) < max_n:
        tc = timecode(starts, left, threshold, mode="peak")
        if tc is None:
            break
        out.append(tc)
        left[np.abs(centres - tc) < min_gap] = -np.inf
    return out


def main(argv: list[str]) -> None:
    start = time.perf_counter()
    default_embedder(), default_model()
    loaded = time.perf_counter()
    violations = analyze(argv[0])
    done = time.perf_counter()
    summary = {
        "file": argv[0],
        "verdict": "violation" if violations else "no_violation",
        "violations": [
            {"type": v.type, "timecode_sec": round(v.timecode_sec, 2), "frames": len(v.evidence_frames)}
            for v in violations
        ],
        "load_seconds": round(loaded - start, 1),
        "analyze_seconds": round(done - loaded, 1),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
