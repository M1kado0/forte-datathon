"""`analyze(path)`: video path -> violations (TZ-4.4). The verdict is `violation` iff the list is non-empty.

    uv run python -m src.models.analyze data/raw/videos/X-14.mov

Branch A scores 2 s windows with the saved classifier (src/models/train.py). Branch B tracks people, objects and
cash and runs the type heuristics (src/heuristics) when the saved settings let them name a type or vote. What is
reported, of which type and at which second is decided in src/models/violations.py: the verdict is a window at
or above the threshold OR a long enough hit of a voting type, the type comes from a heuristic, then kNN, then the
fallback (ADR 0005), and the timecode is the start of a state or the peak of an action.
"""

from __future__ import annotations

import functools
import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from contracts.schemas import Violation
from src.data.evidence import evidence_frames
from src.features.detectors import LocalDetector
from src.features.embeddings import EMBED_FPS, FINGERPRINT, Embedder, embed_video, window_features
from src.features.tracks import build_tracks
from src.heuristics import detect
from src.models.train import load_model
from src.models.violations import (
    FALLBACK_TYPE,
    HEURISTIC_CONFIGS,
    MAX_VIOLATIONS,
    MIN_GAP_SEC,
    Settings,
    decide,
    peak_feature,
)
from src.models.windows import WINDOW_SEC, make_windows
from src.rules.describe import describe_violation

__all__ = ["FALLBACK_TYPE", "MAX_VIOLATIONS", "MIN_GAP_SEC", "analyze"]

log = logging.getLogger(__name__)


@functools.cache
def default_embedder() -> Embedder:
    return Embedder()


@functools.cache
def default_detector() -> LocalDetector:
    return LocalDetector()


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


def default_tracks(path: Path) -> pd.DataFrame:
    """Tracks of an uploaded video, built and dropped with the job: nothing of an upload is cached."""
    return build_tracks(path, default_detector())


def analyze(
    path: str | Path,
    *,
    embedder: Callable[[list[np.ndarray]], np.ndarray] | None = None,
    model: dict | None = None,
    tracks: Callable[[Path], pd.DataFrame] | None = None,
) -> list[Violation]:
    """Violations in one video, strongest first; [] means no violation. The upload is read, never cached."""
    path = Path(path)
    model = model or default_model()
    settings: Settings = model.get("settings") or Settings()
    seconds, emb, _ = embed_video(path, embedder or default_embedder())
    starts, scores, feats = score_windows(seconds, emb, model["model"])
    hits = []
    if settings.allowed or settings.vote.enabled:  # a model saved before #34 has neither: branch A alone
        try:
            hits = detect((tracks or default_tracks)(path), thresholds=HEURISTIC_CONFIGS[settings.heuristics])
        except Exception:  # noqa: BLE001 - heuristics only raise the verdict, so branch A must still answer
            log.exception("branch B failed on %s, answering from branch A alone", path.name)
    found = decide(starts, scores, model["threshold"], hits, settings, peak_feature(scores, feats))
    violations = []
    for f in found[:MAX_VIOLATIONS]:
        frames = evidence_frames(path, f.timecode_sec, f.kind)
        name, description = describe_violation(f.type_id, f.facts)
        # report the second of the frame actually decoded, so timecode and first frame are equal
        violations.append(
            Violation(
                type=name,
                timecode_sec=frames[0].timestamp_sec,
                description=description,
                evidence_frames=frames,
            )
        )
    return violations


def score_windows(seconds: np.ndarray, emb: np.ndarray, clf) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(window starts, violation probability, window features) over the video, windows built as in training."""
    duration = float(seconds[-1]) + 1.0 / EMBED_FPS if len(seconds) else 0.0
    windows = make_windows(duration)
    feats, kept = window_features(seconds, emb, windows)
    starts = windows["start"].to_numpy()[kept]
    scores = clf.predict_proba(feats[kept])[:, 1] if kept.any() else np.zeros(0)
    return starts, scores, feats[kept]


def main(argv: list[str]) -> None:
    start = time.perf_counter()
    (
        default_embedder(),
        default_model(),
        default_detector(),
    )  # warm the caches, so load and analysis are timed apart
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
