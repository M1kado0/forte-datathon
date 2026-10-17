"""Branch A: DINOv2 window embeddings -> logistic regression, leave-one-video-out CV, ADR 0002 threshold.

    uv run python -m src.models.train

Writes reports/metrics.json, models/window_clf.pkl and one row to experiments.csv.
"""

from __future__ import annotations

import csv
import functools
import json
import pickle
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import EXPERIMENTS, MODELS, REPORTS, SEED
from src.data.labels import load_labels
from src.features.embeddings import (
    EMBED_FPS,
    FINGERPRINT,
    Embedder,
    load_or_embed,
    video_path,
    window_features,
)
from src.models.windows import (
    NEGATIVE,
    POSITIVE,
    WINDOW_SEC,
    choose_threshold,
    labels_by_video,
    lovo_folds,
    make_windows,
    timecode,
    video_metrics,
    video_score,
)

MODEL_PATH = MODELS / "window_clf.pkl"
LOGREG_C = 0.1
REFERENCE_DURATION = 145.0  # the ticket records embedding time for a 145 s video
EXPERIMENT_COLUMNS = [
    "ts", "note", "roc_auc", "pr_auc", "brier", "n_train", "n_valid", "params",
    "video_recall", "fp_rate", "median_abs_dt", "type_accuracy",
]  # fmt: skip


@dataclass
class Clip:
    """One video: windows with labels, their pooled embeddings, and how long embedding took."""

    windows: pd.DataFrame
    feats: np.ndarray
    embed_seconds: float
    duration: float
    label_seconds: list[int]

    @property
    def is_violation(self) -> bool:
        return bool(self.label_seconds)


def make_classifier() -> BaseEstimator:
    """Scaler lives inside the pipeline, so cross-validation fits it on train folds only."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=LOGREG_C, class_weight="balanced", max_iter=2000, random_state=SEED),
    )


def fit_on(clips: list[Clip]) -> BaseEstimator:
    """Fit on the labelled (positive / negative) windows of the given clips; unlabeled windows are skipped."""
    X = np.concatenate([c.feats for c in clips])
    y = np.concatenate([c.windows["label"].to_numpy() for c in clips])
    labelled = np.isin(y, (POSITIVE, NEGATIVE))
    return make_classifier().fit(X[labelled], y[labelled])


def build_dataset(labels: pd.DataFrame) -> tuple[list[str], list[Clip]]:
    by_video = labels_by_video(labels)
    files = labels.drop_duplicates("video_id").set_index("video_id")["file"]
    ids = sorted(by_video, key=lambda v: int(v.split("-")[1]))
    embedder = functools.cache(Embedder)  # built once, and only if some video is not cached yet
    clips = []
    for vid in ids:
        seconds, emb, elapsed = load_or_embed(vid, video_path(files[vid]), embedder)
        duration = float(seconds[-1]) + 1.0 / EMBED_FPS if len(seconds) else 0.0
        windows = make_windows(duration, by_video[vid])
        feats, kept = window_features(seconds, emb, windows)
        clips.append(
            Clip(windows[kept].reset_index(drop=True), feats[kept], elapsed, duration, by_video[vid])
        )
    return ids, clips


def cross_validate(ids: list[str], clips: list[Clip]) -> list[np.ndarray]:
    """Out-of-fold score of every window of every video (unlabeled windows are scored, not trained on)."""
    scores = [np.zeros(len(c.windows)) for c in clips]
    for train_idx, val_idx in lovo_folds(ids):
        clf = fit_on([clips[i] for i in train_idx])
        for i in val_idx:
            if len(clips[i].feats):
                scores[i] = clf.predict_proba(clips[i].feats)[:, 1]
    return scores


def run() -> dict:
    ids, clips = build_dataset(load_labels())
    scores = cross_validate(ids, clips)
    video_scores = [video_score(s) for s in scores]
    is_violation = [c.is_violation for c in clips]
    # Threshold and metrics share the out-of-fold scores: recall 1.0 is by construction (ADR 0002).
    threshold = choose_threshold(video_scores, is_violation)
    timecodes = [timecode(c.windows["start"], s, threshold) for c, s in zip(clips, scores, strict=True)]
    metrics = video_metrics(
        video_scores, is_violation, threshold, timecodes, [c.label_seconds for c in clips]
    )

    ref = min(range(len(clips)), key=lambda i: abs(clips[i].duration - REFERENCE_DURATION))
    metrics |= {
        "threshold": threshold,
        "n_videos": len(ids),
        "n_violation_videos": sum(is_violation),
        "n_false_positives": sum(s >= threshold and not v for s, v in zip(video_scores, is_violation, strict=True)),
        "embed_seconds_ref_video": {
            "video_id": ids[ref], "duration": clips[ref].duration, "seconds": clips[ref].embed_seconds,
        },
        "video_scores": dict(zip(ids, video_scores, strict=True)),
    }  # fmt: skip

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "model": fit_on(clips),
                "threshold": threshold,
                "embedding": FINGERPRINT,
                "window_sec": WINDOW_SEC,
            },
            f,
        )
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    log_experiment("dinov2-small-logreg-lovo", metrics)
    return metrics


def log_experiment(note: str, metrics: dict) -> None:
    """Append one row. A file with an older, narrower header is widened once; rows are never dropped."""
    params = {
        k: v
        for k, v in metrics.items()
        if k not in ("video_scores", "video_recall", "fp_rate", "median_abs_dt")
    }
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "note": note,
        "params": json.dumps(params | {"logreg_c": LOGREG_C}, ensure_ascii=False),
        "video_recall": metrics["video_recall"],
        "fp_rate": metrics["fp_rate"],
        "median_abs_dt": metrics["median_abs_dt"],
    }
    kept = []
    if EXPERIMENTS.exists():
        with open(EXPERIMENTS, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames == EXPERIMENT_COLUMNS:
                kept = None  # header is current: plain append below
            else:
                kept = list(reader)
    if kept is None:
        with open(EXPERIMENTS, "a", newline="") as f:
            csv.DictWriter(f, EXPERIMENT_COLUMNS).writerow(row)
        return
    with open(EXPERIMENTS, "w", newline="") as f:
        w = csv.DictWriter(f, EXPERIMENT_COLUMNS)
        w.writeheader()
        w.writerows([*kept, row])


def load_model() -> dict:
    """Saved classifier + threshold; plain pickle of sklearn objects, no network."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    out = run()
    print(json.dumps({k: v for k, v in out.items() if k != "video_scores"}, indent=2, ensure_ascii=False))
