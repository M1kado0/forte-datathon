"""Branch A: DINOv2 window embeddings -> logistic regression, leave-one-video-out CV, ADR 0002 threshold.

    uv run python -m src.models.train

Writes reports/metrics.json, models/window_clf.pkl and one row to experiments.csv.
"""

from __future__ import annotations

import csv
import functools
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import EXPERIMENTS, MODELS, REPORTS, SEED
from src.data.labels import load_labels
from src.features.embeddings import Embedder, load_or_embed, video_path, window_features
from src.models.windows import (
    NEGATIVE,
    POSITIVE,
    choose_threshold,
    labels_by_video,
    lovo_folds,
    make_windows,
    timecode,
    video_metrics,
    video_score,
)

MODEL_PATH = MODELS / "window_clf.pkl"
C = 0.1
EXPERIMENT_COLUMNS = ["ts", "note", "roc_auc", "pr_auc", "brier", "n_train", "n_valid", "params"]
REFERENCE_DURATION = 145.0  # the ticket records embedding time for a 145 s video


def make_classifier() -> object:
    """Scaler lives inside the pipeline, so cross-validation fits it on train folds only."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=SEED),
    )


def build_dataset(labels: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """Per video: windows with labels, window features, frame seconds and embed time."""
    by_video = labels_by_video(labels)
    files = labels.drop_duplicates("video_id").set_index("video_id")["file"]
    ids = sorted(by_video, key=lambda v: int(v.split("-")[1]))
    embedder = functools.cache(Embedder)  # built once, and only if some video is not cached yet
    clips = []
    for vid in ids:
        seconds, emb, elapsed = load_or_embed(vid, video_path(files[vid]), embedder)
        duration = float(seconds[-1]) + 1.0 / 4 if len(seconds) else 0.0
        windows = make_windows(duration, by_video[vid])
        feats, kept = window_features(seconds, emb, windows)
        clips.append(
            {
                "windows": windows[kept].reset_index(drop=True),
                "feats": feats[kept],
                "elapsed": elapsed,
                "duration": duration,
                "labels": by_video[vid],
            }
        )
    return ids, clips


def cross_validate(ids: list[str], clips: list[dict]) -> list[np.ndarray]:
    """Out-of-fold score of every window of every video (unlabeled windows are scored, not trained on)."""
    scores = [np.zeros(len(c["windows"])) for c in clips]
    for train_idx, val_idx in lovo_folds(ids):
        X = np.concatenate([clips[i]["feats"] for i in train_idx])
        y = np.concatenate([clips[i]["windows"]["label"].to_numpy() for i in train_idx])
        use = np.isin(y, (POSITIVE, NEGATIVE))
        clf = make_classifier().fit(X[use], y[use])
        for i in val_idx:
            if len(clips[i]["feats"]):
                scores[i] = clf.predict_proba(clips[i]["feats"])[:, 1]
    return scores


def run() -> dict:
    labels = load_labels()
    ids, clips = build_dataset(labels)
    scores = cross_validate(ids, clips)
    video_scores = [video_score(s) for s in scores]
    is_violation = [bool(c["labels"]) for c in clips]
    threshold = choose_threshold(video_scores, is_violation)
    timecodes = [timecode(c["windows"]["start"], s, threshold) for c, s in zip(clips, scores, strict=True)]
    metrics = video_metrics(video_scores, is_violation, threshold, timecodes, [c["labels"] for c in clips])

    ref = min(range(len(clips)), key=lambda i: abs(clips[i]["duration"] - REFERENCE_DURATION))
    metrics |= {
        "threshold": threshold,
        "n_videos": len(ids),
        "n_violation_videos": int(sum(is_violation)),
        "n_false_positives": int(
            sum(s >= threshold and not v for s, v in zip(video_scores, is_violation, strict=True))
        ),
        "embed_seconds_ref_video": {
            "video_id": ids[ref],
            "duration": clips[ref]["duration"],
            "seconds": clips[ref]["elapsed"],
        },
        "video_scores": dict(zip(ids, video_scores, strict=True)),
    }

    X = np.concatenate([c["feats"] for c in clips])
    y = np.concatenate([c["windows"]["label"].to_numpy() for c in clips])
    use = np.isin(y, (POSITIVE, NEGATIVE))
    final = make_classifier().fit(X[use], y[use])
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "model": final,
                "threshold": threshold,
                "embedding": "dinov2-small cls+meanpatch, 4 fps, 2 s windows",
            },
            f,
        )
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    log_experiment("dinov2-small-logreg-lovo", metrics)
    return metrics


def log_experiment(note: str, metrics: dict) -> None:
    new = not EXPERIMENTS.exists()
    params = {k: v for k, v in metrics.items() if k != "video_scores"} | {"C": C}
    with open(EXPERIMENTS, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(EXPERIMENT_COLUMNS)
        w.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                note,
                "",
                "",
                "",
                "",
                "",
                json.dumps(params, ensure_ascii=False),
            ]
        )


def load_model() -> dict:
    """Saved classifier + threshold; plain pickle of sklearn objects, no network."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    out = run()
    print({k: v for k, v in out.items() if k != "video_scores"})
