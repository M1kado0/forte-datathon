"""Branch A: DINOv2 window embeddings -> logistic regression, leave-one-video-out CV, ADR 0002 threshold.

    uv run python -m src.models.train

Then the type choice and the heuristics' vote (#34) are judged on the same out-of-fold scores: heuristics run on
the cached tracks of every video, their settings are fitted on the other clips and applied to the held-out one
(src/models/evaluate.py). Writes reports/metrics.json, models/window_clf.pkl and one row to experiments.csv.
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
    embed_video,
    load_or_embed,
    video_path,
    window_features,
)
from src.models.evaluate import (
    Case,
    bank_windows,
    files_by_video,
    fit_settings,
    heuristic_hits,
    lovo,
    payout_sweep,
)
from src.models.violations import HEURISTIC_CONFIGS, peak_feature
from src.models.windows import (
    NEGATIVE,
    POSITIVE,
    TIMECODE_MODE,
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
    files = files_by_video(labels)
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


def build_cases(
    labels: pd.DataFrame, ids: list[str], clips: list[Clip], scores: list[np.ndarray]
) -> tuple[list[Case], dict]:
    """One Case per video for the type and vote evaluation, and the kNN bank (positive windows by type)."""
    files = files_by_video(labels)
    hits_by = {name: heuristic_hits({v: files[v] for v in ids}, th) for name, th in HEURISTIC_CONFIGS.items()}
    cases, banks = [], {}
    for vid, clip, s in zip(ids, clips, scores, strict=True):
        rows = labels[(labels["video_id"] == vid) & (labels["verdict"] == "violation")]
        truth = [(int(sec), t) for sec, t in zip(rows["second"], rows["violation_type"], strict=True)]
        by_config = {name: hits[vid] for name, hits in hits_by.items()}
        cases.append(
            Case(
                vid,
                clip.windows["start"].to_numpy(),
                s,
                peak_feature(s, clip.feats),
                by_config["default"],
                truth,
                by_config,
            )
        )
        banks[vid] = bank_windows(clip.windows, clip.feats, truth)
    return cases, banks


def reference_embed_seconds(video_id: str, labels: pd.DataFrame) -> float:
    """Embedding time of the reference video measured now: a cached time was measured on whatever load the GPU
    had on the day, and the number is what the time budget is checked against."""
    path = video_path(files_by_video(labels)[video_id])
    return embed_video(path, Embedder())[2]


def run() -> dict:
    labels = load_labels()
    ids, clips = build_dataset(labels)
    scores = cross_validate(ids, clips)
    video_scores = [video_score(s) for s in scores]
    is_violation = [c.is_violation for c in clips]
    # Threshold and metrics share the out-of-fold scores: recall 1.0 is by construction (ADR 0002).
    threshold = choose_threshold(video_scores, is_violation)

    def metrics_for(mode: str) -> dict:
        timecodes = [
            timecode(c.windows["start"], s, threshold, mode) for c, s in zip(clips, scores, strict=True)
        ]
        return video_metrics(
            video_scores, is_violation, threshold, timecodes, [c.label_seconds for c in clips]
        )

    branch_a = metrics_for(TIMECODE_MODE)

    cases, banks = build_cases(labels, ids, clips, scores)
    settings, notes = fit_settings(cases, threshold, banks)
    final = lovo(cases, threshold, banks)
    metrics = {k: final[k] for k in ("video_recall", "fp_rate", "median_abs_dt", "type_accuracy")}
    metrics |= {
        "branch_a": branch_a,  # the window classifier alone, as before #34
        "vote": {
            "enabled": settings.vote.enabled,  # kept only if the OR adds no false positive (ADR 0005)
            "n_sec": settings.vote.n_sec,
            "types": sorted(settings.vote.types),
            **notes["vote"],
        },
        "types": {
            "heuristics": notes["heuristics"],  # which HEURISTIC_CONFIGS the saved model uses, and why
            "allowed": sorted(settings.allowed),  # heuristic types that may name a violation
            "per_type": notes["per_type"],
            "n_typed_right": final["n_typed_right"],
            "n_named": final["n_named"],
            "by_video": final["types_by_video"],
            "id_not_presented_sweep": payout_sweep(labels, ids),
        },
        "knn": {"enabled": settings.knn is not None, **notes["knn"]},
        "timecode_choice": final["timecode_choice"],
    }

    ref = min(range(len(clips)), key=lambda i: abs(clips[i].duration - REFERENCE_DURATION))
    metrics |= {
        "timecode_mode": TIMECODE_MODE,  # median_abs_dt measures the timecode analyze() reports
        # the mode is chosen by median |Δt| on LOVO (docs/acceptance.md, criterion 2)
        "median_abs_dt_by_mode": {m: metrics_for(m)["median_abs_dt"] for m in ("onset", "peak")},
        "threshold": threshold,
        "n_videos": len(ids),
        "n_violation_videos": sum(is_violation),
        "n_clean_videos": final["n_clean_videos"],
        "n_false_positives": final["n_false_positives"],
        "embed_seconds_ref_video": {
            "video_id": ids[ref], "duration": clips[ref].duration, "seconds": reference_embed_seconds(ids[ref], labels),
        },
        "video_scores": dict(zip(ids, video_scores, strict=True)),
    }  # fmt: skip

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "model": fit_on(clips),
                "threshold": threshold,
                "settings": settings,
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
    quiet = ("video_scores", "video_recall", "fp_rate", "median_abs_dt", "type_accuracy", "types", "knn")
    params = {k: v for k, v in metrics.items() if k not in quiet}
    params["vote"] = {k: v for k, v in metrics.get("vote", {}).items() if k != "grid"}
    params["types"] = (metrics.get("types") or {}).get("allowed")
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "note": note,
        "params": json.dumps(params | {"logreg_c": LOGREG_C}, ensure_ascii=False),
        "video_recall": metrics["video_recall"],
        "fp_rate": metrics["fp_rate"],
        "median_abs_dt": metrics["median_abs_dt"],
        "type_accuracy": metrics.get("type_accuracy"),
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


def summary(out: dict) -> dict:
    """What the run prints: the numbers of the ticket and the decisions, not the per-video tables."""
    return {
        **{k: out[k] for k in ("video_recall", "fp_rate", "median_abs_dt", "type_accuracy", "threshold")},
        "branch_a": out["branch_a"],
        "vote": {k: v for k, v in out["vote"].items() if k != "grid"},
        "types_allowed": out["types"]["allowed"],
        "knn_enabled": out["knn"]["enabled"],
        "timecode_choice": out["timecode_choice"],
    }


if __name__ == "__main__":
    print(json.dumps(summary(run()), indent=2, ensure_ascii=False))
