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
import sys
import warnings
from dataclasses import dataclass, replace
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

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
from src.models.errors import write_report
from src.models.evaluate import (
    Case,
    bank_windows,
    files_by_video,
    fit_settings,
    heuristic_hits,
    lovo,
    payout_sweep,
    run_case,
    summarize,
)
from src.models.violations import HEURISTIC_CONFIGS, peak_feature
from src.models.windows import (
    NEGATIVE,
    POSITIVE,
    TIMECODE_MODE,
    WINDOW_SEC,
    choose_threshold,
    group_of,
    labels_by_video,
    layout_folds,
    logit_shift,
    lovo_folds,
    make_windows,
    shift_threshold,
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


def _video_score(clf: BaseEstimator, clip: Clip) -> float:
    return video_score(clf.predict_proba(clip.feats)[:, 1]) if len(clip.feats) else 0.0


def _nested_fold(
    ids: list[str], clips: list[Clip], outer_train: np.ndarray, outer_val: np.ndarray
) -> list[dict]:
    """Hold out one group, do the whole procedure on the rest, score the held-out videos with the result."""
    with threadpool_limits(1):
        inner = [clips[i] for i in outer_train]
        inner_ids = [ids[i] for i in outer_train]
        inner_scores = np.zeros(len(inner))
        fold_models = []
        for train_idx, val_idx in lovo_folds(inner_ids):
            clf = fit_on([inner[i] for i in train_idx])
            fold_models.append(clf)
            for i in val_idx:
                inner_scores[i] = _video_score(clf, inner[i])
        threshold = choose_threshold(inner_scores, [c.is_violation for c in inner])
        full = fit_on(inner)
        return [
            {
                "video_id": ids[v],
                "is_violation": clips[v].is_violation,
                "threshold": threshold,
                "full_score": _video_score(full, clips[v]),
                # the scale the threshold was measured on: a typical fold model, averaged over the folds
                "fold_score": float(np.mean([_video_score(m, clips[v]) for m in fold_models])),
            }
            for v in outer_val
            if len(clips[v].feats)
        ]


LAYOUT_SUMMARY = ("video_recall", "fp_rate", "median_abs_dt", "type_accuracy")
LOW_TAIL = 5  # the violation videos with the lowest scores: the threshold lives there


def nested_check(ids: list[str], clips: list[Clip]) -> dict:
    """Does a threshold chosen on out-of-fold scores (ADR 0002) carry over to a model fit on all training videos?

    Per held-out video: threshold from LOVO on the other videos, a model fit on all the other videos (what
    window_clf.pkl is, one level up), verdict of that model vs that threshold. Also how far the two score scales
    sit apart, over all videos, over violation videos and over the LOW_TAIL weakest of them.
    """
    folds = Parallel(n_jobs=-1)(delayed(_nested_fold)(ids, clips, a, b) for a, b in lovo_folds(ids))
    rows = [r for fold in folds for r in fold]
    violation = np.array([r["is_violation"] for r in rows])
    threshold = np.array([r["threshold"] for r in rows])
    full = np.array([r["full_score"] for r in rows])
    fold = np.array([r["fold_score"] for r in rows])
    hit = full >= threshold
    shift = logit_shift(full, fold)
    weakest = np.flatnonzero(violation)[np.argsort(fold[violation])[:LOW_TAIL]]

    def spread(idx: np.ndarray | slice) -> dict:
        return {"median": float(np.median(shift[idx])), "mean_abs": float(np.abs(shift[idx]).mean())}

    # The saved threshold moves by the median shift of the weakest violation videos (never up).
    # The shift is estimated on all folds, so this recall is in-sample for the shift (not for the threshold).
    delta = min(spread(weakest)["median"], 0.0)
    shifted = np.array([shift_threshold(t, delta) for t in threshold])
    hit_shifted = full >= shifted

    return {
        "nested_video_recall": float(hit[violation].mean()),
        # same verdict with the fold models' own scores: separates the threshold rule from the score scale
        "nested_video_recall_fold_scores": float((fold >= threshold)[violation].mean()),
        "nested_fp_rate": float(hit[~violation].mean()),
        "threshold_shift_logit": delta,
        "nested_video_recall_shifted": float(hit_shifted[violation].mean()),
        "nested_fp_rate_shifted": float(hit_shifted[~violation].mean()),
        "nested_missed": [r["video_id"] for r, v, h in zip(rows, violation, hit, strict=True) if v and not h],
        "by_video": {r["video_id"]: {"score": r["full_score"], "threshold": r["threshold"]} for r in rows},
        "logit_shift_all": spread(slice(None)),
        "logit_shift_violation": spread(violation),
        "logit_shift_weakest_violation": spread(weakest),
    }


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


def _without(summary: dict, *keys: str) -> dict:
    return {k: v for k, v in summary.items() if k not in keys}


def _rescored(case: Case, clip: Clip, scores: np.ndarray) -> Case:
    return replace(case, scores=scores, peak_feature=peak_feature(scores, clip.feats))


def layout_check(
    ids: list[str], clips: list[Clip], cases: list[Case], banks: dict, layout_of: dict[str, str]
) -> dict:
    """Leave-one-layout-out: what happens on a furniture arrangement the model has never seen (#38).

    Per layout, the whole procedure runs on the other layouts only: the window model, the ADR 0002 threshold (from
    leave-one-video-out scores inside them), the heuristics' settings and kNN's bank. The held-out videos are scored
    by one model fit on all the other layouts, the way `window_clf.pkl` scores a new video. The camera is the same in
    every layout (FAQ 3), so this says nothing about a new camera angle (#43). `cases` carry the heuristics' hits;
    their window scores are replaced here.
    """
    layouts, seen, first = {}, [], []
    for name, train_idx, val_idx in layout_folds(ids, layout_of):
        rest = [clips[i] for i in train_idx]
        rest_scores = cross_validate([ids[i] for i in train_idx], rest)
        threshold = choose_threshold([video_score(s) for s in rest_scores], [c.is_violation for c in rest])
        full = fit_on(rest)
        held_scores = [
            full.predict_proba(clips[i].feats)[:, 1] if len(clips[i].feats) else np.zeros(0) for i in val_idx
        ]

        settings, _ = fit_settings(
            [_rescored(cases[i], clips[i], s) for i, s in zip(train_idx, rest_scores, strict=True)],
            threshold,
            {ids[i]: banks[ids[i]] for i in train_idx} if banks else None,
        )
        held = [_rescored(cases[i], clips[i], s) for i, s in zip(val_idx, held_scores, strict=True)]
        found = [run_case(c, threshold, settings) for c in held]
        fold_first = [f[0] if f else None for f in found]
        fold_seen = [c.view(settings.heuristics) for c in held]
        scores = [video_score(s) for s in held_scores]
        is_violation = [clips[i].is_violation for i in val_idx]
        timecodes = [
            timecode(clips[i].windows["start"], s, threshold)
            for i, s in zip(val_idx, held_scores, strict=True)
        ]
        branch_a = video_metrics(
            scores, is_violation, threshold, timecodes, [clips[i].label_seconds for i in val_idx]
        )
        branch_a["roc_auc"] = (
            float(roc_auc_score(is_violation, scores)) if len(set(is_violation)) == 2 else None
        )
        layouts[name] = {
            "n_videos": len(val_idx),
            "trained_on": " + ".join(sorted({layout_of[group_of(ids[i])] for i in train_idx})),
            "n_train_videos": len(train_idx),
            "threshold": threshold,
            **_without(summarize(fold_seen, fold_first), "timecode_choice"),
            "branch_a": branch_a,  # the window classifier alone, comparable with LOVO's branch_a
        }
        seen += fold_seen
        first += fold_first
    return {
        "layouts": layouts,
        "pooled": _without(summarize(seen, first), "timecode_choice", "verdicts_by_video"),
    }


def reference_embed_seconds(video_id: str, labels: pd.DataFrame) -> float:
    """Embedding time of the reference video measured now: a cached time was measured on whatever load the GPU
    had on the day, and the number is what the time budget is checked against."""
    path = video_path(files_by_video(labels)[video_id])
    return embed_video(path, Embedder())[2]


def run(nested: bool = True) -> dict:
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
    headline = {k: final[k] for k in LAYOUT_SUMMARY}
    layout_of = labels.drop_duplicates("video_id").set_index("video_id")["arrangement"].to_dict()
    metrics = (
        headline
        | {
            "lovo": headline,  # next to leave_one_layout_out, which is judged by the same decision on a held-out layout
            "leave_one_layout_out": layout_check(ids, clips, cases, banks, layout_of),
            "verdicts_by_video": final["verdicts_by_video"],  # the error table is built from these
        }
    )
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

    if nested:
        metrics["nested"] = nested_check(ids, clips)
    # window_clf.pkl holds a model fit on all videos: its threshold is the ADR 0002 one moved by the measured shift
    saved_threshold = (
        shift_threshold(threshold, metrics["nested"]["threshold_shift_logit"]) if nested else threshold
    )
    metrics["saved_threshold"] = saved_threshold
    if not nested:
        warnings.warn(
            f"--skip-nested: window_clf.pkl gets the unshifted threshold {threshold:.4f}; "
            "run without the flag before serving (ADR 0002)",
            stacklevel=2,
        )
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "model": fit_on(clips),
                "threshold": saved_threshold,  # the ADR 0002 one moved by the nested check (#63)
                "settings": settings,
                "embedding": FINGERPRINT,
                "window_sec": WINDOW_SEC,
            },
            f,
        )
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    write_report(metrics, labels)
    log_experiment("dinov2-small-logreg-lovo", metrics)
    log_layout_experiment(metrics)
    if nested:
        log_nested_experiment(metrics)
    return metrics


def log_experiment(note: str, metrics: dict) -> None:
    """Append one row. A file with an older, narrower header is widened once; rows are never dropped."""
    quiet = (
        "video_scores", "video_recall", "fp_rate", "median_abs_dt", "type_accuracy", "types", "knn", "nested",
        "lovo", "leave_one_layout_out", "verdicts_by_video",
    )  # fmt: skip
    params = {k: v for k, v in metrics.items() if k not in quiet}
    if "vote" in metrics:
        params["vote"] = {k: v for k, v in metrics["vote"].items() if k != "grid"}
    if "types" in metrics:
        params["types"] = metrics["types"]["allowed"]
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


def log_nested_experiment(metrics: dict) -> None:
    """A verdict-only row: recall and FP rate of the nested check, no timecodes or types."""
    nested = metrics["nested"]
    log_experiment(
        "dinov2-small-logreg-nested-threshold",
        {
            "video_recall": nested["nested_video_recall"],
            "fp_rate": nested["nested_fp_rate"],
            "median_abs_dt": None,
            "threshold": metrics["threshold"],
            **{k: v for k, v in nested.items() if k != "by_video"},
        },
    )


def log_layout_experiment(metrics: dict) -> None:
    """The pooled decision on held-out layouts, with each layout's recall and FP rate in params."""
    check = metrics["leave_one_layout_out"]
    log_experiment(
        "dinov2-small-logreg-leave-one-layout-out",
        {
            **{k: check["pooled"][k] for k in LAYOUT_SUMMARY},
            "layouts": {
                name: {k: v[k] for k in ("trained_on", "n_videos", "video_recall", "fp_rate", "threshold")}
                for name, v in check["layouts"].items()
            },
        },
    )


def load_model() -> dict:
    """Saved classifier + threshold; plain pickle of sklearn objects, no network."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def summary(out: dict) -> dict:
    """What the run prints: the numbers of the ticket and the decisions, not the per-video tables."""
    return {
        **{k: out[k] for k in ("video_recall", "fp_rate", "median_abs_dt", "type_accuracy", "threshold")},
        "saved_threshold": out["saved_threshold"],  # what window_clf.pkl holds (#63)
        "lovo": out["lovo"],
        "leave_one_layout_out": {
            "pooled": {k: out["leave_one_layout_out"]["pooled"][k] for k in LAYOUT_SUMMARY},
            **{
                name: {k: v[k] for k in ("trained_on", "n_videos", *LAYOUT_SUMMARY)}
                | {"roc_auc": v["branch_a"]["roc_auc"]}
                for name, v in out["leave_one_layout_out"]["layouts"].items()
            },
        },
        "nested": {
            k: v
            for k, v in out.get("nested", {}).items()
            if k.startswith("nested_video_recall") or k in ("nested_fp_rate", "threshold_shift_logit")
        },
        "branch_a": out["branch_a"],
        "vote": {k: v for k, v in out["vote"].items() if k != "grid"},
        "types_allowed": out["types"]["allowed"],
        "knn_enabled": out["knn"]["enabled"],
        "timecode_choice": out["timecode_choice"],
    }


if __name__ == "__main__":
    # The nested check refits ~1500 small models (about 3 min on 16 cores); --skip-nested leaves it out.
    print(json.dumps(summary(run(nested="--skip-nested" not in sys.argv)), indent=2, ensure_ascii=False))
