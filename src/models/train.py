"""Baseline: LightGBM -> isotonic calibration on a holdout -> artifacts + metrics + experiments.csv row."""

from __future__ import annotations

import csv
import json
import pickle
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import EXPERIMENTS, MODELS, REPORTS, SEED, TARGET
from src.features.build import to_model_frame

PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1,
)


def train(df: pd.DataFrame, note: str = "baseline") -> dict:
    # ASSUMPTION: random split for the stub. Real data -> time-based or grouped split.
    tr, va = train_test_split(df, test_size=0.25, random_state=SEED, stratify=df[TARGET])
    Xtr, Xva = to_model_frame(tr), to_model_frame(va)
    ytr, yva = tr[TARGET].to_numpy(), va[TARGET].to_numpy()

    model = lgb.LGBMClassifier(**PARAMS).fit(Xtr, ytr)
    raw_va = model.predict_proba(Xva)[:, 1]
    calib = IsotonicRegression(out_of_bounds="clip").fit(raw_va, yva)
    p_va = calib.predict(raw_va)

    metrics = {
        "roc_auc": round(float(roc_auc_score(yva, p_va)), 4),
        "pr_auc": round(float(average_precision_score(yva, p_va)), 4),
        "brier": round(float(brier_score_loss(yva, p_va)), 4),
        "base_rate": round(float(yva.mean()), 4),
        "n_train": int(len(tr)),
        "n_valid": int(len(va)),
    }

    with open(MODELS / "model.pkl", "wb") as f:
        pickle.dump({"model": model, "calibrator": calib, "metrics": metrics}, f)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    _log_experiment(note, metrics)
    return metrics


def _log_experiment(note: str, metrics: dict) -> None:
    new = not EXPERIMENTS.exists()
    with open(EXPERIMENTS, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "note", "roc_auc", "pr_auc", "brier", "n_train", "n_valid", "params"])
        w.writerow([
            datetime.now().isoformat(timespec="seconds"), note, metrics["roc_auc"], metrics["pr_auc"],
            metrics["brier"], metrics["n_train"], metrics["n_valid"], json.dumps(PARAMS),
        ])


def load_artifacts() -> dict | None:
    path = MODELS / "model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_with_reasons(art: dict, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated probability + per-feature contributions (LightGBM native SHAP, last column = bias)."""
    raw = art["model"].predict_proba(X)[:, 1]
    p = np.clip(art["calibrator"].predict(raw), 0.01, 0.99)  # never show 0% or 100%
    contrib = art["model"].predict(X, pred_contrib=True)
    return p, contrib[:, :-1]
