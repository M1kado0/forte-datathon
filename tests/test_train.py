"""Saved model loads without network; experiment log widens an old header."""

import csv
import pickle

import numpy as np

from src.models import train


def test_saved_model_loads_and_scores_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    y = np.array([0, 1] * 10)
    path = tmp_path / "m.pkl"
    path.write_bytes(pickle.dumps({"model": train.make_classifier().fit(X, y), "threshold": 0.5}))
    monkeypatch.setattr(train, "MODEL_PATH", path)
    art = train.load_model()
    assert art["model"].predict_proba(X).shape == (20, 2)


def test_log_experiment_widens_old_header(tmp_path, monkeypatch):
    log = tmp_path / "experiments.csv"
    log.write_text("ts,note,roc_auc,pr_auc,brier,n_train,n_valid,params\n2026-01-01T00:00:00,old,,,,,,{}\n")
    monkeypatch.setattr(train, "EXPERIMENTS", log)
    train.log_experiment("new", {"video_recall": 1.0, "fp_rate": 0.5, "median_abs_dt": 3.0, "threshold": 0.1})
    rows = list(csv.DictReader(log.open()))
    assert [r["note"] for r in rows] == ["old", "new"]
    assert rows[1]["video_recall"] == "1.0" and rows[0]["fp_rate"] == ""
