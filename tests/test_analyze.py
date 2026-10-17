"""analyze(path) seam: video -> violations that satisfy the wire contract and the ticket's invariants (#28).

Fast tests use a fake embedder (frame brightness) and a fake classifier (brightness / 255), so bright
seconds of a synthetic clip become score peaks. One smoke test runs the real DINOv2 + saved classifier.
"""

import base64
from itertools import combinations

import cv2
import numpy as np
import pytest

from contracts.schemas import JobResponse
from src.config import WEIGHTS
from src.features.embeddings import MODEL_NAME
from src.models import analyze as analyze_mod
from src.models.analyze import (
    FALLBACK_TYPE,
    MAX_FALLBACK_VIOLATIONS,
    MAX_VIOLATIONS,
    MIN_GAP_SEC,
    analyze,
    pick_timecodes,
)
from src.models.train import MODEL_PATH
from src.rules.describe import rule_catalog

SRC_FPS = 10
SECONDS = 20
W, H = 96, 64
DARK, BRIGHT, DIM = 20, 255, 220
# A 2 s window holding one second at BRIGHT scores ~0.54, at DIM ~0.47; half a bright second ~0.31.
THRESHOLD = 0.4
CATALOG = {t["name"]: t for t in rule_catalog()}
FALLBACK_NAME = next(t["name"] for t in rule_catalog() if t["id"] == FALLBACK_TYPE)


def make_clip(path, peaks=None):
    """`peaks` maps a whole second to its brightness; every other second is DARK."""
    peaks = peaks or {}
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), SRC_FPS, (W, H))
    for i in range(SRC_FPS * SECONDS):
        writer.write(np.full((H, W, 3), peaks.get(i // SRC_FPS, DARK), dtype=np.uint8))
    writer.release()
    return path


def brightness_embedder(frames):
    return np.array([[f.mean()] for f in frames], dtype=np.float32)


class BrightnessClassifier:
    def predict_proba(self, X):
        p = np.clip(X[:, 0] / 255.0, 0, 1)
        return np.column_stack([1 - p, p])


FAKE_MODEL = {"model": BrightnessClassifier(), "threshold": THRESHOLD}


def run(tmp_path, peaks):
    return analyze(make_clip(tmp_path / "clip.mov", peaks), embedder=brightness_embedder, model=FAKE_MODEL)


def assert_contract(violations):
    """Every invariant of the ticket's smoke test and ADR 0005, for any verdict."""
    verdict = "violation" if violations else "no_violation"
    JobResponse(job_id="t", status="done", verdict=verdict, violations=violations)  # raises on mismatch
    only_fallback = all(v.type == FALLBACK_NAME for v in violations)
    assert len(violations) <= (MAX_FALLBACK_VIOLATIONS if only_fallback else MAX_VIOLATIONS)
    for a, b in combinations(violations, 2):
        assert abs(a.timecode_sec - b.timecode_sec) >= MIN_GAP_SEC
    for v in violations:
        assert 1 <= len(v.evidence_frames) <= 3
        assert v.evidence_frames[0].timestamp_sec == v.timecode_sec
        for frame in v.evidence_frames:
            raw = base64.b64decode(frame.image_base64)
            assert raw[:2] == b"\xff\xd8"
            assert cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) is not None
        assert v.type in CATALOG
        # the reference's citation of each document (and пункт, where the type has one) closes the text
        assert v.description.endswith(CATALOG[v.type]["citation_suffix"])
        assert all(rule["title"] in v.description for rule in CATALOG[v.type]["rules"])


@pytest.mark.parametrize(
    ("peaks", "expected_sec"),
    [({4: DIM, 14: BRIGHT}, 14.5), ({4: BRIGHT, 14: DIM}, 4.5)],
    ids=["later-stronger", "earlier-stronger"],
)
def test_fallback_result_is_one_violation_at_the_strongest_peak(tmp_path, peaks, expected_sec):
    violations = run(tmp_path, peaks)
    assert_contract(violations)
    assert [v.type for v in violations] == [FALLBACK_NAME]
    assert violations[0].timecode_sec == pytest.approx(expected_sec, abs=0.6)


def test_no_window_above_threshold_is_no_violation(tmp_path):
    violations = run(tmp_path, {})
    assert violations == []
    assert_contract(violations)


def test_pick_timecodes_takes_best_peak_then_next_one_past_the_gap():
    starts = np.arange(20) * 0.5  # window centres 1.0, 1.5, ..., 10.5
    scores = np.zeros(20)
    scores[[2, 4, 14]] = [0.6, 0.9, 0.7]  # centres 2.0, 3.0, 8.0
    assert pick_timecodes(starts, scores, threshold=0.5) == [3.0, 8.0]
    assert pick_timecodes(starts, scores, threshold=0.5, max_n=1) == [3.0]
    assert pick_timecodes(starts, scores, threshold=0.95) == []
    assert pick_timecodes(np.zeros(0), np.zeros(0), threshold=0.5) == []


def test_pick_timecodes_stops_at_max_violations():
    starts = np.arange(40) * 0.5
    scores = np.zeros(40)
    scores[[2, 16, 30]] = [0.9, 0.8, 0.7]  # centres 2.0, 9.0, 16.0, all past the gap
    assert MAX_VIOLATIONS == 2
    assert pick_timecodes(starts, scores, threshold=0.5) == [2.0, 9.0]


def test_model_from_other_embeddings_is_refused(monkeypatch):
    monkeypatch.setattr(analyze_mod, "load_model", lambda: {"embedding": "old", "window_sec": 2.0})
    analyze_mod.default_model.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="rerun"):
            analyze_mod.default_model()
    finally:
        analyze_mod.default_model.cache_clear()


@pytest.mark.skipif(
    not (WEIGHTS / MODEL_NAME).exists() or not MODEL_PATH.exists(),
    reason="needs `make weights` and models/window_clf.pkl from `uv run python -m src.models.train`",
)
def test_smoke_real_model_on_short_clip(tmp_path):
    """DINOv2 + saved classifier on a 20 s clip: whatever the verdict, the result keeps the contract."""
    assert_contract(analyze(make_clip(tmp_path / "clip.mov", {4: BRIGHT, 14: BRIGHT})))
