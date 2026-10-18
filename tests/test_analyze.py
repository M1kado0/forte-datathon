"""analyze(path) seam: video -> violations that satisfy the wire contract and the ticket's invariants (#28).

Fast tests use a fake embedder (frame brightness) and a fake classifier (brightness / 255), so bright
seconds of a synthetic clip become score peaks. One smoke test runs the real DINOv2 + saved classifier
on the shortest real clip.
"""

import base64
import functools
from itertools import combinations

import cv2
import numpy as np
import pandas as pd
import pytest

from contracts.data import TRACKS_SCHEMA
from contracts.schemas import JobResponse
from src.config import VIDEOS_DIR, WEIGHTS
from src.features.embeddings import MODEL_NAME
from src.models import analyze as analyze_mod
from src.models.analyze import FALLBACK_TYPE, MAX_VIOLATIONS, MIN_GAP_SEC, analyze
from src.models.train import MODEL_PATH
from src.models.violations import HEURISTIC_CONFIGS, HEURISTIC_TYPES, Settings, Vote
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
    assert len(violations) <= MAX_VIOLATIONS
    if violations and all(v.type == FALLBACK_NAME for v in violations):
        assert len(violations) == 1  # ADR 0005: a fallback-typed result is one violation per video
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


OUTSIDER = "outsider_in_cash_room"
OUTSIDER_NAME = next(t["name"] for t in rule_catalog() if t["id"] == OUTSIDER)


def people_tracks(second=(2.0, 12.0)):
    """The cashier at the seat all clip and a second person beside the seat between `second` seconds."""

    def person(track_id, t0, t1, cy):
        return [
            (track_id, "person", "person", i / 4, 180.0, cy - 100.0, 220.0, cy + 100.0, 0.9)
            for i in range(round(t0 * 4), round(t1 * 4))
        ]

    rows = person(1, 0, SECONDS, 400.0) + person(2, *second, 520.0)
    return TRACKS_SCHEMA.validate(pd.DataFrame(rows, columns=list(TRACKS_SCHEMA.columns)))


def run_with_tracks(tmp_path, peaks, settings, table):
    model = {**FAKE_MODEL, "settings": settings}
    clip = make_clip(tmp_path / "clip.mov", peaks)
    return analyze(clip, embedder=brightness_embedder, model=model, tracks=lambda _: table)


VOTING = Settings(
    allowed=frozenset({OUTSIDER}), vote=Vote(enabled=True, n_sec=3.0, types=frozenset({OUTSIDER}))
)


def test_heuristic_type_at_the_start_of_the_state_when_branch_a_agrees(tmp_path):
    violations = run_with_tracks(tmp_path, {7: BRIGHT}, VOTING, people_tracks())
    assert_contract(violations)
    assert [v.type for v in violations] == [OUTSIDER_NAME]
    assert violations[0].timecode_sec == pytest.approx(2.0, abs=0.2)


def test_voting_hit_raises_a_verdict_branch_a_did_not_give(tmp_path):
    violations = run_with_tracks(tmp_path, {}, VOTING, people_tracks())
    assert_contract(violations)
    assert [v.type for v in violations] == [OUTSIDER_NAME]


def test_vote_switched_off_leaves_the_verdict_to_branch_a(tmp_path):
    no_vote = Settings(allowed=frozenset({OUTSIDER}), vote=Vote(enabled=False))
    assert run_with_tracks(tmp_path, {}, no_vote, people_tracks()) == []


def test_a_model_without_settings_never_builds_tracks(tmp_path):
    def forbidden(_):
        raise AssertionError("branch B must not run")

    clip = make_clip(tmp_path / "clip.mov", {7: BRIGHT})
    violations = analyze(clip, embedder=brightness_embedder, model=FAKE_MODEL, tracks=forbidden)
    assert [v.type for v in violations] == [FALLBACK_NAME]


def test_failed_tracks_leave_the_answer_to_branch_a(tmp_path):
    def broken(_):
        raise RuntimeError("out of memory")

    clip = make_clip(tmp_path / "clip.mov", {7: BRIGHT})
    model = {**FAKE_MODEL, "settings": VOTING}
    violations = analyze(clip, embedder=brightness_embedder, model=model, tracks=broken)
    assert_contract(violations)
    assert [v.type for v in violations] == [FALLBACK_NAME]


def test_model_from_other_embeddings_is_refused(monkeypatch):
    monkeypatch.setattr(analyze_mod, "load_model", lambda: {"embedding": "old", "window_sec": 2.0})
    analyze_mod.default_model.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="rerun"):
            analyze_mod.default_model()
    finally:
        analyze_mod.default_model.cache_clear()


def shortest_real_clip(limit_sec: float = 30.0):
    """The shortest dataset clip of at most limit_sec, read from container headers; None without videos."""
    best = None
    for path in sorted(VIDEOS_DIR.glob("*.mov")):
        cap = cv2.VideoCapture(str(path))
        fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps > 0 and frames / fps <= limit_sec and (best is None or frames / fps < best[0]):
            best = (frames / fps, path)
    return best[1] if best else None


REAL_CLIP = shortest_real_clip()


@pytest.mark.skipif(
    not (WEIGHTS / MODEL_NAME).exists() or not MODEL_PATH.exists() or REAL_CLIP is None,
    reason="needs `make weights`, models/window_clf.pkl and a ≤ 30 s clip in data/raw/videos",
)
def test_smoke_real_model_on_shortest_real_clip(monkeypatch):
    """DINOv2 + saved classifier on a real camera clip (native size, h264); no score is asserted.

    This is branch A, the smoke test of #28: the saved heuristic settings are left out, so no tracks are built
    (the real detector run is checked by `make jury-sim`, and takes about 1 s more than this fits in). With the
    saved threshold the contract holds whatever the verdict. Threshold 0 flags every video, so the violation path
    (evidence frames, type, description) also runs on real frames. The clip is embedded once and shared by both
    calls, which keeps the test under the 5 s limit of AGENTS.md.
    """
    monkeypatch.setattr(analyze_mod, "embed_video", functools.cache(analyze_mod.embed_video))
    branch_a = {**analyze_mod.default_model(), "settings": None}
    assert_contract(analyze(REAL_CLIP, model=branch_a))
    forced = analyze(REAL_CLIP, model={**branch_a, "threshold": 0.0})
    assert len(forced) == 1
    assert_contract(forced)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="needs models/window_clf.pkl")
def test_saved_settings_load_and_name_known_things():
    settings = analyze_mod.default_model()["settings"]
    assert isinstance(settings, Settings)
    assert settings.heuristics in HEURISTIC_CONFIGS
    assert settings.allowed <= set(HEURISTIC_TYPES)
    assert settings.vote.types <= set(HEURISTIC_TYPES)
