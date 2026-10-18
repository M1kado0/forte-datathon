"""Deterministic sequencing, temporal boundaries, grouping and metric checks."""

import numpy as np
import pytest

from src.features.actions import Proposals, resize_frame, sample_indices, window_starts
from src.models.action_pipeline import Clip, classify_events, targets
from src.models.action_train import summarize


def proposals():
    return Proposals(
        np.arange(4.0),
        np.arange(4.0) + 2,
        np.arange(8.0).reshape(4, 2),
        np.array([0, 0, 1, 1]),
        np.ones(4) * 0.5,
        ("counting", "walking"),
        0.0,
        5.0,
    )


class TypeSpy:
    def __init__(self, type_id="other"):
        self.inputs = []
        self.type_id = type_id

    def predict(self, x):
        self.inputs.append(x.copy())
        return np.array([self.type_id] * len(x))


def test_negative_actions_never_reach_type_classifier():
    model = TypeSpy()
    assert classify_events(proposals(), np.zeros(4), {"type": model}, 0.5) == []
    assert model.inputs == []


def test_only_accepted_action_event_peaks_are_typed():
    model = TypeSpy()
    events = classify_events(proposals(), np.array([0.1, 0.8, 0.2, 0.3]), {"type": model}, 0.5)
    np.testing.assert_array_equal(model.inputs[0], [[2, 3]])
    assert events[0]["timecode_sec"] == 2.0
    assert events[0]["action"] == "counting"


def test_adjacent_same_action_merges_before_type_prediction():
    model = TypeSpy()
    events = classify_events(proposals(), np.array([0.6, 0.9, 0.1, 0.1]), {"type": model}, 0.5)
    assert len(model.inputs[0]) == 1
    assert (events[0]["start_sec"], events[0]["end_sec"]) == (0, 3)
    assert events[0]["timecode_sec"] == 2


def test_object_description_cannot_be_invented_from_action_label():
    events = classify_events(proposals(), np.ones(4), {"type": TypeSpy("foreign_object_at_desk")}, 0.5)
    assert events[0]["type"] == "other"
    assert events[0]["raw_type"] == "foreign_object_at_desk"


def test_reported_events_are_limited_and_separated_in_time():
    p = Proposals(
        np.array([0.0, 8.0, 16.0]),
        np.array([2.0, 10.0, 18.0]),
        np.ones((3, 2)),
        np.array([0, 1, 2]),
        np.ones(3),
        ("a", "b", "c"),
        0.0,
        18.0,
    )
    events = classify_events(p, np.array([0.8, 0.9, 0.7]), {"type": TypeSpy()}, 0.5)
    assert len(events) == 2
    assert [e["timecode_sec"] for e in events] == [9.0, 1.0]


def test_short_video_and_fractional_tail_are_covered():
    assert window_starts(0.5).tolist() == [0.0]
    assert window_starts(4.3).tolist() == pytest.approx([0, 1, 2, 2.3])
    for value in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            window_starts(value)


def test_sampling_never_uses_a_frame_after_target_or_out_of_bounds():
    assert sample_indices(np.array([0, 0.25, 0.5]), 0, 1, 4).tolist() == [0, 1, 2, 2]


def test_letterbox_preserves_whole_frame_and_rgb_order():
    frame = np.zeros((4, 8, 3), dtype=np.uint8)
    frame[:, :, 2] = 255
    image = resize_frame(frame, 8)
    assert image.shape == (8, 8, 3)
    assert (image[2:6, :, 0] == 255).all()
    assert (image[:2] == 0).all()


def test_unannotated_windows_are_not_assumed_negative_after_event():
    clip = Clip("X-1", proposals(), [(1, "other")])
    y, types = targets(clip)
    assert y.tolist() == [1, 1, 1, 1]
    assert set(types) == {"other"}


def test_metrics_match_hand_counted_events():
    clips = [
        Clip("X-1", proposals(), [(2, "other")]),
        Clip("X-2", proposals(), [(2, "other")]),
        Clip("X-3", proposals(), []),
    ]
    event = {"type": "other", "timecode_sec": 3.0}
    result = summarize(clips, [[event], [], [event]], [0.8, 0.1, 0.9])
    assert result["video_recall"] == 0.5
    assert result["fp_rate"] == 1
    assert result["median_abs_dt"] == 1
    assert result["type_accuracy"] == 0.5


def test_action_serving_is_opt_in_and_warmed(monkeypatch):
    from api.analyzer import load_analyzer

    monkeypatch.setenv("FORTE_ACTION_MODEL", "r3d-18")
    loaded = []
    marker = object()

    def factory(name):
        loaded.append(name)
        return marker

    monkeypatch.setattr("src.models.action_pipeline.ActionAnalyzer", factory)
    assert load_analyzer() is marker
    assert loaded == ["r3d-18"]


def test_nested_fold_excludes_held_video_from_heads_and_threshold(monkeypatch):
    from src.models import action_train

    clips = [Clip(f"X-{i}", proposals(), [(2, "other")] if i < 3 else []) for i in range(1, 4)]
    ids = {id(c.proposals): c.video_id for c in clips}
    trained, calibrated = [], []

    def fit(rows):
        seen = {c.video_id for c in rows}
        trained.append(seen)
        return {"ids": seen, "type": TypeSpy()}

    def calibrate(rows):
        calibrated.append({c.video_id for c in rows})
        return 0.5

    def score(p, heads):
        assert ids[id(p)] not in heads["ids"]
        return np.full(len(p.starts), 0.8)

    monkeypatch.setattr(action_train, "fit_heads", fit)
    monkeypatch.setattr(action_train, "calibrate", calibrate)
    monkeypatch.setattr(action_train, "binary_scores", score)
    action_train.evaluate_cv(clips)
    assert trained == calibrated
    assert len(trained) == 3 and all(len(group) == 2 for group in trained)
