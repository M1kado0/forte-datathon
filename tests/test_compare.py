"""Held-out data boundaries and validation; no learned model scores."""

import pandas as pd
import pytest

from src.models import compare


def inputs(tmp_path, monkeypatch, **changes):
    organizer = tmp_path / "labels.csv"
    organizer.write_text("video_id,file\nX-1,X-1.mov\n")
    own = tmp_path / "labels_self.csv"
    row = dict(video_id="own", file="own.mov", verdict="violation", second=3, violation_type="other")
    pd.DataFrame([row | changes]).to_csv(own, index=False)
    monkeypatch.setattr(compare, "LABELS_CSV", organizer)
    monkeypatch.setattr(compare, "SELF_LABELS_CSV", own)
    monkeypatch.setattr(compare, "SELF_VIDEOS_DIR", tmp_path)
    monkeypatch.setattr("src.data.video.duration", lambda path: 10.0)


def test_self_labels_allow_own_names_without_arrangements(tmp_path, monkeypatch):
    inputs(tmp_path, monkeypatch)
    assert compare.read_holdout().video_id.tolist() == ["own"]


@pytest.mark.parametrize("changes", [{"video_id": "X-1"}, {"file": "X-1.mov"}])
def test_self_labels_cannot_overlap_training(tmp_path, monkeypatch, changes):
    inputs(tmp_path, monkeypatch, **changes)
    with pytest.raises(ValueError, match="overlaps"):
        compare.read_holdout()


@pytest.mark.parametrize(
    "changes",
    [
        {"second": -1},
        {"second": 10},
        {"violation_type": "unknown"},
        {"second": None},
        {"file": "../outside.mov"},
    ],
)
def test_invalid_holdout_is_rejected_before_training(tmp_path, monkeypatch, changes):
    inputs(tmp_path, monkeypatch, **changes)
    with pytest.raises(ValueError):
        compare.read_holdout()
