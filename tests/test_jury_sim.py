"""Pure helpers of the jury simulation; no network, no ffmpeg."""

import pytest

from scripts.jury_sim import percentile, pick_clips, schema_problem


def test_percentile_is_nearest_rank():
    assert percentile([10, 1, 5, 7, 3], 0.95) == 10
    assert percentile(list(range(1, 21)), 0.95) == 19
    assert percentile([4.0], 0.95) == 4.0
    with pytest.raises(ValueError):
        percentile([], 0.95)


def test_schema_problem_accepts_only_finished_valid_jobs():
    assert (
        schema_problem({"job_id": "a", "status": "done", "verdict": "no_violation", "violations": []}) == ""
    )
    assert schema_problem({"job_id": "a", "status": "failed"}) == "status failed"
    assert schema_problem(
        {"job_id": "a", "status": "done", "verdict": "violation", "violations": []}
    ).startswith("schema")
    assert schema_problem(
        {"job_id": "a", "status": "done", "verdict": "no_violation", "extra": 1}
    ).startswith("schema")


def test_pick_clips_cycles_through_available_files(tmp_path):
    for name in ("b.mov", "a.mov"):
        (tmp_path / name).touch()
    assert [clip.name for clip in pick_clips(tmp_path, 3)] == ["a.mov", "b.mov", "a.mov"]
    with pytest.raises(SystemExit):
        pick_clips(tmp_path / "missing", 1)
