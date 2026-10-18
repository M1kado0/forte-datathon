"""Jury simulation checks with an in-memory HTTP transport; no network or ffmpeg."""

import argparse
import asyncio

import httpx
import pytest

from scripts.jury_sim import Outcome, main, percentile, pick_clips, schema_problem, stress_failures


@pytest.mark.parametrize(
    ("health_status", "health_body", "error"),
    [
        (200, {"status": "ok", "model_loaded": True}, None),
        (200, {"status": "ok", "model_loaded": False}, "model_loaded"),
        (200, {"status": "ok"}, "model_loaded"),
        (200, {"status": "ok", "model_loaded": None}, "model_loaded"),
        (200, {"status": "ok", "model_loaded": "true"}, "model_loaded"),
        (200, {"status": "ok", "model_loaded": 1}, "model_loaded"),
        (200, {"status": "loading", "model_loaded": True}, "status ok"),
        (503, {"status": "ok", "model_loaded": True}, "503"),
        (200, [], "status ok"),
        (200, "<html>tunnel unavailable</html>", "status ok"),
    ],
)
def test_jury_sim_requires_ready_model_before_uploading(
    monkeypatch, tmp_path, capsys, health_status, health_body, error
):
    requests = []

    def respond(request):
        requests.append(request.url.path)
        if request.url.path == "/api/health":
            if isinstance(health_body, str):
                return httpx.Response(health_status, text=health_body)
            return httpx.Response(health_status, json=health_body)
        if request.url.path == "/api/v1/jobs/none":
            return httpx.Response(401)
        if request.url.path == "/api/v1/analyze":
            return httpx.Response(200, json={"job_id": "test-job"})
        return httpx.Response(
            200,
            json={"job_id": "test-job", "status": "done", "verdict": "no_violation", "violations": []},
        )

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: client_type(**kwargs, transport=httpx.MockTransport(respond))
    )
    (tmp_path / "sample.mov").write_bytes(b"synthetic fixture")
    args = argparse.Namespace(
        url="http://jury.test", api_key="test-key", stress=0, parallel=1, videos=tmp_path, timeout=1
    )
    result = asyncio.run(main(args))
    if error:
        assert result == 1
        assert "/api/v1/analyze" not in requests
        assert error in capsys.readouterr().out
    else:
        assert result == 0
        assert "/api/v1/analyze" in requests


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


def test_stress_judges_solo_clip_time_not_queue_wait():
    batch = [Outcome(f"s{i}.mov", 22.0 * (i + 1), True) for i in range(8)]  # one worker: the last waits 176 s
    assert stress_failures(solo_sec=20.2, wall_sec=177.3, outcomes=batch) == []


def test_stress_fails_on_slow_solo_clip_slow_batch_or_error():
    batch = [Outcome("s0.mov", 10.0, False, "timeout after 360 s")]
    problems = stress_failures(solo_sec=50.0, wall_sec=310.0, outcomes=batch)
    assert len(problems) == 3
    assert any("50.0" in text and "45" in text for text in problems)
    assert any("310.0" in text and "300" in text for text in problems)
