"""Deterministic HTTP contract tests; no model or dataset required."""

import base64
import logging
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from functools import lru_cache, partial
from io import BytesIO
from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from api.analyzer import analyze_stub, load_analyzer
from api.main import create_app
from contracts.schemas import JobResponse
from src.rules.describe import describe_violation, rule_catalog

KEY = {"X-API-Key": "jury-key"}
CONTEXT = multiprocessing.get_context("spawn")


@dataclass
class ScenarioAnalyzer:
    """Inject behavior at the analyzer boundary, including across process restarts."""

    paths: object
    outcome: str = "stub"
    release: object = None
    started: object = None
    frame_time: float | None = None
    calls: int = 0

    def __call__(self, path):
        self.paths.put(path)
        self.calls += 1
        if self.started is not None:
            self.started.set()
            assert self.release.wait(2)
        if self.calls == 1:
            if self.outcome == "exception":
                raise RuntimeError("Do not expose this error")
            if self.outcome == "invalid":
                return [{"type": "invalid"}]
        if self.frame_time is not None:
            violation = analyze_stub(path)[0].model_dump()
            violation["timecode_sec"] = 44.0
            violation["evidence_frames"][0]["timestamp_sec"] = self.frame_time
            return [violation]
        return analyze_stub(path) if self.outcome == "stub" else []


def hang_once(path, marker, crash):
    if not marker.exists():
        marker.write_text(str(path), encoding="utf-8")
        if crash:
            os._exit(1)
        while True:
            time.sleep(1)
    return []


def load_supplied_analyzer(analyzer):
    return analyzer


def test_health_tracks_worker_crash_and_reload(tmp_path):
    analyzer = partial(hang_once, marker=tmp_path / "crashed", crash=True)
    app = create_app(api_key="jury-key", analyzer_loader=partial(load_supplied_analyzer, analyzer))
    with TestClient(app) as client:
        assert client.get("/api/health").json()["model_loaded"] is True
        assert finished(client, upload(client))["status"] == "failed"
        assert client.get("/api/health").json()["model_loaded"] is False
        assert finished(client, upload(client))["status"] == "done"
        assert client.get("/api/health").json()["model_loaded"] is True


@pytest.mark.parametrize("crash", [False, True], ids=["timeout", "crash"])
def test_timeout_fails_job_and_queue_continues(tmp_path, caplog, crash):
    caplog.set_level(logging.INFO)
    marker = tmp_path / "started"
    analyzer = partial(hang_once, marker=marker, crash=crash)
    with TestClient(create_app(api_key="jury-key", analyzer=analyzer, job_timeout_sec=1.5)) as client:
        first = upload(client)
        second = upload(client)
        assert finished(client, first) == {
            "job_id": first,
            "status": "failed",
            "verdict": None,
            "violations": [],
        }
        assert finished(client, second)["status"] == "done"
        assert not Path(marker.read_text(encoding="utf-8")).exists()
        # A late result must not overwrite the terminal failed state.
        assert client.get(f"/api/v1/jobs/{first}", headers=KEY).json()["status"] == "failed"
        assert_job_duration(caplog, first, "failed")
        assert_job_duration(caplog, second, "done")


def assert_job_duration(caplog, job_id, status):
    messages = [record.getMessage() for record in caplog.records]
    prefix = f"Job {job_id} status={status} duration_sec="
    durations = [float(message.removeprefix(prefix)) for message in messages if message.startswith(prefix)]
    assert len(durations) == 1
    assert durations[0] >= 0


@lru_cache
def video_bytes():
    """Generate a blank video locally; never use dataset footage in API tests."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "blank.mov"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1, (32, 32))
        assert writer.isOpened()
        try:
            writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
        finally:
            writer.release()
        return path.read_bytes()


@pytest.mark.parametrize("prefix", ["/api/v1", "/api/ui"])
@pytest.mark.parametrize("data", [b"", b"not a video", b"\x00\x00\x00\x14ftypqt  \x00\x00\x00\x00qt  ", None])
def test_non_video_upload_is_rejected(prefix, data):
    if data is None:
        image = BytesIO()
        Image.new("RGB", (32, 32)).save(image, format="PNG")
        data = image.getvalue()
    with TestClient(create_app(api_key="jury-key", analyzer=analyze_stub)) as client:
        response = client.post(
            f"{prefix}/analyze",
            headers=KEY,
            files={"file": ("fake.mov", data, "video/quicktime")},
        )
        assert response.status_code == 422
        assert "видео" in response.json()["detail"].lower()
        assert finished(client, upload(client, prefix), prefix)["status"] == "done"


@pytest.mark.parametrize("prefix", ["/api/v1", "/api/ui"])
def test_upload_size_limit_and_recovery(prefix):
    with TestClient(
        create_app(api_key="jury-key", analyzer=analyze_stub, max_upload_bytes=len(video_bytes()))
    ) as client:
        response = client.post(
            f"{prefix}/analyze",
            headers=KEY,
            files={"file": ("large.mov", video_bytes() + b"x", "video/quicktime")},
        )
        assert response.status_code == 413
        assert "размер" in response.json()["detail"].lower()
        assert finished(client, upload(client, prefix), prefix)["status"] == "done"


@pytest.mark.parametrize("length_header", [None, "1", "1000000"])
def test_request_limit_before_multipart_parsing(length_header):
    headers = {**KEY, "Content-Type": "multipart/form-data; boundary=upload"}
    if length_header is not None:
        headers["Content-Length"] = length_header
    with TestClient(create_app(api_key="jury-key", analyzer=analyze_stub, max_upload_bytes=1024)) as client:
        response = client.post(
            "/api/v1/analyze",
            headers=headers,
            content=iter([b"--upload\r\n", b"x" * 70000]),
        )
        assert response.status_code == 413
        assert "размер" in response.json()["detail"].lower()


@pytest.fixture(autouse=True)
def real_analyzer_by_default(monkeypatch):
    monkeypatch.setenv("FORTE_ANALYZER", "real")


def test_explicit_stub_mode_starts_without_weights_and_completes_jobs(monkeypatch):
    monkeypatch.setenv("FORTE_ANALYZER", "stub")
    monkeypatch.setitem(sys.modules, "src.models.analyze", None)
    with TestClient(create_app(api_key="jury-key")) as client:
        assert client.get("/api/health").json()["model_loaded"] is False
        result = finished(client, upload(client))
        assert result["status"] == "done"
        assert result["verdict"] == "violation"
        assert result["violations"][0]["timecode_sec"] == 0


def load_test_model(warmed=None, paths=None, failing_loader=None):
    """Substitute the ML boundary inside the child, without importing model libraries."""
    warmed_names = []
    calls = 0

    def warm(name):
        if name == failing_loader:
            raise RuntimeError("Local model is unavailable")
        warmed_names.append(name)
        if warmed is not None:
            warmed.put((name, os.getpid()))

    def analyze(path):
        nonlocal calls
        assert set(warmed_names) == {"default_model", "default_embedder"}
        paths.put(path)
        calls += 1
        if calls == 1:
            return []
        violation = analyze_stub(path)[0]
        violation.timecode_sec = 12.5
        violation.evidence_frames[0].timestamp_sec = 12.5
        return [violation]

    sys.modules["src.models.analyze"] = SimpleNamespace(
        default_model=lambda: warm("default_model"),
        default_embedder=lambda: warm("default_embedder"),
        analyze=analyze,
    )
    return load_analyzer()


@pytest.mark.parametrize("mode", [None, "real"])
def test_default_app_warms_model_once_and_serves_its_results(monkeypatch, mode):
    if mode is None:
        monkeypatch.delenv("FORTE_ANALYZER")
    warmed = CONTEXT.Queue()
    paths = CONTEXT.Queue()
    app = create_app(api_key="jury-key", analyzer_loader=partial(load_test_model, warmed, paths))
    with pytest.raises(Empty):
        warmed.get(timeout=0.02)
    with TestClient(app) as client:
        assert client.get("/api/health").json()["model_loaded"] is True
        clean = finished(client, upload(client))
        assert clean["status"] == "done"
        assert clean["verdict"] == "no_violation"
        assert clean["violations"] == []
        violation = finished(client, upload(client))
        assert violation["status"] == "done"
        assert violation["verdict"] == "violation"
        assert violation["violations"][0]["timecode_sec"] == 12.5
        assert violation["violations"][0]["evidence_frames"][0]["timestamp_sec"] == 12.5
        loaded = [warmed.get(timeout=1), warmed.get(timeout=1)]
        assert {name for name, _ in loaded} == {"default_model", "default_embedder"}
        assert all(pid != os.getpid() for _, pid in loaded)
        with pytest.raises(Empty):
            warmed.get(timeout=0.02)
        assert all(not paths.get(timeout=1).exists() for _ in range(2))


@pytest.mark.parametrize("failing_loader", ["default_model", "default_embedder"])
def test_model_load_failure_stops_startup_without_stub_fallback(failing_loader):
    app = create_app(analyzer_loader=partial(load_test_model, failing_loader=failing_loader))
    with pytest.raises(RuntimeError, match="Local model is unavailable"), TestClient(app):
        pytest.fail("Application must not accept requests without the model")


def clean_analyzer(path):
    return []


@pytest.mark.parametrize("mode", ["real", "stub"])
def test_injected_analyzer_does_not_load_model(monkeypatch, mode):
    monkeypatch.setenv("FORTE_ANALYZER", mode)
    monkeypatch.setitem(sys.modules, "src.models.analyze", None)
    with TestClient(create_app(analyzer=clean_analyzer, api_key="jury-key")) as client:
        assert client.get("/api/health").json()["model_loaded"] is False
        result = finished(client, upload(client))
        assert result["status"] == "done"
        assert result["verdict"] == "no_violation"


def test_unknown_analyzer_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("FORTE_ANALYZER", "stbu")
    with pytest.raises(ValueError, match="FORTE_ANALYZER must be 'real' or 'stub'"):
        create_app()


def test_ui_rule_catalog_has_russian_names_and_exact_sources_without_key():
    with TestClient(create_app(api_key="jury-key", analyzer=analyze_stub)) as client:
        response = client.get("/api/ui/rules")
        assert response.status_code == 200
        catalog = {item["id"]: item for item in response.json()}
        assert (
            catalog["count_without_controller"]["name"] == "Пересчёт наличных без контролирующего работника"
        )
        rule = catalog["count_without_controller"]["rules"][0]
        assert rule["punkt"] == "п. 42"
        assert rule["url"] == "https://adilet.zan.kz/rus/docs/V1900019680"
        assert "№ 231" in rule["approved_by"]
        assert rule["title"].startswith("Правила осуществления кассовых операций")
        assert catalog["other"]["rules"][0]["punkt"] is None
        for item in catalog.values():
            citations = [rule["citation"] for rule in item["rules"]]
            assert len(citations) == len(set(citations))
            description = describe_violation(item["id"], {"object": "предмет"}).description
            assert description.endswith(" " + item["citation_suffix"])
        assert rule_catalog() is rule_catalog()
        assert client.get("/api/v1/rules", headers=KEY).status_code == 404


def upload(client, prefix="/api/v1"):
    response = client.post(
        f"{prefix}/analyze",
        headers=KEY,
        files={"file": ("sample.mov", video_bytes(), "video/quicktime")},
    )
    assert response.status_code == 200
    assert set(response.json()) == {"job_id"}
    return response.json()["job_id"]


def finished(client, job_id, prefix="/api/v1"):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        response = client.get(f"{prefix}/jobs/{job_id}", headers=KEY)
        assert response.status_code == 200
        if response.json()["status"] != "processing":
            return response.json()
        time.sleep(0.005)
    pytest.fail("Job did not finish within two seconds")


def test_processing_to_done_and_single_worker_cleanup():
    release = CONTEXT.Event()
    started = CONTEXT.Event()
    observed = CONTEXT.Queue()
    analyzer = ScenarioAnalyzer(observed, release=release, started=started)

    with TestClient(create_app(api_key="jury-key", analyzer=analyzer)) as client:
        try:
            first = upload(client)
            assert started.wait(2)
            paths = [observed.get(timeout=1)]
            assert paths[0].read_bytes() == video_bytes()
            second = upload(client)
            assert second != first
            for job_id in (first, second):
                assert client.get(f"/api/v1/jobs/{job_id}", headers=KEY).json() == {
                    "job_id": job_id,
                    "status": "processing",
                    "verdict": None,
                    "violations": [],
                }
            with pytest.raises(Empty):
                observed.get(timeout=0.02)
        finally:
            release.set()
        for job_id in (first, second):
            body = finished(client, job_id)
            assert set(body) == {"job_id", "status", "verdict", "violations"}
            assert body["verdict"] == "violation"
            violation = body["violations"][0]
            assert set(violation) == {"type", "timecode_sec", "description", "evidence_frames"}
            assert violation["type"] == "Иное подозрительное действие"
            frame = violation["evidence_frames"][0]
            assert set(frame) == {"timestamp_sec", "image_base64"}
            assert frame["timestamp_sec"] == violation["timecode_sec"] == 0
            with Image.open(BytesIO(base64.b64decode(frame["image_base64"], validate=True))) as image:
                assert image.format == "JPEG"
                assert image.getextrema() == ((255, 255), (255, 255), (255, 255))
        paths.append(observed.get(timeout=1))
        assert all(not path.exists() for path in paths)


@pytest.mark.parametrize("outcome", ["clean", "exception", "invalid"])
def test_worker_result_failure_and_recovery(outcome, caplog):
    caplog.set_level(logging.INFO)
    observed = CONTEXT.Queue()
    analyzer = ScenarioAnalyzer(observed, outcome=outcome)

    with TestClient(create_app(api_key="jury-key", analyzer=analyzer)) as client:
        job_id = upload(client, "/api/ui")
        body = finished(client, job_id)
        assert body == {
            "job_id": job_id,
            "status": "done" if outcome == "clean" else "failed",
            "verdict": "no_violation" if outcome == "clean" else None,
            "violations": [],
        }
        assert client.get(f"/api/ui/jobs/{job_id}").json() == body
        assert_job_duration(caplog, job_id, body["status"])
        assert not observed.get(timeout=1).exists()
        recovered = finished(client, upload(client))
        assert recovered["status"] == "done"
        assert recovered["verdict"] == "no_violation"
        assert not observed.get(timeout=1).exists()


def test_health_validation_and_removed_routes():
    with TestClient(create_app(api_key="jury-key", analyzer=analyze_stub)) as client:
        health = client.get("/api/health").json()
        assert set(health) == {"status", "version", "model_loaded"}
        assert health["status"] == "ok" and health["model_loaded"] is False
        assert client.get("/api/v1/jobs/missing", headers=KEY).status_code == 404
        assert client.post("/api/v1/analyze", headers=KEY).status_code == 422
        assert client.post("/api/ui/analyze", files={"file": ("x.mp4", b"x")}).status_code == 422
        for path in ("/api/predict", "/api/meta", "/api/insights", "/api/clients/123"):
            assert client.get(path).status_code == 404
        assert finished(client, upload(client, "/api/ui"), "/api/ui")["status"] == "done"


def test_missing_server_key_fails_closed(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with TestClient(create_app(api_key="", analyzer=analyze_stub)) as client:
        assert client.get("/api/v1/jobs/missing", headers=KEY).status_code == 401


@pytest.mark.parametrize(
    "body",
    [
        {"status": "processing", "verdict": "no_violation"},
        {"status": "failed", "verdict": "violation"},
        {"status": "done", "verdict": None},
        {"status": "done", "verdict": "violation"},
        {"status": "done", "verdict": "no_violation", "extra": 1},
    ],
)
def test_job_schema_rejects_inconsistent_results(body):
    with pytest.raises(ValidationError):
        JobResponse(job_id="example", **body)


def test_v1_requires_key_before_parsing_upload():
    with TestClient(create_app(api_key="jury-key", analyzer=analyze_stub)) as client:
        for headers in ({}, {"X-API-Key": "wrong"}):
            assert client.post("/api/v1/analyze", headers=headers).status_code == 401
            assert client.get("/api/v1/jobs/missing", headers=headers).status_code == 401
            assert client.get("/api/v1/unknown", headers=headers).status_code == 401


@pytest.mark.parametrize("frame_time", [43.001, 43.8, 44.2, 44.999, 43.0, 45.0])
def test_decoded_frame_time_tolerance_through_job_api(frame_time):
    observed = CONTEXT.Queue()
    analyzer = ScenarioAnalyzer(observed, frame_time=frame_time)

    with TestClient(create_app(api_key="jury-key", analyzer=analyzer)) as client:
        body = finished(client, upload(client))
        if frame_time in (43.0, 45.0):
            assert body["status"] == "failed"
            assert body["verdict"] is None
            assert body["violations"] == []
        else:
            assert body["status"] == "done"
            assert body["verdict"] == "violation"
            assert body["violations"][0]["timecode_sec"] == 44.0
            assert body["violations"][0]["evidence_frames"][0]["timestamp_sec"] == frame_time
        assert not observed.get(timeout=1).exists()
