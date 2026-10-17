"""Deterministic HTTP contract tests; no model or dataset required."""

import base64
import sys
import time
from io import BytesIO
from threading import Event
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from api.analyzer import analyze_stub
from api.main import create_app
from contracts.schemas import JobResponse
from src.rules.describe import describe_violation, rule_catalog

KEY = {"X-API-Key": "jury-key"}


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


@pytest.mark.parametrize("mode", [None, "real"])
def test_default_app_warms_model_once_and_serves_its_results(monkeypatch, mode):
    if mode is None:
        monkeypatch.delenv("FORTE_ANALYZER")
    warmed = []
    paths = []

    def analyze(path):
        assert set(warmed) == {"classifier", "embedder"}
        paths.append(path)
        if len(paths) == 1:
            return []
        violation = analyze_stub(path)[0]
        violation.timecode_sec = 12.5
        violation.evidence_frames[0].timestamp_sec = 12.5
        return [violation]

    # Substitute the ML boundary: contract tests require neither weights nor ML imports.
    monkeypatch.setitem(
        sys.modules,
        "src.models.analyze",
        SimpleNamespace(
            default_model=lambda: warmed.append("classifier"),
            default_embedder=lambda: warmed.append("embedder"),
            analyze=analyze,
        ),
    )
    app = create_app(api_key="jury-key")
    assert warmed == []
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
        assert len(warmed) == 2
        assert len(paths) == 2
        assert all(not path.exists() for path in paths)


@pytest.mark.parametrize("failing_loader", ["default_model", "default_embedder"])
def test_model_load_failure_stops_startup_without_stub_fallback(monkeypatch, failing_loader):
    def fail():
        raise RuntimeError("Local model is unavailable")

    def unexpected_analyze(path):
        pytest.fail("Analyzer must not run after failed startup")

    model_api = SimpleNamespace(
        default_model=lambda: None,
        default_embedder=lambda: None,
        analyze=unexpected_analyze,
    )
    setattr(model_api, failing_loader, fail)
    monkeypatch.setitem(sys.modules, "src.models.analyze", model_api)
    with pytest.raises(RuntimeError, match="Local model is unavailable"), TestClient(create_app()):
        pytest.fail("Application must not accept requests without the model")


@pytest.mark.parametrize("mode", ["real", "stub"])
def test_injected_analyzer_does_not_load_model(monkeypatch, mode):
    monkeypatch.setenv("FORTE_ANALYZER", mode)
    monkeypatch.setitem(sys.modules, "src.models.analyze", None)
    with TestClient(create_app(analyzer=lambda path: [], api_key="jury-key")) as client:
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
        files={"file": ("sample.mov", b"synthetic fixture", "video/quicktime")},
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
    release = Event()
    started = Event()
    paths = []

    def analyzer(path):
        paths.append(path)
        assert path.read_bytes() == b"synthetic fixture"
        started.set()
        assert release.wait(2)
        return analyze_stub(path)

    with TestClient(create_app(api_key="jury-key", analyzer=analyzer)) as client:
        try:
            first = upload(client)
            assert started.wait(1)
            second = upload(client)
            assert second != first
            for job_id in (first, second):
                assert client.get(f"/api/v1/jobs/{job_id}", headers=KEY).json() == {
                    "job_id": job_id,
                    "status": "processing",
                    "verdict": None,
                    "violations": [],
                }
            assert len(paths) == 1
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
        assert len(paths) == 2
        assert all(not path.exists() for path in paths)


@pytest.mark.parametrize("outcome", ["clean", "exception", "invalid"])
def test_worker_result_failure_and_recovery(outcome):
    paths = []

    def analyzer(path):
        paths.append(path)
        if len(paths) == 1:
            if outcome == "exception":
                raise RuntimeError("Do not expose this error")
            if outcome == "invalid":
                return [{"type": "invalid"}]
        return []

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
        assert not paths[0].exists()
        recovered = finished(client, upload(client))
        assert recovered["status"] == "done"
        assert recovered["verdict"] == "no_violation"
        assert all(not path.exists() for path in paths)


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
    paths = []

    def analyzer(path):
        paths.append(path)
        violation = analyze_stub(path)[0].model_dump()
        violation["timecode_sec"] = 44.0
        violation["evidence_frames"][0]["timestamp_sec"] = frame_time
        return [violation]

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
        assert not paths[0].exists()
