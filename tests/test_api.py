"""API contract + smoke test. Runs the stub pipeline once (~5 s)."""

import pytest
from fastapi.testclient import TestClient

from contracts.schemas import PredictResponse


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from src.pipeline import main

    # The pipeline appends a run to experiments.csv; keep test runs out of the real log.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.models.train.EXPERIMENTS", tmp_path_factory.mktemp("log") / "experiments.csv")
        main(stub=True)
    from api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_contract(client):
    r = client.post("/api/predict", json={"input": {"client_id": "KZ-100001"}})
    assert r.status_code == 200
    body = PredictResponse.model_validate(r.json())
    assert 0 <= body.result <= 1
    assert 1 <= len(body.reasons) <= 5
    assert body.label and body.action


def test_predict_whatif_override(client):
    lo = client.post("/api/predict", json={"input": {"days_overdue_max": 0, "debt_ratio": 0.1}}).json()
    hi = client.post("/api/predict", json={"input": {"days_overdue_max": 110, "debt_ratio": 2.5}}).json()
    assert hi["result"] > lo["result"]  # monotonic sanity on the stub, not a score assertion


def test_unknown_client_404(client):
    assert client.post("/api/predict", json={"input": {"client_id": "nope"}}).status_code == 404


def test_insights_shape(client):
    r = client.get("/api/insights")
    assert r.status_code == 200 and len(r.json()) >= 1
