"""Serving contract. The registry is stubbed out - these never touch the network."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as main

N_FEATURES = 30
ROW = [1.0] * N_FEATURES


class DummyModel:
    """Stands in for the registered sklearn model."""

    feature_names_in_ = np.array([f"f{i}" for i in range(N_FEATURES)])

    def __init__(self, probability_benign: float = 0.9):
        self.probability_benign = probability_benign

    def predict_proba(self, frame):
        return np.array([[1 - self.probability_benign, self.probability_benign]] * len(frame))


def _reset():
    main.STATE.update(
        {"model": None, "version": None, "run_id": None, "metrics": {}, "error": None}
    )
    main._last_attempt = 0.0


@pytest.fixture
def client(monkeypatch):
    def no_registry():
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(main, "refresh_model", no_registry)
    _reset()
    with TestClient(main.app) as c:
        yield c


def _load(model):
    main.STATE.update({"model": model, "version": "1", "run_id": "r", "metrics": {}})


def test_health_is_200_and_degraded_before_any_model_exists(client):
    body = client.get("/health").json()
    assert client.get("/health").status_code == 200
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert body["model_uri"].startswith("models:/")
    assert body["error"]


def test_health_reports_ok_once_a_model_is_loaded(client):
    _load(DummyModel())
    body = client.get("/health").json()
    assert body == {
        "status": "ok",
        "model_loaded": True,
        "model_uri": main.SETTINGS.model_uri,
    }


def test_predict_answers_benign_above_the_threshold(client):
    _load(DummyModel(probability_benign=0.91))
    body = client.post("/predict", json={"features": ROW}).json()
    assert body["prediction"] == "benign"
    assert body["probability_benign"] == 0.91
    assert body["served_by"] == main.SETTINGS.model_uri


def test_predict_answers_malignant_below_the_threshold(client):
    _load(DummyModel(probability_benign=0.02))
    assert client.post("/predict", json={"features": ROW}).json()["prediction"] == "malignant"


def test_predict_is_503_without_a_model(client):
    response = client.post("/predict", json={"features": ROW})
    assert response.status_code == 503
    assert "wdbc_pipeline" in response.json()["detail"]


def test_predict_rejects_the_wrong_number_of_features(client):
    _load(DummyModel())
    assert client.post("/predict", json={"features": [1.0, 2.0]}).status_code == 422


def test_predict_rejects_a_missing_body(client):
    assert client.post("/predict", json={}).status_code == 422


def test_readyz_is_503_without_a_model(client):
    assert client.get("/readyz").status_code == 503


def test_healthz_is_up_even_without_a_model(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_reload_reports_503_when_the_registry_is_down(client):
    assert client.post("/reload").status_code == 503


def test_metrics_endpoint_is_exposed(client):
    assert client.get("/metrics").status_code == 200


def test_the_api_picks_up_a_model_without_being_told(monkeypatch):
    """Cold start: the registry fills in later and the API must notice by itself."""
    _reset()

    def empty_registry():
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(main, "refresh_model", empty_registry)
    with TestClient(main.app) as c:
        assert c.get("/health").json()["model_loaded"] is False

        def registry_now_has_a_model():
            _load(DummyModel())
            return main.STATE

        monkeypatch.setattr(main, "refresh_model", registry_now_has_a_model)
        main._last_attempt = 0.0
        assert c.get("/health").json()["model_loaded"] is True


def test_empty_registry_is_not_retried_on_every_request(monkeypatch):
    """The retry is throttled so an empty registry costs one timeout, not five."""
    _reset()
    calls = []

    def counting_failure():
        calls.append(1)
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(main, "refresh_model", counting_failure)
    with TestClient(main.app) as c:
        calls.clear()
        main._last_attempt = 0.0
        for _ in range(5):
            c.get("/health")
    assert len(calls) == 1, f"5 requests triggered {len(calls)} registry calls"
