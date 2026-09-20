"""The API must stay up before the first model exists - that is the cold-start case.

The registry is stubbed out here: these tests cover the serving contract, not
MLflow connectivity, and must never touch the network.
"""

import pytest
from fastapi.testclient import TestClient

import api.main as main


@pytest.fixture
def client(monkeypatch):
    def no_registry():
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(main, "refresh_model", no_registry)
    main.STATE.update({"model": None, "version": None, "run_id": None, "metrics": {}})
    with TestClient(main.app) as c:
        yield c


class DummyModel:
    def predict(self, frame):
        return [1.0] * len(frame)


def test_healthz_is_up_even_without_a_model(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_readyz_is_503_without_a_model(client):
    assert client.get("/readyz").status_code == 503


def test_model_info_is_503_without_a_model(client):
    assert client.get("/model").status_code == 503


def test_predict_is_503_without_a_model(client):
    response = client.post("/predict", json={"rows": [{"mean_radius": 1.0}]})
    assert response.status_code == 503
    assert "training_pipeline" in response.json()["detail"]


def test_predict_rejects_an_empty_batch(client):
    assert client.post("/predict", json={"rows": []}).status_code == 422


def test_reload_reports_503_when_the_registry_is_down(client):
    assert client.post("/reload").status_code == 503


def test_predict_serves_once_a_model_is_loaded(client):
    main.STATE.update({"model": DummyModel(), "version": "3", "run_id": "abc", "metrics": {}})
    response = client.post("/predict", json={"rows": [{"mean_radius": 1.0}]})
    assert response.status_code == 200
    body = response.json()
    assert body["predictions"] == [1.0]
    assert body["model_version"] == "3"


def test_model_info_reports_the_loaded_version(client):
    main.STATE.update({"model": DummyModel(), "version": "3", "run_id": "abc", "metrics": {}})
    assert client.get("/model").json()["version"] == "3"


def test_metrics_endpoint_is_exposed(client):
    response = client.get("/metrics")
    assert response.status_code == 200
