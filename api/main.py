"""FastAPI serving layer.

The service holds no model of its own: it resolves `SETTINGS.model_uri` from the
MLflow Model Registry at startup, and `POST /reload` picks up whatever the
pipeline promoted since. That keeps deployment and promotion independent.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops import registry  # noqa: E402
from mlops.config import load_settings  # noqa: E402

logger = logging.getLogger("uvicorn.error")
SETTINGS = load_settings()

STATE: dict[str, Any] = {
    "model": None,
    "version": None,
    "run_id": None,
    "metrics": {},
    "error": None,
}

# How long to wait before retrying a registry that had nothing to serve. Without
# a throttle every request to an empty registry would pay the client timeout.
RETRY_INTERVAL_SECONDS = 15.0
_last_attempt = 0.0

# WDBC as the training set encodes it: class 1 is benign (diagnosis "B"), class 0
# is malignant, so column 1 of predict_proba is the benign probability.
BENIGN_CLASS_INDEX = 1
N_FEATURES = 30


def refresh_model() -> dict[str, Any]:
    """(Re)load the registered model. Raises if the registry has nothing to serve."""
    client = registry.get_client(SETTINGS.tracking_uri)
    version = (
        client.get_model_version(SETTINGS.model_name, SETTINGS.model_version)
        if SETTINGS.model_version
        else client.get_model_version_by_alias(SETTINGS.model_name, SETTINGS.model_alias)
    )
    # sklearn flavour rather than pyfunc: /predict reports a class probability and
    # the pyfunc wrapper does not expose predict_proba.
    model = mlflow.sklearn.load_model(SETTINGS.model_uri)
    loaded = {
        "model": model,
        "version": version.version,
        "run_id": version.run_id,
        "metrics": client.get_run(version.run_id).data.metrics,
        "error": None,
    }
    STATE.update(loaded)
    logger.info("loaded %s version %s", SETTINGS.model_uri, loaded["version"])
    return loaded


def ensure_model() -> bool:
    """Make sure a model is loaded, retrying a previously empty registry.

    The API usually starts before the pipeline has ever run, so the startup load
    finds nothing. Retrying here means the service starts serving on its own once
    the first model is promoted - nobody has to call /reload.
    """
    global _last_attempt

    if STATE["model"] is not None:
        return True

    now = time.monotonic()
    if now - _last_attempt < RETRY_INTERVAL_SECONDS:
        return False
    _last_attempt = now

    try:
        refresh_model()
    except Exception as exc:  # noqa: BLE001
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        logger.info("registry still has nothing to serve: %s", exc)
        return False
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A cold registry is normal before the first DAG run, so startup must not
    # crash-loop the container - /health and /readyz report the gap instead.
    try:
        refresh_model()
    except Exception as exc:  # noqa: BLE001
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("no model to serve yet (%s): %s", SETTINGS.model_uri, exc)
    yield


app = FastAPI(
    title="mlops-mini serving API",
    version="0.1.0",
    description="Serves the model the Airflow pipeline promoted in the MLflow registry.",
    lifespan=lifespan,
)
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


class PredictRequest(BaseModel):
    """One WDBC row as a plain vector, in training column order."""

    features: list[float] = Field(
        ...,
        min_length=N_FEATURES,
        max_length=N_FEATURES,
        description="The 30 WDBC measurements, in the column order the model was trained on.",
    )


class PredictResponse(BaseModel):
    prediction: str
    probability_benign: float
    served_by: str


def _require_model() -> Any:
    if not ensure_model():
        raise HTTPException(
            status_code=503,
            detail=(
                f"{SETTINGS.model_uri} not loaded - run the wdbc_pipeline DAG, "
                f"then POST /reload ({STATE.get('error')})"
            ),
        )
    return STATE["model"]


@app.get("/health", tags=["ops"])
def health() -> dict[str, Any]:
    """Health with the model's state attached.

    Answers 200 even when degraded: an empty registry is an expected phase, not a
    dead process, and a 5xx here would take the container out of rotation for it.
    """
    if ensure_model():
        return {"status": "ok", "model_loaded": True, "model_uri": SETTINGS.model_uri}
    return {
        "status": "degraded",
        "model_loaded": False,
        "model_uri": SETTINGS.model_uri,
        "error": STATE.get("error") or "no model loaded from the registry",
    }


@app.get("/healthz", tags=["ops"])
def healthz() -> dict[str, str]:
    """Liveness: the process is up. Says nothing about the model."""
    return {"status": "ok"}


@app.get("/readyz", tags=["ops"])
def readyz() -> dict[str, Any]:
    """Readiness: a model is loaded and /predict will answer."""
    if not ensure_model():
        raise HTTPException(status_code=503, detail="no model loaded from the registry")
    return {"status": "ready", "model_version": STATE["version"]}


@app.get("/model", tags=["model"])
def model_info() -> dict[str, Any]:
    if not ensure_model():
        raise HTTPException(status_code=503, detail="no model loaded from the registry")
    return {
        "name": SETTINGS.model_name,
        "alias": SETTINGS.model_alias,
        "uri": SETTINGS.model_uri,
        "version": STATE["version"],
        "run_id": STATE["run_id"],
        "metrics": STATE["metrics"],
    }


@app.post("/reload", tags=["model"])
def reload_model() -> dict[str, Any]:
    """Pull whatever the pipeline promoted most recently."""
    try:
        loaded = refresh_model()
    except Exception as exc:  # noqa: BLE001
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        raise HTTPException(status_code=503, detail=f"reload failed: {exc}") from exc
    return {"status": "reloaded", "model_version": loaded["version"]}


@app.post("/predict", response_model=PredictResponse, tags=["model"])
def predict(request: PredictRequest) -> PredictResponse:
    """Score one row and answer with the diagnosis and its probability."""
    model = _require_model()

    # Training used named columns, so feed the vector back under those names -
    # positional input would silently misalign the features.
    columns = getattr(model, "feature_names_in_", None)
    frame = pd.DataFrame([request.features], columns=columns)

    try:
        proba = float(model.predict_proba(frame)[0][BENIGN_CLASS_INDEX])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"prediction failed: {exc}") from exc

    return PredictResponse(
        prediction="benign" if proba >= 0.5 else "malignant",
        probability_benign=round(proba, 4),
        served_by=SETTINGS.model_uri,
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
