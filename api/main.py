"""FastAPI serving layer.

The service holds no model of its own: it resolves `models:/<name>@champion`
from the MLflow Model Registry at startup, and `POST /reload` picks up whatever
the pipeline promoted since. That keeps deployment and promotion independent.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops import registry  # noqa: E402
from mlops.config import load_settings  # noqa: E402

logger = logging.getLogger("uvicorn.error")
SETTINGS = load_settings()

STATE: dict[str, Any] = {"model": None, "version": None, "run_id": None, "metrics": {}}


def refresh_model() -> dict[str, Any]:
    """(Re)load the aliased model. Raises if the registry has nothing to serve."""
    loaded = registry.load_champion(
        SETTINGS.tracking_uri, SETTINGS.model_name, SETTINGS.model_alias
    )
    STATE.update(loaded)
    logger.info("loaded %s version %s", SETTINGS.model_uri, loaded["version"])
    return loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A cold registry is normal before the first DAG run, so startup must not
    # crash-loop the container - /readyz reports the gap instead.
    try:
        refresh_model()
    except Exception as exc:  # noqa: BLE001
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
    rows: list[dict[str, float]] = Field(
        ...,
        min_length=1,
        description="One object per row, keyed by feature name.",
    )


class PredictResponse(BaseModel):
    predictions: list[float]
    model_version: str
    model_name: str


@app.get("/healthz", tags=["ops"])
def healthz() -> dict[str, str]:
    """Liveness: the process is up. Says nothing about the model."""
    return {"status": "ok"}


@app.get("/readyz", tags=["ops"])
def readyz() -> dict[str, Any]:
    """Readiness: a model is loaded and /predict will answer."""
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="no model loaded from the registry")
    return {"status": "ready", "model_version": STATE["version"]}


@app.get("/model", tags=["model"])
def model_info() -> dict[str, Any]:
    if STATE["model"] is None:
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
        raise HTTPException(status_code=503, detail=f"reload failed: {exc}") from exc
    return {"status": "reloaded", "model_version": loaded["version"]}


@app.post("/predict", response_model=PredictResponse, tags=["model"])
def predict(request: PredictRequest) -> PredictResponse:
    if STATE["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="no model loaded - run the training_pipeline DAG, then POST /reload",
        )

    frame = pd.DataFrame(request.rows)
    try:
        predictions = STATE["model"].predict(frame)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"prediction failed: {exc}") from exc

    return PredictResponse(
        predictions=[float(p) for p in predictions],
        model_version=str(STATE["version"]),
        model_name=SETTINGS.model_name,
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
