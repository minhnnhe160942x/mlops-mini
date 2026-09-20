"""Model Registry operations.

MLflow 3 drops the old Staging/Production stages in favour of aliases, so
promotion here means moving the `champion` alias onto a new version. The API
resolves `models:/<name>@champion` and never needs to know version numbers.
"""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


def get_client(tracking_uri: str) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def champion_metric(
    client: MlflowClient, model_name: str, alias: str, metric: str
) -> float | None:
    """Metric of the current champion, or None when there is no champion yet."""
    try:
        version = client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None

    run = client.get_run(version.run_id)
    value = run.data.metrics.get(metric)
    return float(value) if value is not None else None


def register_version(
    client: MlflowClient, run_id: str, model_name: str, tags: dict[str, str] | None = None
) -> str:
    """Register the run's model as a new version and return that version number."""
    try:
        client.create_registered_model(model_name)
    except MlflowException:
        pass  # already exists

    version = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
        tags=tags or {},
    )
    return version.version


def promote(client: MlflowClient, model_name: str, version: str, alias: str) -> None:
    client.set_registered_model_alias(model_name, alias, version)


def load_champion(tracking_uri: str, model_name: str, alias: str) -> dict[str, Any]:
    """Load the aliased model for serving, with the metadata the API reports."""
    client = get_client(tracking_uri)
    version = client.get_model_version_by_alias(model_name, alias)
    model = mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")
    run = client.get_run(version.run_id)
    return {
        "model": model,
        "version": version.version,
        "run_id": version.run_id,
        "metrics": run.data.metrics,
    }
