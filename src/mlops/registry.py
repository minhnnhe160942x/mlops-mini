"""Model Registry operations.

MLflow 3 replaced the old Staging/Production stages with aliases, so promotion
means moving the `champion` alias onto a version. Pinning an exact version with
MODEL_VERSION still works and is what a rollback uses.
"""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


def get_client(tracking_uri: str) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def latest_version(client: MlflowClient, model_name: str) -> str | None:
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None
    return str(max(int(v.version) for v in versions))


def champion_metric(
    client: MlflowClient, model_name: str, alias: str, metric: str
) -> float | None:
    """Metric of the current champion, or None when there is no champion yet."""
    try:
        version = client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None

    value = client.get_run(version.run_id).data.metrics.get(metric)
    return float(value) if value is not None else None


def promote(client: MlflowClient, model_name: str, version: str, alias: str) -> None:
    client.set_registered_model_alias(model_name, alias, version)


def load_serving_model(tracking_uri: str, model_uri: str) -> dict[str, Any]:
    """Load a model for serving and describe what was loaded.

    Works with both URI shapes: `models:/name/3` pins a version and
    `models:/name@champion` follows the alias.
    """
    client = get_client(tracking_uri)
    model = mlflow.sklearn.load_model(model_uri)

    reference = model_uri.removeprefix("models:/")
    if "@" in reference:
        name, alias = reference.split("@", 1)
        version_info = client.get_model_version_by_alias(name, alias)
    else:
        name, version = reference.rsplit("/", 1)
        version_info = client.get_model_version(name, version)

    metrics = client.get_run(version_info.run_id).data.metrics
    return {
        "model": model,
        "name": name,
        "version": version_info.version,
        "run_id": version_info.run_id,
        "metrics": metrics,
    }
