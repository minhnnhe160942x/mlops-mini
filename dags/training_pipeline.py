"""Training pipeline: ingest -> validate -> drift gate -> train -> evaluate -> promote.

The drift gate is what makes this more than a cron job: a batch that looks like
the reference data skips retraining entirely, and a trained model is only
promoted when it beats the incumbent champion.

Trigger with a config to simulate drift:
    {"shift": 1.5, "batch_rows": 200}
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# Airflow 3 exposes the decorators through the Task SDK; fall back to the 2.x
# location so the DAG file stays importable for local linting and tests.
try:
    from airflow.sdk import dag, task
except ImportError:  # pragma: no cover - only hit on Airflow 2.x
    from airflow.decorators import dag, task

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops import data, drift, registry  # noqa: E402
from mlops.config import load_settings  # noqa: E402
from mlops.train import PRIMARY_METRIC, train_and_log  # noqa: E402

SETTINGS = load_settings()


def materialise(spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild this run's (reference, batch) pair. See `mlops.data.materialise`."""
    return data.materialise(spec, SETTINGS.random_state)


@dag(
    dag_id="training_pipeline",
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    tags=["mlops", "mlflow"],
    params={"shift": 0.0, "batch_rows": 200},
    doc_md=__doc__,
)
def training_pipeline():
    @task
    def ingest(**context) -> dict:
        """Describe the incoming batch. `shift` fakes an upstream distribution change."""
        params = context["params"]
        spec = {"shift": float(params["shift"]), "batch_rows": int(params["batch_rows"])}
        _, batch = materialise(spec)
        print(f"ingested {len(batch)} rows with shift={spec['shift']}")
        return spec

    @task
    def validate(spec: dict) -> dict:
        """Fail fast on an empty or malformed batch before spending time training."""
        reference, batch = materialise(spec)

        if batch.empty:
            raise ValueError("incoming batch is empty")
        missing = set(reference.columns) - set(batch.columns)
        if missing:
            raise ValueError(f"batch is missing columns: {sorted(missing)}")
        if batch[data.TARGET].isna().any():
            raise ValueError("batch contains rows without a label")
        return spec

    @task
    def check_drift(spec: dict) -> dict:
        reference, batch = materialise(spec)
        scores = drift.dataset_psi(reference, batch, data.feature_columns(reference))
        verdict = drift.summarise(scores, SETTINGS.drift_psi_threshold)
        print(
            f"PSI max={verdict['max_psi']:.4f} threshold={SETTINGS.drift_psi_threshold} "
            f"drifted={verdict['drifted']}"
        )
        return {**spec, "drift": verdict}

    @task.branch
    def drift_gate(spec: dict) -> str:
        """Retrain only when the batch actually looks different."""
        return "train" if spec["drift"]["drifted"] else "skip_retrain"

    @task
    def skip_retrain() -> str:
        return "no drift detected - champion kept"

    @task
    def train(spec: dict) -> dict:
        reference, batch = materialise(spec)
        combined = pd.concat([reference, batch], ignore_index=True)

        mlflow_run = train_and_log(
            combined,
            experiment_name=SETTINGS.experiment_name,
            random_state=SETTINGS.random_state,
            extra_tags={
                "trigger": "drift",
                "max_psi": f"{spec['drift']['max_psi']:.4f}",
            },
        )
        print(f"run {mlflow_run['run_id']} metrics {mlflow_run['metrics']}")
        return mlflow_run

    @task.branch
    def evaluate(mlflow_run: dict) -> str:
        """Promote only if the candidate beats the incumbent on the primary metric."""
        client = registry.get_client(SETTINGS.tracking_uri)
        incumbent = registry.champion_metric(
            client, SETTINGS.model_name, SETTINGS.model_alias, PRIMARY_METRIC
        )
        candidate = mlflow_run["metrics"][PRIMARY_METRIC]

        if incumbent is None:
            print(f"no champion yet - promoting the first model ({candidate:.4f})")
            return "register_and_promote"
        if candidate >= incumbent:
            print(f"candidate {candidate:.4f} >= champion {incumbent:.4f} - promoting")
            return "register_and_promote"
        print(f"candidate {candidate:.4f} < champion {incumbent:.4f} - rejected")
        return "reject"

    @task
    def register_and_promote(mlflow_run: dict) -> dict:
        client = registry.get_client(SETTINGS.tracking_uri)
        version = registry.register_version(
            client,
            run_id=mlflow_run["run_id"],
            model_name=SETTINGS.model_name,
            tags={PRIMARY_METRIC: f"{mlflow_run['metrics'][PRIMARY_METRIC]:.4f}"},
        )
        registry.promote(client, SETTINGS.model_name, version, SETTINGS.model_alias)
        print(f"promoted version {version} to @{SETTINGS.model_alias}")
        return {"version": version, "alias": SETTINGS.model_alias}

    @task
    def reject() -> str:
        return "candidate did not beat the champion - not promoted"

    checked = check_drift(validate(ingest()))
    trained = train(checked)

    drift_gate(checked) >> [trained, skip_retrain()]
    evaluate(trained) >> [register_and_promote(trained), reject()]


training_pipeline()
