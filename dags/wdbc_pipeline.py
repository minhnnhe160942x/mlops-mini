"""DDM501 — a data and ML pipeline that runs end to end.

Six tasks: ingest -> validate -> split -> scale -> train_and_register -> report.

Each task writes its output into `data/staging/<logical date>/`, so a run can be
inspected after the fact and re-running a date only overwrites that date.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Airflow 3 exposes both the decorators and the exceptions through the Task SDK;
# fall back to the 2.x locations so the DAG file stays importable for linting.
try:
    from airflow.sdk import dag, task
    from airflow.sdk.exceptions import AirflowFailException
except ImportError:  # pragma: no cover - only hit on Airflow 2.x
    from airflow.decorators import dag, task
    from airflow.exceptions import AirflowFailException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlops import drift, extract, registry, report, scale, split, validate  # noqa: E402
from mlops.config import load_settings  # noqa: E402
from mlops.train import PRIMARY_METRIC, train_and_register  # noqa: E402

SETTINGS = load_settings()


@dag(
    dag_id="wdbc_pipeline",
    description=(
        "Breast cancer pipeline: ingest, validate, split, scale, "
        "train_and_register, report"
    ),
    schedule="@daily",
    start_date=datetime(2026, 8, 20),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 3,
        "retry_delay": timedelta(seconds=10),
        "retry_exponential_backoff": True,
    },
    tags=["ddm501", "mlflow", "mlops"],
    doc_md=__doc__,
)
def wdbc_pipeline():
    # Function names avoid clashing with the modules they call, so the six
    # task ids the pipeline is specified with are pinned explicitly.
    @task(task_id="ingest")
    def ingest(ds: str = None) -> dict:
        """Freeze this run's copy of the extract before anything touches it."""
        meta = extract.ingest(SETTINGS.raw_extract, SETTINGS.staging_root, ds)
        print(f"ingested {meta['rows']} rows, {meta['cols']} columns")
        return meta

    @task(task_id="validate")
    def validate_extract(meta: dict, ds: str = None) -> dict:
        """Quarantine bad rows and stop the run if too many of them are bad."""
        verdict = validate.validate(
            Path(meta["path"]), SETTINGS.staging_root, ds, SETTINGS.max_bad_fraction
        )
        print(
            f"validation: {verdict['rejected_rows']} rejected of "
            f"{verdict['rejected_rows'] + verdict['clean_rows']} "
            f"({verdict['bad_fraction']:.2%} of rows)"
        )

        if not verdict["passed"]:
            # AirflowFailException skips the retries: a broken extract will not
            # repair itself, so three more attempts only waste time.
            raise AirflowFailException(validate.failure_message(verdict))

        clean = pd.read_parquet(verdict["path"])
        raw = pd.read_parquet(meta["path"])
        scores = drift.dataset_psi(raw, clean, extract.feature_columns(raw))
        summary = drift.summarise(scores, SETTINGS.drift_psi_threshold)
        print(f"PSI of the clean set against the extract: max={summary['max_psi']:.4f}")
        return {**verdict, "max_psi": summary["max_psi"]}

    @task(task_id="split")
    def split_rows(verdict: dict, ds: str = None) -> dict:
        """Deterministic split by hashing the id - no random seed involved."""
        sizes = split.split(
            Path(verdict["path"]), SETTINGS.staging_root, ds, SETTINGS.test_fraction
        )
        print(f"split: {sizes['train']} train / {sizes['test']} test")
        return sizes

    @task(task_id="scale")
    def scale_features(ds: str = None) -> dict:
        """Fit the scaler on train only, then apply it to both halves."""
        info = scale.scale(SETTINGS.staging_root, ds)
        print(f"scaled {info['scaled_columns']} columns using {info['fitted_on']} rows")
        return info

    @task(task_id="train_and_register")
    def train(verdict: dict, ds: str = None) -> dict:
        """Train, log to MLflow and register a new model version."""
        result = train_and_register(
            SETTINGS.staging_root,
            ds,
            tracking_uri=SETTINGS.tracking_uri,
            experiment_name=SETTINGS.experiment_name,
            model_name=SETTINGS.model_name,
            random_state=SETTINGS.random_state,
            extra_tags={"logical_date": ds, "max_psi": f"{verdict['max_psi']:.4f}"},
        )

        client = registry.get_client(SETTINGS.tracking_uri)
        version = registry.latest_version(client, SETTINGS.model_name)
        candidate = result["metrics"][PRIMARY_METRIC]
        incumbent = registry.champion_metric(
            client, SETTINGS.model_name, SETTINGS.model_alias, PRIMARY_METRIC
        )

        # Every run registers a version so nothing is lost, but the alias the API
        # follows only moves when the new model is at least as good.
        promoted = incumbent is None or candidate >= incumbent
        if promoted:
            registry.promote(client, SETTINGS.model_name, version, SETTINGS.model_alias)
            print(f"version {version} promoted to @{SETTINGS.model_alias}")
        else:
            print(f"version {version} kept: {candidate:.4f} < champion {incumbent:.4f}")

        return {
            "run_id": result["run_id"],
            "model_name": SETTINGS.model_name,
            "model_version": int(version),
            "promoted": promoted,
            **{k: round(v, 4) for k, v in result["metrics"].items()},
        }

    @task(task_id="report")
    def write_report(
        verdict: dict, sizes: dict, scaling: dict, training: dict, ds: str = None
    ) -> str:
        """One summary per run, appended to a history the whole pipeline shares."""
        summary = report.write_report(
            SETTINGS.staging_root, ds, {**verdict, **sizes, **scaling, **training}
        )
        print(f"summary: {summary}")
        return str(summary)

    ingested = ingest()
    verdict = validate_extract(ingested)
    sizes = split_rows(verdict)
    scaling = scale_features()
    trained = train(verdict)

    sizes >> scaling >> trained
    write_report(verdict, sizes, scaling, trained)


wdbc_pipeline()
