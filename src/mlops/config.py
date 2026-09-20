"""Single source of truth for settings shared by Airflow tasks and the API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Inside the containers the project lives at /opt/airflow (Airflow) or /app (API);
# locally it is the repo root. DATA_ROOT lets each of them say where data lives.
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Settings:
    tracking_uri: str
    experiment_name: str
    model_name: str
    model_alias: str
    model_version: str
    drift_psi_threshold: float
    random_state: int
    max_bad_fraction: float
    test_fraction: float
    data_root: Path

    @property
    def raw_extract(self) -> Path:
        return self.data_root / "raw" / "wdbc.csv"

    @property
    def staging_root(self) -> Path:
        return self.data_root / "staging"

    @property
    def model_uri(self) -> str:
        """What the API resolves at load time.

        MODEL_VERSION pins an exact version so a release can be rolled forward or
        back by changing one variable; leaving it unset follows the `champion`
        alias the pipeline moves on every promotion.
        """
        if self.model_version:
            return f"models:/{self.model_name}/{self.model_version}"
        return f"models:/{self.model_name}@{self.model_alias}"


def load_settings() -> Settings:
    return Settings(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        experiment_name=os.getenv("MLFLOW_EXPERIMENT", "wdbc-pipeline-airflow"),
        model_name=os.getenv("MODEL_NAME", "breast-cancer-classifier"),
        model_alias=os.getenv("MODEL_ALIAS", "champion"),
        model_version=os.getenv("MODEL_VERSION", "").strip(),
        drift_psi_threshold=float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2")),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
        max_bad_fraction=float(os.getenv("MAX_BAD_FRACTION", "0.05")),
        test_fraction=float(os.getenv("TEST_FRACTION", "0.20")),
        data_root=Path(os.getenv("DATA_ROOT", str(DEFAULT_DATA_ROOT))),
    )
