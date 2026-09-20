"""Single source of truth for settings shared by Airflow tasks and the API."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    tracking_uri: str
    experiment_name: str
    model_name: str
    model_alias: str
    drift_psi_threshold: float
    random_state: int

    @property
    def model_uri(self) -> str:
        """Registry URI the API resolves at load time."""
        return f"models:/{self.model_name}@{self.model_alias}"


def load_settings() -> Settings:
    return Settings(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        experiment_name=os.getenv("MLFLOW_EXPERIMENT", "breast-cancer"),
        model_name=os.getenv("MODEL_NAME", "breast-cancer-classifier"),
        model_alias=os.getenv("MODEL_ALIAS", "champion"),
        drift_psi_threshold=float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2")),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
    )
