"""Model training and evaluation, logged to MLflow."""

from __future__ import annotations

from typing import Any

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import TARGET, feature_columns

PRIMARY_METRIC = "roc_auc"


def build_estimator(random_state: int = 42) -> Pipeline:
    """Scaler + forest. Kept in one Pipeline so serving needs no preprocessing code."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate(estimator: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = estimator.predict(x_test)
    probabilities = estimator.predict_proba(x_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        PRIMARY_METRIC: float(roc_auc_score(y_test, probabilities)),
    }


def train_and_log(
    frame: pd.DataFrame,
    experiment_name: str,
    random_state: int = 42,
    extra_tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Train, log params/metrics/model to MLflow, return the run id and metrics."""
    features = feature_columns(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        frame[features],
        frame[TARGET],
        test_size=0.25,
        random_state=random_state,
        stratify=frame[TARGET],
    )

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        estimator = build_estimator(random_state)
        estimator.fit(x_train, y_train)
        metrics = evaluate(estimator, x_test, y_test)

        mlflow.log_params(
            {
                "n_estimators": 200,
                "max_depth": 8,
                "random_state": random_state,
                "n_train_rows": len(x_train),
                "n_features": len(features),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(extra_tags or {})
        mlflow.sklearn.log_model(
            sk_model=estimator,
            name="model",
            input_example=x_train.head(2),
        )
        return {"run_id": run.info.run_id, "metrics": metrics, "features": features}
