"""Model training, evaluation and registration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .extract import LABEL, feature_columns, run_dir
from .scale import fit as scale_fit

PRIMARY_METRIC = "test_roc_auc"
BENIGN = "B"


def build_estimator(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)


def to_target(frame: pd.DataFrame) -> pd.Series:
    """1 means benign. The positive class is stated once, here, so the metrics
    and the label the API reports cannot drift apart."""
    return (frame[LABEL] == BENIGN).astype(int)


def evaluate(model: RandomForestClassifier, features: pd.DataFrame, target: pd.Series) -> dict:
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "test_accuracy": float(accuracy_score(target, predictions)),
        "test_f1": float(f1_score(target, predictions)),
        PRIMARY_METRIC: float(roc_auc_score(target, probabilities)),
    }


def serving_pipeline(model: RandomForestClassifier, unscaled_train: pd.DataFrame) -> Pipeline:
    """Bundle the training-time scaling into the model that gets registered.

    The model is fitted on scaled data, so serving it raw measurements would feed
    it numbers from a completely different range - the reason a benign row and a
    malignant row can otherwise come back with the same probability. Attaching
    the scaling lets the API post raw features without knowing a scaler exists.

    The statistics are recomputed with the scale task's own function rather than
    read back from scaler.json, which is rounded for readability; a served row
    has to be transformed with the exact numbers training used.
    """
    mean, std = scale_fit(unscaled_train)

    # Assembled from known statistics instead of re-fitted: pandas uses the
    # sample standard deviation and StandardScaler the population one, and that
    # small difference would shift every served row.
    scaler = StandardScaler()
    scaler.mean_ = mean.to_numpy(dtype=float)
    scaler.scale_ = std.to_numpy(dtype=float)
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = len(mean)
    scaler.feature_names_in_ = np.array(mean.index, dtype=object)
    scaler.n_samples_seen_ = np.int64(0)

    return Pipeline([("scaler", scaler), ("model", model)])


def train_and_register(
    staging_root: Path,
    ds: str,
    tracking_uri: str,
    experiment_name: str,
    model_name: str,
    random_state: int = 42,
    extra_tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Train on the scaled training half, log the run, register a new version."""
    directory = run_dir(staging_root, ds)
    train = pd.read_parquet(directory / "train.parquet")
    test = pd.read_parquet(directory / "test.parquet")
    numeric = feature_columns(train)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        model = build_estimator(random_state)
        model.fit(train[numeric], to_target(train))
        metrics = evaluate(model, test[numeric], to_target(test))

        mlflow.log_params(
            {
                "n_estimators": 200,
                "random_state": random_state,
                "logical_date": ds,
                "n_train_rows": len(train),
                "n_features": len(numeric),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(extra_tags or {})
        unscaled_train = pd.read_parquet(directory / "train_unscaled.parquet")
        served = serving_pipeline(model, unscaled_train)
        raw_example = unscaled_train[numeric].head(2)
        mlflow.sklearn.log_model(
            sk_model=served,
            name="model",
            input_example=raw_example,
            registered_model_name=model_name,
        )

    return {"run_id": run.info.run_id, "model_name": model_name, "metrics": metrics}
