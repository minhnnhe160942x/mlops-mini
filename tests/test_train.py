"""Training logic, exercised without MLflow."""

import numpy as np
import pandas as pd

from mlops import extract, scale, split
from mlops.train import PRIMARY_METRIC, build_estimator, evaluate, to_target


def test_benign_is_the_positive_class(tiny_frame):
    assert to_target(tiny_frame).tolist() == [0, 1, 1, 0]


def test_the_model_separates_the_two_classes(snapshot, staging):
    split.split(snapshot, staging, "2026-08-25", test_fraction=0.20)
    scale.scale(staging, "2026-08-25")

    train = pd.read_parquet(staging / "2026-08-25" / "train.parquet")
    test = pd.read_parquet(staging / "2026-08-25" / "test.parquet")
    numeric = extract.feature_columns(train)

    model = build_estimator()
    model.fit(train[numeric], to_target(train))
    metrics = evaluate(model, test[numeric], to_target(test))

    assert {"test_accuracy", "test_f1", PRIMARY_METRIC} == metrics.keys()
    assert metrics[PRIMARY_METRIC] > 0.9
    assert metrics["test_accuracy"] > 0.9


def test_the_registered_model_accepts_raw_measurements(snapshot, staging):
    """Regression: the model is fitted on scaled data but served raw features.

    Without the scaler travelling with the model, a benign row and a malignant
    row came back with the same probability, because both were far outside the
    range the forest was trained on.
    """
    from mlops.train import serving_pipeline

    ds = "2026-08-25"
    split.split(snapshot, staging, ds, test_fraction=0.20)
    scale.scale(staging, ds)

    train_scaled = pd.read_parquet(staging / ds / "train.parquet")
    numeric = extract.feature_columns(train_scaled)
    model = build_estimator()
    model.fit(train_scaled[numeric], to_target(train_scaled))

    served = serving_pipeline(model, pd.read_parquet(staging / ds / "train_unscaled.parquet"))
    raw_test = pd.read_parquet(staging / ds / "test_unscaled.parquet")

    benign = raw_test[raw_test[extract.LABEL] == "B"][numeric].head(1)
    malignant = raw_test[raw_test[extract.LABEL] == "M"][numeric].head(1)

    p_benign = served.predict_proba(benign)[0][1]
    p_malignant = served.predict_proba(malignant)[0][1]

    assert p_benign > 0.5, "a benign row must not be called malignant"
    assert p_malignant < 0.5, "a malignant row must not be called benign"
    assert p_benign != p_malignant, "different rows must not score identically"


def test_scaling_through_the_served_model_matches_the_scale_task(snapshot, staging):
    """The bundled scaler must reproduce the scale task's numbers exactly."""
    from mlops.train import serving_pipeline

    ds = "2026-08-25"
    split.split(snapshot, staging, ds, test_fraction=0.20)
    scale.scale(staging, ds)

    raw = pd.read_parquet(staging / ds / "train_unscaled.parquet")
    expected = pd.read_parquet(staging / ds / "train.parquet")
    numeric = extract.feature_columns(raw)

    served = serving_pipeline(build_estimator(), raw)
    produced = served.named_steps["scaler"].transform(raw[numeric])

    assert np.allclose(produced, expected[numeric].to_numpy(), atol=1e-9)
