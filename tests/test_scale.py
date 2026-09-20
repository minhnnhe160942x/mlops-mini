"""Scaling must take its statistics from the training half only."""

import json

import pandas as pd
import pytest

from mlops import extract, scale, split


@pytest.fixture
def prepared(snapshot, staging):
    split.split(snapshot, staging, "2026-08-25", test_fraction=0.20)
    return staging, "2026-08-25"


def test_scaled_training_data_is_centred(prepared):
    staging, ds = prepared
    scale.scale(staging, ds)
    train = pd.read_parquet(staging / ds / "train.parquet")
    numeric = extract.feature_columns(train)
    assert train[numeric].mean().abs().max() < 1e-9
    assert (train[numeric].std() - 1).abs().max() < 1e-9


def test_test_data_is_not_centred_on_itself(prepared):
    """If the test half were also centred on itself, that would be leakage."""
    staging, ds = prepared
    scale.scale(staging, ds)
    test = pd.read_parquet(staging / ds / "test.parquet")
    numeric = extract.feature_columns(test)
    assert test[numeric].mean().abs().max() > 1e-6


def test_the_scaler_statistics_are_persisted(prepared):
    staging, ds = prepared
    scale.scale(staging, ds)
    scaler = json.loads((staging / ds / "scaler.json").read_text())
    assert set(scaler) == {"mean", "std"}
    assert len(scaler["mean"]) == 30


def test_statistics_come_from_the_training_rows_only(prepared):
    staging, ds = prepared
    info = scale.scale(staging, ds)
    train = pd.read_parquet(staging / ds / "train_unscaled.parquet")
    assert info["fitted_on"] == len(train)


def test_a_constant_column_does_not_divide_by_zero():
    frame = pd.DataFrame(
        {extract.ID: ["a", "b"], "flat": [5.0, 5.0], extract.LABEL: ["M", "B"]}
    )
    mean, std = scale.fit(frame)
    assert std["flat"] == 1
    assert scale.apply(frame, mean, std)["flat"].tolist() == [0.0, 0.0]
