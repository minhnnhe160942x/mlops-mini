"""Z-score normalisation.

The statistics come from the training half only. Fitting them on everything
would leak information about the test rows into the model and quietly inflate
every metric the pipeline reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .extract import feature_columns, run_dir


def fit(train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Mean and standard deviation per feature; a constant column keeps std 1."""
    numeric = feature_columns(train)
    mean = train[numeric].mean()
    std = train[numeric].std().replace(0, 1)
    return mean, std


def apply(frame: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    numeric = feature_columns(frame)
    scaled = frame.copy()
    scaled[numeric] = (frame[numeric] - mean) / std
    return scaled


def scale(staging_root: Path, ds: str) -> dict:
    """Scale both halves with the training statistics and persist them."""
    directory = run_dir(staging_root, ds)
    train = pd.read_parquet(directory / "train_unscaled.parquet")
    test = pd.read_parquet(directory / "test_unscaled.parquet")

    mean, std = fit(train)
    apply(train, mean, std).to_parquet(directory / "train.parquet", index=False)
    apply(test, mean, std).to_parquet(directory / "test.parquet", index=False)

    (directory / "scaler.json").write_text(
        json.dumps(
            {"mean": mean.round(6).to_dict(), "std": std.round(6).to_dict()}, indent=2
        )
    )
    return {"scaled_columns": len(feature_columns(train)), "fitted_on": len(train)}
