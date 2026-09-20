"""Shared fixtures. Tests never touch the network or a real MLflow server."""

from __future__ import annotations

import pandas as pd
import pytest

from mlops.extract import ID, LABEL

RAW = "data/raw/wdbc.csv"


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    return pd.read_csv(RAW)


@pytest.fixture
def staging(tmp_path):
    root = tmp_path / "staging"
    root.mkdir()
    return root


@pytest.fixture
def snapshot(raw_frame, tmp_path):
    """The parquet snapshot `ingest` would have written."""
    path = tmp_path / "raw.parquet"
    raw_frame.to_parquet(path, index=False)
    return path


@pytest.fixture
def tiny_frame() -> pd.DataFrame:
    """A four-row frame with the shape the pipeline expects."""
    return pd.DataFrame(
        {
            ID: ["WDBC-0001", "WDBC-0002", "WDBC-0003", "WDBC-0004"],
            "mean_radius": [10.0, 12.0, 14.0, 16.0],
            "mean_area": [100.0, 120.0, 140.0, 160.0],
            LABEL: ["M", "B", "B", "M"],
        }
    )
