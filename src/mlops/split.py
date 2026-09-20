"""Train/test split that does not depend on a random seed.

Bucketing a hash of the sample id means a row lands in the same side of the
split on every machine and every re-run, and a row added to the extract later
cannot reshuffle the rows already there.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .extract import ID, run_dir

BUCKETS = 100


def bucket(sample_id: str) -> int:
    """Map an id onto 0..99, evenly and deterministically."""
    digest = hashlib.sha256(str(sample_id).encode()).hexdigest()
    return int(digest[:8], 16) % BUCKETS


def test_mask(frame: pd.DataFrame, test_fraction: float) -> pd.Series:
    return frame[ID].map(bucket) < test_fraction * BUCKETS


def split(source_path: Path, staging_root: Path, ds: str, test_fraction: float) -> dict:
    """Write the two unscaled halves and report their sizes."""
    frame = pd.read_parquet(source_path)
    is_test = test_mask(frame, test_fraction)

    directory = run_dir(staging_root, ds)
    frame[~is_test].to_parquet(directory / "train_unscaled.parquet", index=False)
    frame[is_test].to_parquet(directory / "test_unscaled.parquet", index=False)
    return {"train": int((~is_test).sum()), "test": int(is_test.sum())}
