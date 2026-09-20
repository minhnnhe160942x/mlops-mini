"""Dataset access and batch simulation.

The dataset ships inside scikit-learn, so the pipeline never touches the network.
`make_batch` optionally shifts feature means to simulate an upstream data change,
which is what gives the drift gate something to detect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

TARGET = "target"


def load_reference() -> pd.DataFrame:
    """Return the reference dataset the model was last trained against."""
    bundle = load_breast_cancer(as_frame=True)
    frame = bundle.frame.copy()
    frame.columns = [c.replace(" ", "_") for c in frame.columns]
    return frame


def split_reference_and_pool(
    frame: pd.DataFrame, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold back half the rows so incoming batches are genuinely unseen."""
    reference, pool = train_test_split(
        frame, test_size=0.5, random_state=random_state, stratify=frame[TARGET]
    )
    return reference.reset_index(drop=True), pool.reset_index(drop=True)


def make_batch(
    pool: pd.DataFrame,
    n_rows: int = 200,
    shift: float = 0.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Draw a batch from the pool, optionally shifting numeric features.

    `shift` is expressed in standard deviations; 0.0 means "no drift".
    """
    rng = np.random.default_rng(random_state)
    take = min(n_rows, len(pool))
    idx = rng.choice(len(pool), size=take, replace=False)
    batch = pool.iloc[idx].reset_index(drop=True)

    if shift:
        features = [c for c in batch.columns if c != TARGET]
        batch[features] = batch[features] + shift * batch[features].std(ddof=0)
    return batch


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c != TARGET]


def materialise(spec: dict, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the (reference, batch) pair described by a run spec.

    Pushing frames between Airflow tasks through XCom would write hundreds of
    kilobytes into the metadata database on every run. The dataset is bundled and
    the draw is seeded, so each task rebuilds the identical pair from two numbers.
    """
    reference, pool = split_reference_and_pool(load_reference(), random_state)
    batch = make_batch(
        pool,
        n_rows=int(spec["batch_rows"]),
        shift=float(spec["shift"]),
        random_state=random_state,
    )
    return reference, batch
