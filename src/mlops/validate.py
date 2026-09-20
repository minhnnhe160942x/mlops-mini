"""Data quality gate.

Bad rows are quarantined rather than dropped silently, and the run is failed only
when too many of them are bad — one malformed row is noise, a tenth of the file
is a broken upstream extract and nothing downstream should be trusted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .extract import ID, LABEL, VALID_LABELS, feature_columns, run_dir

FEATURE_MINIMUM = 0.0  # every WDBC measurement is a non-negative size
OUTLIER_COLUMN = "mean_area"
OUTLIER_MULTIPLIER = 20.0


def find_problems(frame: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per defect, one row per input row."""
    numeric = feature_columns(frame)
    problems = pd.DataFrame(index=frame.index)
    problems["null"] = frame[numeric].isna().any(axis=1)
    problems["negative"] = (frame[numeric] < FEATURE_MINIMUM).any(axis=1)
    problems["bad_label"] = ~frame[LABEL].isin(VALID_LABELS)
    problems["duplicate"] = frame.duplicated(subset=ID, keep="first")

    if OUTLIER_COLUMN in frame.columns:
        cutoff = frame[OUTLIER_COLUMN].quantile(0.99) * OUTLIER_MULTIPLIER
        problems["outlier"] = frame[OUTLIER_COLUMN] > cutoff
    else:
        problems["outlier"] = False
    return problems


def validate(
    source_path: Path, staging_root: Path, ds: str, max_bad_fraction: float
) -> dict:
    """Split the snapshot into clean and rejected rows and report on both.

    Returns a verdict; it never raises. The DAG decides what to do with
    `passed` so the failure stays visible in one place.
    """
    frame = pd.read_parquet(source_path)
    problems = find_problems(frame)

    bad = problems.any(axis=1)
    counts = {key: int(value) for key, value in problems.sum().items()}
    fraction = float(bad.mean()) if len(frame) else 0.0

    directory = run_dir(staging_root, ds)
    clean_path = directory / "clean.parquet"
    frame[~bad].to_parquet(clean_path, index=False)
    frame[bad].to_parquet(directory / "rejected.parquet", index=False)

    verdict = {
        "path": str(clean_path),
        "clean_rows": int((~bad).sum()),
        "rejected_rows": int(bad.sum()),
        "bad_fraction": fraction,
        "limit": max_bad_fraction,
        "passed": fraction <= max_bad_fraction,
        **counts,
    }
    (directory / "validation_report.json").write_text(json.dumps(verdict, indent=2))
    return verdict


def failure_message(verdict: dict) -> str:
    return (
        f"{verdict['bad_fraction']:.1%} of rows rejected, limit is {verdict['limit']:.0%}"
    )
