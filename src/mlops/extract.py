"""Reading the raw extract and freezing a per-run snapshot of it."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ID = "sample_id"
LABEL = "diagnosis"
VALID_LABELS = {"M", "B"}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Every column that is a measurement, i.e. not the id and not the label."""
    return [c for c in frame.columns if c not in (ID, LABEL)]


def run_dir(staging_root: Path, ds: str) -> Path:
    """One folder per logical date.

    Re-running a date overwrites that date's folder and touches nothing else,
    which is what makes a re-run safe to do at any time.
    """
    directory = staging_root / ds
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ingest(raw_path: Path, staging_root: Path, ds: str) -> dict:
    """Copy the extract into this run's folder and freeze it as parquet."""
    if not raw_path.exists():
        raise FileNotFoundError(f"source extract missing: {raw_path}")

    frame = pd.read_csv(raw_path)
    missing = {ID, LABEL} - set(frame.columns)
    if missing:
        raise ValueError(f"extract is missing required columns: {sorted(missing)}")

    destination = run_dir(staging_root, ds) / "raw.parquet"
    frame.to_parquet(destination, index=False)
    return {"rows": len(frame), "cols": frame.shape[1], "path": str(destination)}
