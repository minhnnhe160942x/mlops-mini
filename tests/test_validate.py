"""The 5% gate decides whether a run continues, so it gets the most tests."""

import json

import numpy as np
import pandas as pd

from mlops import extract, validate


def test_a_clean_extract_passes(snapshot, staging):
    verdict = validate.validate(snapshot, staging, "2026-08-25", max_bad_fraction=0.05)
    assert verdict["passed"] is True
    assert verdict["bad_fraction"] == 0.0
    assert verdict["rejected_rows"] == 0
    assert verdict["clean_rows"] == 569


def test_nulls_are_caught(tiny_frame):
    frame = tiny_frame.copy()
    frame.loc[0, "mean_radius"] = np.nan
    assert validate.find_problems(frame)["null"].sum() == 1


def test_negative_measurements_are_caught(tiny_frame):
    frame = tiny_frame.copy()
    frame.loc[1, "mean_area"] = -5.0
    assert validate.find_problems(frame)["negative"].sum() == 1


def test_an_unknown_label_is_caught(tiny_frame):
    frame = tiny_frame.copy()
    frame.loc[2, extract.LABEL] = "X"
    assert validate.find_problems(frame)["bad_label"].sum() == 1


def test_a_duplicate_id_is_caught_once_not_twice(tiny_frame):
    frame = tiny_frame.copy()
    frame.loc[3, extract.ID] = frame.loc[0, extract.ID]
    problems = validate.find_problems(frame)
    assert problems["duplicate"].sum() == 1  # the first occurrence is kept


def test_an_extreme_outlier_is_caught():
    """The cutoff is a multiple of the 99th percentile, so it needs enough rows.

    On a handful of rows a single huge value drags the percentile up with it and
    hides itself; this is what the rule looks like on a realistic extract.
    """
    frame = pd.DataFrame(
        {
            extract.ID: [f"WDBC-{i:04d}" for i in range(100)],
            "mean_radius": [10.0] * 100,
            "mean_area": [100.0] * 99 + [100_000.0],
            extract.LABEL: ["B"] * 100,
        }
    )
    assert validate.find_problems(frame)["outlier"].sum() == 1


def test_a_run_fails_once_too_many_rows_are_bad(snapshot, staging, raw_frame):
    frame = raw_frame.copy()
    bad = int(len(frame) * 0.12)
    frame.loc[: bad - 1, extract.LABEL] = "X"
    path = staging / "dirty.parquet"
    frame.to_parquet(path, index=False)

    verdict = validate.validate(path, staging, "2026-08-26", max_bad_fraction=0.05)
    assert verdict["passed"] is False
    assert verdict["bad_fraction"] > 0.05
    assert "limit is 5%" in validate.failure_message(verdict)


def test_a_few_bad_rows_are_quarantined_without_failing(snapshot, staging, raw_frame):
    frame = raw_frame.copy()
    frame.loc[:4, extract.LABEL] = "X"  # 5 of 569 rows, under 1%
    path = staging / "slightly-dirty.parquet"
    frame.to_parquet(path, index=False)

    verdict = validate.validate(path, staging, "2026-08-27", max_bad_fraction=0.05)
    assert verdict["passed"] is True
    assert verdict["rejected_rows"] == 5
    assert verdict["clean_rows"] == 564


def test_clean_and_rejected_rows_are_written_separately(snapshot, staging, raw_frame):
    frame = raw_frame.copy()
    frame.loc[:2, extract.LABEL] = "X"
    path = staging / "mixed.parquet"
    frame.to_parquet(path, index=False)

    validate.validate(path, staging, "2026-08-28", max_bad_fraction=0.05)
    directory = staging / "2026-08-28"
    assert len(pd.read_parquet(directory / "rejected.parquet")) == 3
    assert len(pd.read_parquet(directory / "clean.parquet")) == 566


def test_a_validation_report_is_written(snapshot, staging):
    validate.validate(snapshot, staging, "2026-08-25", max_bad_fraction=0.05)
    report = json.loads((staging / "2026-08-25" / "validation_report.json").read_text())
    assert report["passed"] is True
    assert set(report) >= {"null", "negative", "bad_label", "duplicate", "outlier"}
