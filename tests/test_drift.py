"""Population Stability Index, reported alongside validation as a data signal."""

import numpy as np
import pandas as pd
import pytest

from mlops import drift, extract, split


def test_psi_is_near_zero_for_identical_samples():
    values = pd.Series(np.random.default_rng(0).normal(size=1000))
    assert drift.psi(values, values.copy()) == pytest.approx(0.0, abs=1e-6)


def test_psi_grows_when_the_distribution_shifts():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(loc=0.0, size=1000))
    shifted = pd.Series(rng.normal(loc=3.0, size=1000))
    assert drift.psi(reference, shifted) > 0.25


def test_psi_handles_a_constant_feature():
    constant = pd.Series([5.0] * 100)
    assert drift.psi(constant, constant) == 0.0


def test_psi_handles_empty_input():
    assert drift.psi(pd.Series(dtype=float), pd.Series([1.0, 2.0])) == 0.0


def test_bin_count_shrinks_for_small_samples():
    assert drift.choose_bins(80, 80) == 2
    assert drift.choose_bins(200, 200) == 4
    assert drift.choose_bins(10_000, 10_000) == drift.MAX_BINS


def test_bin_count_follows_the_smaller_sample():
    assert drift.choose_bins(10_000, 200) == 4


def test_summarise_flags_features_over_the_threshold():
    verdict = drift.summarise({"a": 0.05, "b": 0.4, "c": 0.3}, threshold=0.2)
    assert verdict["drifted"] is True
    assert verdict["drifted_features"] == ["b", "c"]
    assert verdict["max_psi"] == pytest.approx(0.4)


def test_summarise_stays_quiet_below_the_threshold():
    verdict = drift.summarise({"a": 0.01, "b": 0.02}, threshold=0.2)
    assert verdict["drifted"] is False


def test_summarise_on_no_features():
    assert drift.summarise({}, threshold=0.2)["drifted"] is False


def test_the_two_halves_of_the_split_look_alike(raw_frame):
    """Regression: a fixed 10-bin PSI used to fire on sampling noise alone.

    The halves are taken with the pipeline's own hash split rather than by row
    order - the extract ships in its original order, which is not random.
    """
    features = extract.feature_columns(raw_frame)
    is_test = split.test_mask(raw_frame, test_fraction=0.20)
    scores = drift.dataset_psi(raw_frame[~is_test], raw_frame[is_test], features)
    assert drift.summarise(scores, threshold=0.2)["drifted"] is False


def test_a_shifted_extract_registers_as_drift(raw_frame):
    features = extract.feature_columns(raw_frame)
    shifted = raw_frame.copy()
    shifted[features] = shifted[features] + 2 * shifted[features].std(ddof=0)
    scores = drift.dataset_psi(raw_frame, shifted, features)
    assert drift.summarise(scores, threshold=0.2)["drifted"] is True
