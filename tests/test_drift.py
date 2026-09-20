"""The drift gate decides whether the pipeline retrains, so it gets the most tests."""

import numpy as np
import pandas as pd
import pytest

from mlops import data, drift


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


def test_summarise_flags_features_over_the_threshold():
    verdict = drift.summarise({"a": 0.05, "b": 0.4, "c": 0.3}, threshold=0.2)
    assert verdict["drifted"] is True
    assert verdict["drifted_features"] == ["b", "c"]
    assert verdict["max_psi"] == pytest.approx(0.4)


def test_summarise_stays_quiet_below_the_threshold():
    verdict = drift.summarise({"a": 0.01, "b": 0.02}, threshold=0.2)
    assert verdict["drifted"] is False
    assert verdict["drifted_features"] == []


def test_summarise_on_no_features():
    assert drift.summarise({}, threshold=0.2)["drifted"] is False


def test_unshifted_batch_does_not_trip_the_gate():
    reference, pool = data.split_reference_and_pool(data.load_reference())
    batch = data.make_batch(pool, n_rows=200, shift=0.0)
    scores = drift.dataset_psi(reference, batch, data.feature_columns(reference))
    assert drift.summarise(scores, threshold=0.2)["drifted"] is False


def test_shifted_batch_trips_the_gate():
    reference, pool = data.split_reference_and_pool(data.load_reference())
    batch = data.make_batch(pool, n_rows=200, shift=1.5)
    scores = drift.dataset_psi(reference, batch, data.feature_columns(reference))
    assert drift.summarise(scores, threshold=0.2)["drifted"] is True


def test_bin_count_shrinks_for_small_samples():
    assert drift.choose_bins(80, 80) == 2
    assert drift.choose_bins(200, 200) == 4
    assert drift.choose_bins(10_000, 10_000) == drift.MAX_BINS


def test_bin_count_follows_the_smaller_sample():
    assert drift.choose_bins(10_000, 200) == 4


def test_noise_floor_stays_clear_of_the_threshold_across_seeds():
    """Regression: fixed 10-bin PSI fired on undrifted 200-row batches."""
    reference, pool = data.split_reference_and_pool(data.load_reference())
    features = data.feature_columns(reference)
    for seed in range(5):
        batch = data.make_batch(pool, n_rows=200, shift=0.0, random_state=seed)
        scores = drift.dataset_psi(reference, batch, features)
        assert drift.summarise(scores, threshold=0.2)["drifted"] is False, f"seed {seed}"
