"""The pipeline rebuilds its data from a small spec instead of pushing frames through XCom."""

import pandas.testing as pdt

from mlops import data


def test_materialise_is_deterministic():
    spec = {"shift": 1.5, "batch_rows": 120}
    ref_a, batch_a = data.materialise(spec)
    ref_b, batch_b = data.materialise(spec)
    pdt.assert_frame_equal(ref_a, ref_b)
    pdt.assert_frame_equal(batch_a, batch_b)


def test_materialise_honours_the_requested_size():
    _, batch = data.materialise({"shift": 0.0, "batch_rows": 75})
    assert len(batch) == 75


def test_materialise_shift_changes_the_batch_but_not_the_reference():
    ref_clean, batch_clean = data.materialise({"shift": 0.0, "batch_rows": 100})
    ref_shifted, batch_shifted = data.materialise({"shift": 2.0, "batch_rows": 100})
    pdt.assert_frame_equal(ref_clean, ref_shifted)
    features = data.feature_columns(batch_clean)
    assert not batch_clean[features].equals(batch_shifted[features])


def test_materialise_matches_the_drift_gate_expectation():
    from mlops import drift

    reference, batch = data.materialise({"shift": 0.0, "batch_rows": 200})
    scores = drift.dataset_psi(reference, batch, data.feature_columns(reference))
    assert drift.summarise(scores, threshold=0.2)["drifted"] is False
