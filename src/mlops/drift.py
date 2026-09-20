"""Population Stability Index, used as the retrain gate in the DAG.

PSI compares how a feature's distribution is spread across fixed bins in the
reference set versus an incoming batch. The usual reading is:

    < 0.10  stable
    < 0.25  moderate shift, worth watching
    >= 0.25 significant shift

Bin count adapts to sample size. With a fixed 10 bins a 200-row batch drawn from
the *same* distribution already reaches a max PSI of ~0.17 across 30 features -
close enough to a 0.2 threshold to fire on noise. Holding ~50 samples per bin
puts that noise floor near 0.09 while a genuine shift still scores above 10.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON = 1e-6
MIN_SAMPLES_PER_BIN = 50
MAX_BINS = 10
MIN_BINS = 2


def choose_bins(n_reference: int, n_current: int) -> int:
    """Pick a bin count that keeps roughly MIN_SAMPLES_PER_BIN rows in each bin."""
    smaller = min(n_reference, n_current)
    return int(np.clip(smaller // MIN_SAMPLES_PER_BIN, MIN_BINS, MAX_BINS))


def psi(reference: pd.Series, current: pd.Series, bins: int | None = None) -> float:
    """Return the PSI of one feature. 0.0 means the two samples look alike.

    `bins=None` adapts the bin count to the sample size; pass an integer to pin it.
    """
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if ref.size == 0 or cur.size == 0:
        return 0.0

    if bins is None:
        bins = choose_bins(ref.size, cur.size)

    # Quantile edges from the reference so each reference bin starts equally full.
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(ref, quantiles))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_share = np.histogram(ref, bins=edges)[0] / ref.size
    cur_share = np.histogram(cur, bins=edges)[0] / cur.size

    ref_share = np.clip(ref_share, EPSILON, None)
    cur_share = np.clip(cur_share, EPSILON, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def dataset_psi(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    bins: int | None = None,
) -> dict[str, float]:
    """PSI per feature, for the features present in both frames."""
    shared = [f for f in features if f in reference.columns and f in current.columns]
    return {f: psi(reference[f], current[f], bins=bins) for f in shared}


def summarise(scores: dict[str, float], threshold: float) -> dict[str, object]:
    """Collapse per-feature PSI into the verdict the DAG branches on."""
    if not scores:
        return {"max_psi": 0.0, "mean_psi": 0.0, "drifted": False, "drifted_features": []}

    values = list(scores.values())
    drifted = [f for f, v in scores.items() if v >= threshold]
    return {
        "max_psi": float(max(values)),
        "mean_psi": float(sum(values) / len(values)),
        "drifted": bool(drifted),
        "drifted_features": sorted(drifted),
    }
