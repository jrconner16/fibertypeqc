"""Deterministic experimental feature transforms kept outside stable output schemas."""

from __future__ import annotations

import numpy as np
import pandas as pd

RELATIVE_CHANNEL_FEATURES = (
    "log1p_snr_mean_iia_minus_iib",
    "log1p_snr_p90_iia_minus_iib",
    "positive_snr_mean_difference_over_sum",
    "positive_snr_p90_difference_over_sum",
    "coverage_iia_minus_iib",
    "coverage_difference_over_sum",
    "dominant_runner_up_snr_margin",
    "mean_p90_coverage_dominance_agreement",
)

_RELATIVE_SOURCE_COLUMNS = (
    "type_iia.snr_mean",
    "type_iib.snr_mean",
    "type_iia.snr_p90",
    "type_iib.snr_p90",
    "type_iia.coverage_high",
    "type_iib.coverage_high",
)


def _difference_over_sum(first: np.ndarray, second: np.ndarray, epsilon: float) -> np.ndarray:
    return (first - second) / (first + second + epsilon)


def add_relative_channel_features(
    table: pd.DataFrame,
    *,
    epsilon: float = 1.0e-6,
) -> pd.DataFrame:
    """Add the predeclared IIa/IIb F2 transforms without fitting on cohort data."""
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a finite positive number.")
    missing = sorted(set(_RELATIVE_SOURCE_COLUMNS).difference(table.columns))
    if missing:
        raise ValueError(f"Relative-channel feature sources are missing: {missing}")

    sources = table.loc[:, _RELATIVE_SOURCE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(sources.to_numpy(dtype=np.float64))
    if not finite.all():
        bad = sources.columns[~finite.all(axis=0)].tolist()
        raise ValueError(f"Relative-channel feature sources contain non-finite values: {bad}")

    coverage = sources[["type_iia.coverage_high", "type_iib.coverage_high"]].to_numpy()
    if ((coverage < 0) | (coverage > 1)).any():
        raise ValueError("Coverage sources must be within [0, 1].")

    iia_mean = np.maximum(sources["type_iia.snr_mean"].to_numpy(dtype=float), 0.0)
    iib_mean = np.maximum(sources["type_iib.snr_mean"].to_numpy(dtype=float), 0.0)
    iia_p90 = np.maximum(sources["type_iia.snr_p90"].to_numpy(dtype=float), 0.0)
    iib_p90 = np.maximum(sources["type_iib.snr_p90"].to_numpy(dtype=float), 0.0)
    iia_coverage = sources["type_iia.coverage_high"].to_numpy(dtype=float)
    iib_coverage = sources["type_iib.coverage_high"].to_numpy(dtype=float)

    out = table.copy()
    out["log1p_snr_mean_iia_minus_iib"] = np.log1p(iia_mean) - np.log1p(iib_mean)
    out["log1p_snr_p90_iia_minus_iib"] = np.log1p(iia_p90) - np.log1p(iib_p90)
    out["positive_snr_mean_difference_over_sum"] = _difference_over_sum(iia_mean, iib_mean, epsilon)
    out["positive_snr_p90_difference_over_sum"] = _difference_over_sum(iia_p90, iib_p90, epsilon)
    out["coverage_iia_minus_iib"] = iia_coverage - iib_coverage
    out["coverage_difference_over_sum"] = _difference_over_sum(iia_coverage, iib_coverage, epsilon)
    out["dominant_runner_up_snr_margin"] = np.abs(
        out["positive_snr_mean_difference_over_sum"].to_numpy(dtype=float)
    )

    votes = np.column_stack(
        [
            np.sign(iia_mean - iib_mean),
            np.sign(iia_p90 - iib_p90),
            np.sign(iia_coverage - iib_coverage),
        ]
    )
    vote_counts = np.column_stack([(votes == value).sum(axis=1) for value in (-1.0, 0.0, 1.0)])
    out["mean_p90_coverage_dominance_agreement"] = vote_counts.max(axis=1) / votes.shape[1]
    return out
