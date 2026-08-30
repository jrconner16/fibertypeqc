from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fibertypeqc.experimental_features import (
    RELATIVE_CHANNEL_FEATURES,
    add_relative_channel_features,
)


def _sources() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "type_iia.snr_mean": [4.0, -2.0, 0.0],
            "type_iib.snr_mean": [1.0, -3.0, 0.0],
            "type_iia.snr_p90": [8.0, 0.0, 0.0],
            "type_iib.snr_p90": [2.0, -1.0, 0.0],
            "type_iia.coverage_high": [0.8, 0.2, 0.0],
            "type_iib.coverage_high": [0.2, 0.2, 0.0],
        }
    )


def test_relative_features_are_finite_and_zero_safe():
    out = add_relative_channel_features(_sources())

    assert set(RELATIVE_CHANNEL_FEATURES).issubset(out.columns)
    assert np.isfinite(out[list(RELATIVE_CHANNEL_FEATURES)].to_numpy()).all()
    assert out.loc[2, "positive_snr_mean_difference_over_sum"] == pytest.approx(0.0)
    assert out.loc[2, "coverage_difference_over_sum"] == pytest.approx(0.0)


def test_signed_relative_features_reverse_when_channels_are_swapped():
    original = _sources().iloc[[0]].copy()
    swapped = original.rename(
        columns={
            "type_iia.snr_mean": "type_iib.snr_mean",
            "type_iib.snr_mean": "type_iia.snr_mean",
            "type_iia.snr_p90": "type_iib.snr_p90",
            "type_iib.snr_p90": "type_iia.snr_p90",
            "type_iia.coverage_high": "type_iib.coverage_high",
            "type_iib.coverage_high": "type_iia.coverage_high",
        }
    )
    first = add_relative_channel_features(original).iloc[0]
    second = add_relative_channel_features(swapped).iloc[0]

    signed = [
        "log1p_snr_mean_iia_minus_iib",
        "log1p_snr_p90_iia_minus_iib",
        "positive_snr_mean_difference_over_sum",
        "positive_snr_p90_difference_over_sum",
        "coverage_iia_minus_iib",
        "coverage_difference_over_sum",
    ]
    for column in signed:
        assert second[column] == pytest.approx(-first[column])
    assert second["dominant_runner_up_snr_margin"] == pytest.approx(
        first["dominant_runner_up_snr_margin"]
    )
    assert second["mean_p90_coverage_dominance_agreement"] == pytest.approx(
        first["mean_p90_coverage_dominance_agreement"]
    )


def test_difference_over_sum_is_stable_under_common_gain():
    original = _sources().iloc[[0]].copy()
    gained = original.copy()
    snr_columns = [column for column in gained if ".snr_" in column]
    coverage_columns = [column for column in gained if ".coverage_" in column]
    gained[snr_columns] *= 5.0
    gained[coverage_columns] *= 0.5

    first = add_relative_channel_features(original).iloc[0]
    second = add_relative_channel_features(gained).iloc[0]

    for column in [
        "positive_snr_mean_difference_over_sum",
        "positive_snr_p90_difference_over_sum",
        "coverage_difference_over_sum",
    ]:
        assert second[column] == pytest.approx(first[column], abs=1.0e-6)


def test_relative_features_reject_invalid_coverage():
    sources = _sources()
    sources.loc[0, "type_iia.coverage_high"] = 1.1

    with pytest.raises(ValueError, match="within"):
        add_relative_channel_features(sources)
