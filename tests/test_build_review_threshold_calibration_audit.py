from __future__ import annotations

import pandas as pd

from src.build_review_threshold_calibration_audit import (
    _label_band,
    _parse_class_quotas,
    _quota_rows,
    _sampling_priority,
)


def test_parse_class_quotas():
    out = _parse_class_quotas("iix:80,iib:70,iia:50")
    assert out == {"iix": 80, "iib": 70, "iia": 50}


def test_quota_rows_preserve_totals():
    quotas = _quota_rows({"iix": 80, "iib": 70, "iia": 50})
    totals = quotas.groupby("fiber_type")["quota"].sum().to_dict()
    assert totals == {"iix": 80, "iib": 70, "iia": 50}


def test_label_band_marks_low_threshold_and_high():
    out = _label_band(
        pd.Series([0.70, 0.83, 0.93, 0.88]),
        pd.Series([0.04, 0.05, 0.20, 0.02]),
        confidence_threshold=0.85,
        margin_threshold=0.05,
        low_confidence_cutoff=0.75,
        high_confidence_cutoff=0.90,
        confidence_band_halfwidth=0.05,
        margin_band_halfwidth=0.05,
    )
    assert list(out) == ["low", "threshold_band", "high", "threshold_band"]


def test_sampling_priority_prefers_near_threshold_for_threshold_band():
    df = pd.DataFrame(
        {
            "candidate_model_confidence": [0.84, 0.70, 0.97],
            "candidate_model_margin": [0.05, 0.01, 0.30],
            "calibration_band": ["threshold_band", "low", "high"],
        }
    )
    score = _sampling_priority(df, confidence_threshold=0.85, margin_threshold=0.05)
    assert score.iloc[0] < 0
    assert score.iloc[2] > 0
