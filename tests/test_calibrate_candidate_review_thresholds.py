from __future__ import annotations

import pandas as pd

from src.calibrate_candidate_review_thresholds import (
    _choose_recommended_threshold,
    _review_metrics,
)


def test_review_metrics_reports_capture_and_unflagged_error_rate():
    y_true = pd.Series(["iia", "iib", "iix", "iix"])
    y_pred = pd.Series(["iia", "iia", "iix", "iib"])
    review = pd.Series([False, True, False, False])

    metrics = _review_metrics(y_true, y_pred, review)

    assert metrics["total_errors"] == 2
    assert metrics["review_n"] == 1
    assert metrics["error_capture_rate"] == 0.5
    assert metrics["review_rate"] == 0.25
    assert abs(metrics["unflagged_error_rate"] - (1 / 3)) < 1e-9


def test_choose_recommended_threshold_prefers_low_review_with_frozen_capture():
    metrics = pd.DataFrame(
        [
            {
                "strategy": "pipeline_current",
                "confidence_threshold": None,
                "margin_threshold": None,
                "error_capture_rate": 0.6,
                "review_rate": 0.3,
                "unflagged_error_rate": 0.1,
                "balanced_accuracy": 0.7,
                "accuracy": 0.75,
            },
            {
                "strategy": "candidate_gated_sweep",
                "confidence_threshold": 0.6,
                "margin_threshold": 0.1,
                "error_capture_rate": 0.6,
                "review_rate": 0.2,
                "unflagged_error_rate": 0.11,
                "balanced_accuracy": 0.82,
                "accuracy": 0.8,
            },
            {
                "strategy": "candidate_gated_sweep",
                "confidence_threshold": 0.7,
                "margin_threshold": 0.2,
                "error_capture_rate": 0.7,
                "review_rate": 0.25,
                "unflagged_error_rate": 0.08,
                "balanced_accuracy": 0.82,
                "accuracy": 0.81,
            },
        ]
    )

    recommended = _choose_recommended_threshold(metrics)

    assert recommended["confidence_threshold"] == 0.6
    assert recommended["margin_threshold"] == 0.1
