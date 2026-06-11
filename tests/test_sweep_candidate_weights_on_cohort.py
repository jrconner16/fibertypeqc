from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.sweep_candidate_weights_on_cohort import _aggregate_gap_metrics, _summarize_pair_frames


def test_aggregate_gap_metrics() -> None:
    summary = pd.DataFrame(
        [
            {
                "pipeline_iib_pct": 0.50,
                "myosight_iib_pct": 0.60,
                "pipeline_iix_pct": 0.40,
                "myosight_iix_pct": 0.30,
            },
            {
                "pipeline_iib_pct": 0.70,
                "myosight_iib_pct": 0.65,
                "pipeline_iix_pct": 0.20,
                "myosight_iix_pct": 0.25,
            },
        ]
    )
    out = _aggregate_gap_metrics(summary)
    assert out["cohort_mean_abs_gap_iib"] == pytest.approx(0.075)
    assert out["cohort_mean_abs_gap_iix"] == pytest.approx(0.075)


def test_summarize_pair_frames_counts_and_warnings() -> None:
    myo = pd.DataFrame(
        {
            "Label": ["iib", "iix", "iix"],
            "Area": [100.0, 110.0, 120.0],
        }
    )
    pipe = pd.DataFrame(
        {
            "fiber_type": ["iib", "iib", "iix", "exclude"],
            "area_um2": [90.0, 95.0, 105.0, 80.0],
            "needs_review": [False, True, False, False],
            "has_signal_qc_warning": [True, False, True, False],
        }
    )
    row = _summarize_pair_frames("img1", Path("myo.txt"), myo, pipe)
    assert row["myosight_total_fibers"] == 3
    assert row["pipeline_total_fibers"] == 4
    assert row["myosight_iib_n"] == 1
    assert row["pipeline_iib_n"] == 2
    assert row["pipeline_needs_review_n"] == 1
    assert row["pipeline_signal_warning_n"] == 2
