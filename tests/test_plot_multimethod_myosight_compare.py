from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.plot_multimethod_myosight_compare import (
    UNCALIBRATED_REVIEW_LABEL,
    build_method_long,
    build_review_long,
    load_pipeline_summary,
    parse_method_spec,
)


def _write_summary(path: Path, image_id: str, pipeline_iia_pct: float) -> None:
    pd.DataFrame(
        [
            {
                "image_id": image_id,
                "validation_input_kind": "direct_czi",
                "training_or_heldout": "heldout",
                "myosight_total_fibers": 100,
                "pipeline_total_fibers": 110,
                "pipeline_needs_review_n": 11,
                "pipeline_signal_warning_n": 5,
                "myosight_iib_n": 40,
                "pipeline_iib_n": 35,
                "myosight_iib_pct": 0.40,
                "pipeline_iib_pct": 0.35,
                "iib_pct_diff_pipeline_minus_myosight": -0.05,
                "myosight_iia_n": 10,
                "pipeline_iia_n": int(round(110 * pipeline_iia_pct)),
                "myosight_iia_pct": 0.10,
                "pipeline_iia_pct": pipeline_iia_pct,
                "iia_pct_diff_pipeline_minus_myosight": pipeline_iia_pct - 0.10,
                "myosight_iix_n": 50,
                "pipeline_iix_n": 110 - 35 - int(round(110 * pipeline_iia_pct)),
                "myosight_iix_pct": 0.50,
                "pipeline_iix_pct": 1.0 - 0.35 - pipeline_iia_pct,
                "iix_pct_diff_pipeline_minus_myosight": (1.0 - 0.35 - pipeline_iia_pct) - 0.50,
                "myosight_area_median": 1200.0,
                "pipeline_area_median": 1300.0,
            }
        ]
    ).to_csv(path, index=False)


def test_parse_method_spec() -> None:
    label, path = parse_method_spec("Frozen=outputs/demo.csv")
    assert label == "Frozen"
    assert path == Path("outputs/demo.csv")


def test_build_method_long_and_review_long(tmp_path: Path) -> None:
    frozen_path = tmp_path / "frozen.csv"
    candidate_path = tmp_path / "candidate.csv"
    _write_summary(frozen_path, "001_mouse1_mdx_TA_1month", 0.12)
    _write_summary(candidate_path, "001_mouse1_mdx_TA_1month", 0.18)

    frozen = load_pipeline_summary(frozen_path, "Frozen")
    candidate = load_pipeline_summary(candidate_path, "Candidate")

    long = build_method_long([frozen, candidate])
    review = build_review_long([frozen, candidate])

    assert {"MyoSight", "Frozen", "Candidate"} == set(long["method"].astype(str))
    assert set(review["method"].astype(str)) == {"Frozen", "Candidate"}
    assert "fiber_pct_by_type" in set(long["metric"])
    assert UNCALIBRATED_REVIEW_LABEL in set(review["metric"])
    assert review["rate_pct"].max() > 0
