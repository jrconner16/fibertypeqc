from __future__ import annotations

import pandas as pd

from src.build_candidate_split_manifest import build_candidate_split_manifest


def test_build_candidate_split_manifest_populates_filename_and_summary_fields():
    input_manifest = pd.DataFrame(
        {
            "image_id": [
                "001_mouse467_mdx_TA_1month",
                "mouse1_mdxjag_12mo_slide4_TA",
                "section001_mouse355_mdx_TA",
            ],
            "input_kind": ["direct_czi", "direct_czi", "section_tiff_export"],
            "training_image": [True, False, False],
            "source_image": [
                "/tmp/001_mouse467_mdx_TA_1month.czi",
                "/tmp/mouse1_mdxjag_12mo_slide4_TA.czi",
                "/tmp/355_4mo_all_sections_slide10/section001_mouse355_mdx_TA.tif",
            ],
        }
    )
    consolidated_summary = pd.DataFrame(
        {
            "image_id": [
                "001_mouse467_mdx_TA_1month",
                "mouse1_mdxjag_12mo_slide4_TA",
                "section001_mouse355_mdx_TA",
            ],
            "validation_input_kind": ["direct_czi", "direct_czi", "section_series_export"],
            "pipeline_total_fibers": [3300, 5100, 1600],
            "pipeline_needs_review_rate": [0.08, 0.20, 0.12],
            "pipeline_signal_warning_rate": [0.12, 0.30, 0.22],
            "pipeline_iib_pct": [0.5, 0.6, 0.7],
            "pipeline_iia_pct": [0.2, 0.15, 0.1],
            "pipeline_iix_pct": [0.3, 0.25, 0.2],
            "fiber_count_abs_pct_diff": [5.0, 12.0, 28.0],
        }
    )

    out = build_candidate_split_manifest(input_manifest, consolidated_summary)

    by_id = out.set_index("image_id")

    assert set(out["image_id"]) == {
        "001_mouse467_mdx_TA_1month",
        "mouse1_mdxjag_12mo_slide4_TA",
        "section001_mouse355_mdx_TA",
    }
    assert by_id.loc["001_mouse467_mdx_TA_1month", "genotype"] == "mdx"
    assert by_id.loc["section001_mouse355_mdx_TA", "genotype"] == "mdx"
    assert by_id.loc["mouse1_mdxjag_12mo_slide4_TA", "genotype"] == "mdxJag"
    assert by_id.loc["001_mouse467_mdx_TA_1month", "timepoint"] == "1mo"
    assert by_id.loc["section001_mouse355_mdx_TA", "timepoint"] == "4mo"
    assert by_id.loc["mouse1_mdxjag_12mo_slide4_TA", "timepoint"] == "12mo"
    assert set(out["panel_type"]) == {"baseline_iib_iia_membrane_residual_iix"}
    assert by_id.loc["001_mouse467_mdx_TA_1month", "used_in_alpha_training"]
    assert not by_id.loc["section001_mouse355_mdx_TA", "used_in_alpha_training"]
    assert not by_id.loc["mouse1_mdxjag_12mo_slide4_TA", "used_in_alpha_training"]
    assert set(out["candidate_included"]) == {True}
    assert set(out["has_myosight_comparison"]) == {True}
    assert set(out["fiber_count_bin"]) == {"small", "medium", "large"}
    assert set(out["split_suggested"]) == {"train", "heldout"}
    assert set(out["quality_bin_suggested"]) == {"cleaner", "mixed", "harder"}
    assert "saturation_proxy_suggested" in out.columns
    assert "difficulty_score" in out.columns
    assert by_id.loc["mouse1_mdxjag_12mo_slide4_TA", "review_rate_bin"] == "high"
    assert by_id.loc["001_mouse467_mdx_TA_1month", "review_rate_bin"] == "low"
    assert by_id.loc["001_mouse467_mdx_TA_1month", "split_suggested"] == "train"


def test_build_candidate_split_manifest_requires_expected_columns():
    input_manifest = pd.DataFrame({"image_id": ["x"]})
    consolidated_summary = pd.DataFrame({"image_id": ["x"]})

    try:
        build_candidate_split_manifest(input_manifest, consolidated_summary)
    except ValueError as exc:
        assert "input manifest is missing required columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing input manifest columns")
