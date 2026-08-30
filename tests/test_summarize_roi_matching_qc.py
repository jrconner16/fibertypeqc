from __future__ import annotations

import pandas as pd

from src.summarize_roi_matching_qc import audit_match_rows, summarize_image


def test_audit_match_rows_excludes_all_sides_of_ambiguous_assignments():
    matches = pd.DataFrame(
        [
            {"roi_index": 0, "label_id": 10, "centroid_distance_px": 2.0, "area_ratio": 1.1},
            {"roi_index": 1, "label_id": 11, "centroid_distance_px": 3.0, "area_ratio": 1.2},
            {"roi_index": 2, "label_id": 11, "centroid_distance_px": 4.0, "area_ratio": 1.3},
            {"roi_index": 3, "label_id": 12, "centroid_distance_px": 5.0, "area_ratio": 1.4},
            {"roi_index": 3, "label_id": 13, "centroid_distance_px": 6.0, "area_ratio": 1.5},
        ]
    )

    audited = audit_match_rows(matches)

    assert audited["one_to_one_eligible"].tolist() == [True, False, False, False, False]
    assert audited.loc[1, "qc_exclusion_reason"] == "duplicate_pipeline_label"
    assert audited.loc[3, "qc_exclusion_reason"] == "duplicate_roi_index"


def test_summarize_image_uses_only_unique_rows_for_qc_metrics():
    matches = pd.DataFrame(
        [
            {"roi_index": 0, "label_id": 10, "centroid_distance_px": 2.0, "area_ratio": 1.1},
            {"roi_index": 1, "label_id": 11, "centroid_distance_px": 50.0, "area_ratio": 9.0},
            {"roi_index": 2, "label_id": 11, "centroid_distance_px": 60.0, "area_ratio": 10.0},
        ]
    )

    summary = summarize_image(audit_match_rows(matches), total_rois=5, results_rows=5)

    assert summary["raw_matched_rows"] == 3
    assert summary["raw_match_rate"] == 0.6
    assert summary["one_to_one_eligible_rows"] == 1
    assert summary["one_to_one_match_rate"] == 0.2
    assert summary["duplicate_pipeline_label_ids"] == 1
    assert summary["duplicate_pipeline_label_rows"] == 2
    assert summary["median_centroid_distance_px"] == 2.0
    assert summary["median_area_ratio"] == 1.1


def test_audit_match_rows_rejects_missing_columns():
    matches = pd.DataFrame([{"roi_index": 0, "label_id": 1}])

    try:
        audit_match_rows(matches)
    except ValueError as exc:
        assert "area_ratio" in str(exc)
        assert "centroid_distance_px" in str(exc)
    else:
        raise AssertionError("Expected missing columns to be rejected")
