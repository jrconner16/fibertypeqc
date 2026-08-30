from __future__ import annotations

import pandas as pd

from src.compare_roi_boundaries import classify_match_candidates
from src.summarize_roi_matching_coverage import (
    apply_one_to_one_outcomes,
    build_class_summary,
)


def test_classify_match_candidates_records_unmatched_reasons():
    candidates = pd.DataFrame(
        {
            "label_id": [0, 10, 11],
            "centroid_distance_px": [float("nan"), 30.0, 4.0],
        }
    )

    status = classify_match_candidates(candidates, max_centroid_distance=25.0)

    assert status.tolist() == [
        "no_pipeline_label_at_centroid",
        "centroid_distance_exceeds_threshold",
        "matched",
    ]


def test_apply_one_to_one_outcomes_marks_duplicate_claims_ambiguous():
    candidates = pd.DataFrame(
        [
            {
                "roi_index": 0,
                "label_id": 10,
                "centroid_distance_px": 2.0,
                "area_ratio": 1.1,
                "candidate_status": "matched",
            },
            {
                "roi_index": 1,
                "label_id": 10,
                "centroid_distance_px": 3.0,
                "area_ratio": 1.2,
                "candidate_status": "matched",
            },
            {
                "roi_index": 2,
                "label_id": 0,
                "centroid_distance_px": float("nan"),
                "area_ratio": float("nan"),
                "candidate_status": "no_pipeline_label_at_centroid",
            },
        ]
    )

    out = apply_one_to_one_outcomes(candidates)

    assert out["one_to_one_eligible"].tolist() == [False, False, False]
    assert out["match_outcome"].tolist() == [
        "ambiguous_duplicate_assignment",
        "ambiguous_duplicate_assignment",
        "no_pipeline_label_at_centroid",
    ]


def test_build_class_summary_reports_class_specific_coverage():
    all_rois = pd.DataFrame(
        [
            {
                "myosight_label": "iia",
                "match_outcome": "matched_one_to_one",
                "genotype": "wt",
                "group_id": "m1",
            },
            {
                "myosight_label": "iia",
                "match_outcome": "no_pipeline_label_at_centroid",
                "genotype": "wt",
                "group_id": "m1",
            },
            {
                "myosight_label": "iib",
                "match_outcome": "matched_one_to_one",
                "genotype": "het",
                "group_id": "m2",
            },
        ]
    )

    summary = build_class_summary(all_rois)
    overall = summary.loc[summary["scope_type"] == "overall"].set_index("myosight_label")

    assert overall.loc["iia", "total_myosight_rois"] == 2
    assert overall.loc["iia", "matched_one_to_one"] == 1
    assert overall.loc["iia", "one_to_one_coverage"] == 0.5
    assert overall.loc["iib", "one_to_one_coverage"] == 1.0
