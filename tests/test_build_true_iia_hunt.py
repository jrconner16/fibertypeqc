from __future__ import annotations

import pandas as pd

from src.build_true_iia_hunt import build_true_iia_hunt


def test_build_true_iia_hunt_prefers_high_type2_low_type1_candidates():
    df = pd.DataFrame(
        [
            {
                "image_id": "img1",
                "label": 1,
                "fiber_type": "iia",
                "needs_review": False,
                "review_rate_bin": "medium",
                "saturation_proxy_suggested": "no_obvious_issue",
                "type1_mean": 1.0,
                "type2_mean": 8.0,
                "type1_coverage": 0.1,
                "type2_coverage": 0.8,
                "type1_cov_x_snr": 0.1,
                "type2_cov_x_snr": 4.0,
                "type1_snr_mean": 0.5,
                "type2_snr_mean": 5.0,
                "type1_snr_p90": 0.6,
                "type2_snr_p90": 5.5,
            },
            {
                "image_id": "img1",
                "label": 2,
                "fiber_type": "iia",
                "needs_review": False,
                "review_rate_bin": "medium",
                "saturation_proxy_suggested": "no_obvious_issue",
                "type1_mean": 6.0,
                "type2_mean": 7.0,
                "type1_coverage": 0.7,
                "type2_coverage": 0.8,
                "type1_cov_x_snr": 3.0,
                "type2_cov_x_snr": 4.1,
                "type1_snr_mean": 4.0,
                "type2_snr_mean": 5.1,
                "type1_snr_p90": 4.2,
                "type2_snr_p90": 5.2,
            },
            {
                "image_id": "img2",
                "label": 3,
                "fiber_type": "iix",
                "needs_review": False,
                "review_rate_bin": "medium",
                "saturation_proxy_suggested": "no_obvious_issue",
                "type1_mean": 0.5,
                "type2_mean": 9.0,
                "type1_coverage": 0.05,
                "type2_coverage": 0.9,
                "type1_cov_x_snr": 0.05,
                "type2_cov_x_snr": 5.0,
                "type1_snr_mean": 0.4,
                "type2_snr_mean": 6.0,
                "type1_snr_p90": 0.5,
                "type2_snr_p90": 6.1,
            },
        ]
    )

    out = build_true_iia_hunt(
        df,
        max_per_image=2,
        top_k=10,
        min_type2_snr_quantile=0.5,
        min_type2_coverage_quantile=0.5,
        max_type1_snr_quantile=0.8,
    )

    assert {1, 3} & set(out["label"])
    assert 2 not in set(out["label"])
    assert "iia_evidence_score" in out.columns


def test_build_true_iia_hunt_can_exclude_saturation_and_high_review_images():
    df = pd.DataFrame(
        [
            {
                "image_id": "img1",
                "label": 1,
                "fiber_type": "iia",
                "needs_review": False,
                "review_rate_bin": "high",
                "saturation_proxy_suggested": "possible_dynamic_range_issue",
                "type1_mean": 0.5,
                "type2_mean": 7.0,
                "type1_coverage": 0.1,
                "type2_coverage": 0.8,
                "type1_cov_x_snr": 0.1,
                "type2_cov_x_snr": 4.0,
                "type1_snr_mean": 0.2,
                "type2_snr_mean": 4.5,
                "type1_snr_p90": 0.3,
                "type2_snr_p90": 4.8,
            },
            {
                "image_id": "img2",
                "label": 2,
                "fiber_type": "iia",
                "needs_review": False,
                "review_rate_bin": "low",
                "saturation_proxy_suggested": "no_obvious_issue",
                "type1_mean": 0.5,
                "type2_mean": 7.0,
                "type1_coverage": 0.1,
                "type2_coverage": 0.8,
                "type1_cov_x_snr": 0.1,
                "type2_cov_x_snr": 4.0,
                "type1_snr_mean": 0.2,
                "type2_snr_mean": 4.5,
                "type1_snr_p90": 0.3,
                "type2_snr_p90": 4.8,
            },
        ]
    )

    out = build_true_iia_hunt(df, top_k=10)

    assert set(out["label"]) == {2}
