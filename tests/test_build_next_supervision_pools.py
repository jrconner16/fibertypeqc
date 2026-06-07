from __future__ import annotations

import pandas as pd

from src.build_next_supervision_pools import build_next_supervision_pools


def test_build_next_supervision_pools_creates_benchmark_and_supervision_rows():
    df = pd.DataFrame(
        [
            {
                "image_id": "img1",
                "label": 1,
                "split": "heldout",
                "fiber_type": "iix",
                "needs_review": False,
                "model_confidence": 0.95,
                "model_margin": 0.30,
                "prob_iib": 0.02,
                "prob_iia": 0.10,
                "prob_iix": 0.88,
                "type1_mean": 0.1,
                "type2_mean": 0.2,
                "type1_coverage": 0.05,
                "type2_coverage": 0.55,
                "type1_cov_x_snr": 0.01,
                "type1_snr_mean": 0.2,
                "type1_snr_p90": 0.3,
                "type2_cov_x_snr": 0.50,
                "type2_snr_mean": 0.90,
                "type2_snr_p90": 1.0,
                "difficulty_score": 0.2,
                "pipeline_needs_review_rate": 0.1,
            },
            {
                "image_id": "img2",
                "label": 2,
                "split": "train",
                "fiber_type": "iia",
                "needs_review": True,
                "model_confidence": 0.55,
                "model_margin": 0.03,
                "prob_iib": 0.10,
                "prob_iia": 0.47,
                "prob_iix": 0.43,
                "type1_mean": 0.2,
                "type2_mean": 0.21,
                "type1_coverage": 0.10,
                "type2_coverage": 0.18,
                "type1_cov_x_snr": 0.02,
                "type1_snr_mean": 0.25,
                "type1_snr_p90": 0.3,
                "type2_cov_x_snr": 0.09,
                "type2_snr_mean": 0.32,
                "type2_snr_p90": 0.4,
                "difficulty_score": 0.7,
                "pipeline_needs_review_rate": 0.4,
            },
            {
                "image_id": "img3",
                "label": 3,
                "split": "train",
                "fiber_type": "iix",
                "needs_review": False,
                "model_confidence": 0.80,
                "model_margin": 0.20,
                "prob_iib": 0.70,
                "prob_iia": 0.05,
                "prob_iix": 0.25,
                "type1_mean": 0.9,
                "type2_mean": 0.1,
                "type1_coverage": 0.65,
                "type2_coverage": 0.05,
                "type1_cov_x_snr": 0.55,
                "type1_snr_mean": 0.88,
                "type1_snr_p90": 0.95,
                "type2_cov_x_snr": 0.01,
                "type2_snr_mean": 0.10,
                "type2_snr_p90": 0.15,
                "difficulty_score": 0.5,
                "pipeline_needs_review_rate": 0.3,
            },
        ]
    )

    out = build_next_supervision_pools(
        df,
        benchmark_quota=1,
        iia_positive_quota=1,
        iia_iix_ambiguity_quota=1,
        iib_positive_quota=1,
        benchmark_max_per_image=1,
        supervision_max_per_image=1,
        seed=0,
    )

    assert not out.empty
    assert set(out["supervision_pool"]) == {"benchmark_enrichment", "supervision_enrichment"}
    assert "proposed_manual_split" in out.columns
    assert set(out["supervision_stratum"]) >= {
        "broad_random_controls",
        "iia_iix_ambiguity",
        "iib_positive",
    }


def test_build_next_supervision_pools_avoids_duplicate_rows_across_strata():
    df = pd.DataFrame(
        [
            {
                "image_id": "img1",
                "label": 1,
                "split": "train",
                "fiber_type": "iia",
                "needs_review": True,
                "model_confidence": 0.6,
                "model_margin": 0.02,
                "prob_iib": 0.05,
                "prob_iia": 0.48,
                "prob_iix": 0.47,
                "type1_mean": 0.1,
                "type2_mean": 0.8,
                "type1_coverage": 0.04,
                "type2_coverage": 0.7,
                "type1_cov_x_snr": 0.01,
                "type1_snr_mean": 0.1,
                "type1_snr_p90": 0.1,
                "type2_cov_x_snr": 0.5,
                "type2_snr_mean": 0.9,
                "type2_snr_p90": 1.0,
                "difficulty_score": 0.9,
                "pipeline_needs_review_rate": 0.5,
            }
        ]
    )

    out = build_next_supervision_pools(
        df,
        benchmark_quota=1,
        iia_positive_quota=1,
        iia_iix_ambiguity_quota=1,
        iib_positive_quota=0,
        benchmark_max_per_image=1,
        supervision_max_per_image=1,
        seed=0,
    )

    keys = list(zip(out["image_id"], out["label"], strict=False))
    assert len(keys) == len(set(keys))


def test_build_next_supervision_pools_balances_iia_iix_ambiguity_when_available():
    df = pd.DataFrame(
        [
            {
                "image_id": "img_iia",
                "label": 1,
                "split": "train",
                "fiber_type": "iia",
                "needs_review": True,
                "model_confidence": 0.55,
                "model_margin": 0.03,
                "prob_iib": 0.05,
                "prob_iia": 0.49,
                "prob_iix": 0.46,
                "type1_mean": 0.2,
                "type2_mean": 0.3,
                "type1_coverage": 0.10,
                "type2_coverage": 0.15,
                "type1_cov_x_snr": 0.02,
                "type1_snr_mean": 0.20,
                "type1_snr_p90": 0.3,
                "type2_cov_x_snr": 0.08,
                "type2_snr_mean": 0.30,
                "type2_snr_p90": 0.4,
                "difficulty_score": 0.8,
                "pipeline_needs_review_rate": 0.4,
            },
            {
                "image_id": "img_iix",
                "label": 2,
                "split": "train",
                "fiber_type": "iix",
                "needs_review": True,
                "model_confidence": 0.54,
                "model_margin": 0.02,
                "prob_iib": 0.04,
                "prob_iia": 0.45,
                "prob_iix": 0.47,
                "type1_mean": 0.2,
                "type2_mean": 0.22,
                "type1_coverage": 0.10,
                "type2_coverage": 0.12,
                "type1_cov_x_snr": 0.02,
                "type1_snr_mean": 0.22,
                "type1_snr_p90": 0.3,
                "type2_cov_x_snr": 0.07,
                "type2_snr_mean": 0.29,
                "type2_snr_p90": 0.35,
                "difficulty_score": 0.8,
                "pipeline_needs_review_rate": 0.4,
            },
        ]
    )

    out = build_next_supervision_pools(
        df,
        benchmark_quota=0,
        iia_positive_quota=0,
        iia_iix_ambiguity_quota=2,
        iib_positive_quota=0,
        benchmark_max_per_image=1,
        supervision_max_per_image=1,
        seed=0,
    )

    ambiguity = out.loc[out["supervision_stratum"].eq("iia_iix_ambiguity")]
    assert set(ambiguity["fiber_type"]) == {"iia", "iix"}
