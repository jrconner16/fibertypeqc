from __future__ import annotations

import pandas as pd

from src.analyze_iix_iib_boundary import (
    build_iix_iib_summary,
    sample_suspicious_iix_rows,
    score_iix_iib_boundary,
)


def _row(
    *,
    image_id: str,
    label: int,
    split: str,
    fiber_type: str,
    prob_iib: float,
    prob_iix: float,
    type1_mean: float,
    type2_mean: float,
    type1_coverage: float,
    type2_coverage: float,
    type1_snr_mean: float,
    type2_snr_mean: float,
    type1_cov_x_snr: float,
    type2_cov_x_snr: float,
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "label": label,
        "split": split,
        "fiber_type": fiber_type,
        "needs_review": False,
        "model_confidence": 0.8,
        "model_margin": 0.2,
        "prob_iib": prob_iib,
        "prob_iia": 1.0 - prob_iib - prob_iix,
        "prob_iix": prob_iix,
        "type1_mean": type1_mean,
        "type2_mean": type2_mean,
        "type1_coverage": type1_coverage,
        "type2_coverage": type2_coverage,
        "type1_snr_mean": type1_snr_mean,
        "type2_snr_mean": type2_snr_mean,
        "type1_cov_x_snr": type1_cov_x_snr,
        "type2_cov_x_snr": type2_cov_x_snr,
        "genotype": "mdx",
        "timepoint": "4mo",
        "input_kind": "direct_czi",
    }


def test_score_iix_iib_boundary_flags_suspicious_iix_rows():
    df = pd.DataFrame(
        [
            _row(
                image_id="img1",
                label=1,
                split="train",
                fiber_type="iib",
                prob_iib=0.92,
                prob_iix=0.05,
                type1_mean=8.0,
                type2_mean=0.5,
                type1_coverage=0.80,
                type2_coverage=0.08,
                type1_snr_mean=0.90,
                type2_snr_mean=0.10,
                type1_cov_x_snr=0.70,
                type2_cov_x_snr=0.02,
            ),
            _row(
                image_id="img1",
                label=2,
                split="dev",
                fiber_type="iib",
                prob_iib=0.85,
                prob_iix=0.10,
                type1_mean=7.5,
                type2_mean=0.6,
                type1_coverage=0.72,
                type2_coverage=0.12,
                type1_snr_mean=0.82,
                type2_snr_mean=0.15,
                type1_cov_x_snr=0.62,
                type2_cov_x_snr=0.03,
            ),
            _row(
                image_id="img2",
                label=3,
                split="heldout",
                fiber_type="iix",
                prob_iib=0.74,
                prob_iix=0.22,
                type1_mean=7.8,
                type2_mean=0.4,
                type1_coverage=0.76,
                type2_coverage=0.07,
                type1_snr_mean=0.88,
                type2_snr_mean=0.11,
                type1_cov_x_snr=0.69,
                type2_cov_x_snr=0.02,
            ),
            _row(
                image_id="img3",
                label=4,
                split="heldout",
                fiber_type="iix",
                prob_iib=0.10,
                prob_iix=0.82,
                type1_mean=0.8,
                type2_mean=0.6,
                type1_coverage=0.12,
                type2_coverage=0.15,
                type1_snr_mean=0.11,
                type2_snr_mean=0.13,
                type1_cov_x_snr=0.04,
                type2_cov_x_snr=0.04,
            ),
        ]
    )

    scored, thresholds = score_iix_iib_boundary(df, anchor_quantile=0.25)

    assert thresholds["min_prob_iib"] > 0.0
    suspicious = scored.loc[scored["is_suspicious_iix_iib_boundary"]].copy()
    assert list(suspicious["label"]) == [3]
    assert int(suspicious.iloc[0]["iib_anchor_votes"]) >= 5


def test_build_iix_iib_summary_and_sample_prioritize_suspicious_rows():
    df = pd.DataFrame(
        [
            _row(
                image_id="img1",
                label=1,
                split="train",
                fiber_type="iib",
                prob_iib=0.90,
                prob_iix=0.06,
                type1_mean=8.0,
                type2_mean=0.5,
                type1_coverage=0.80,
                type2_coverage=0.08,
                type1_snr_mean=0.90,
                type2_snr_mean=0.10,
                type1_cov_x_snr=0.70,
                type2_cov_x_snr=0.02,
            ),
            _row(
                image_id="img1",
                label=5,
                split="train",
                fiber_type="iib",
                prob_iib=0.86,
                prob_iix=0.09,
                type1_mean=7.2,
                type2_mean=0.6,
                type1_coverage=0.68,
                type2_coverage=0.12,
                type1_snr_mean=0.78,
                type2_snr_mean=0.16,
                type1_cov_x_snr=0.58,
                type2_cov_x_snr=0.03,
            ),
            _row(
                image_id="img2",
                label=2,
                split="heldout",
                fiber_type="iix",
                prob_iib=0.76,
                prob_iix=0.20,
                type1_mean=7.6,
                type2_mean=0.4,
                type1_coverage=0.74,
                type2_coverage=0.07,
                type1_snr_mean=0.86,
                type2_snr_mean=0.09,
                type1_cov_x_snr=0.67,
                type2_cov_x_snr=0.02,
            ),
            _row(
                image_id="img2",
                label=3,
                split="heldout",
                fiber_type="iix",
                prob_iib=0.71,
                prob_iix=0.24,
                type1_mean=7.4,
                type2_mean=0.5,
                type1_coverage=0.70,
                type2_coverage=0.09,
                type1_snr_mean=0.80,
                type2_snr_mean=0.12,
                type1_cov_x_snr=0.60,
                type2_cov_x_snr=0.03,
            ),
            _row(
                image_id="img3",
                label=4,
                split="dev",
                fiber_type="iix",
                prob_iib=0.15,
                prob_iix=0.78,
                type1_mean=1.0,
                type2_mean=0.8,
                type1_coverage=0.15,
                type2_coverage=0.20,
                type1_snr_mean=0.12,
                type2_snr_mean=0.16,
                type1_cov_x_snr=0.04,
                type2_cov_x_snr=0.05,
            ),
        ]
    )

    scored, _ = score_iix_iib_boundary(df, anchor_quantile=0.25)
    summary = build_iix_iib_summary(scored)
    sample = sample_suspicious_iix_rows(
        scored,
        sample_size=2,
        max_per_image=1,
        per_image_top_fraction=0.5,
        max_images=1,
        seed=0,
    )

    assert int(summary.iloc[0]["suspicious_iix_n"]) >= 1
    assert summary.iloc[0]["image_id"] == "img2"
    assert len(sample) == 1
    assert sample.iloc[0]["image_id"] == "img2"
    assert bool(sample.iloc[0]["in_broader_iix_iib_band"])
    assert sample.iloc[0]["supervision_stratum"] == "iix_iib_boundary"
    assert sample.iloc[0]["proposed_manual_split"] == "manual_eval_candidate"


def test_sample_suspicious_iix_rows_uses_broader_per_image_band():
    df = pd.DataFrame(
        [
            _row(
                image_id="img1",
                label=1,
                split="train",
                fiber_type="iib",
                prob_iib=0.92,
                prob_iix=0.05,
                type1_mean=8.0,
                type2_mean=0.5,
                type1_coverage=0.80,
                type2_coverage=0.08,
                type1_snr_mean=0.90,
                type2_snr_mean=0.10,
                type1_cov_x_snr=0.70,
                type2_cov_x_snr=0.02,
            ),
            _row(
                image_id="img1",
                label=2,
                split="heldout",
                fiber_type="iix",
                prob_iib=0.70,
                prob_iix=0.20,
                type1_mean=7.5,
                type2_mean=0.5,
                type1_coverage=0.70,
                type2_coverage=0.08,
                type1_snr_mean=0.82,
                type2_snr_mean=0.11,
                type1_cov_x_snr=0.62,
                type2_cov_x_snr=0.02,
            ),
            _row(
                image_id="img1",
                label=3,
                split="heldout",
                fiber_type="iix",
                prob_iib=0.40,
                prob_iix=0.45,
                type1_mean=5.5,
                type2_mean=0.9,
                type1_coverage=0.45,
                type2_coverage=0.10,
                type1_snr_mean=0.55,
                type2_snr_mean=0.20,
                type1_cov_x_snr=0.25,
                type2_cov_x_snr=0.03,
            ),
            _row(
                image_id="img1",
                label=4,
                split="heldout",
                fiber_type="iix",
                prob_iib=0.10,
                prob_iix=0.80,
                type1_mean=0.8,
                type2_mean=0.7,
                type1_coverage=0.12,
                type2_coverage=0.15,
                type1_snr_mean=0.10,
                type2_snr_mean=0.14,
                type1_cov_x_snr=0.03,
                type2_cov_x_snr=0.04,
            ),
        ]
    )

    scored, _ = score_iix_iib_boundary(df, anchor_quantile=0.25)
    sample = sample_suspicious_iix_rows(
        scored,
        sample_size=5,
        max_per_image=3,
        per_image_top_fraction=0.50,
        max_images=1,
        seed=0,
    )

    assert set(sample["label"]) == {2, 3}
    assert sample["in_broader_iix_iib_band"].all()
