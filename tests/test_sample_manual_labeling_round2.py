from __future__ import annotations

import pandas as pd

from src.sample_manual_labeling_round2 import sample_manual_labeling_round2


def test_round2_sampler_creates_eval_and_train_pools_with_distinct_rows():
    df = pd.DataFrame(
        [
            {
                "image_id": "held1",
                "label": 1,
                "split": "heldout",
                "fiber_type": "iia",
                "audit_bucket_count": 3,
                "bucket_model_disagreement": False,
                "bucket_low_margin": True,
                "bucket_review_flagged": True,
                "bucket_high_conf_iia": True,
                "bucket_matched_myosight_disparity": False,
                "disparity_bucket": "",
                "model_confidence": 0.95,
                "model_margin": 0.02,
            },
            {
                "image_id": "dev1",
                "label": 2,
                "split": "dev",
                "fiber_type": "iib",
                "audit_bucket_count": 2,
                "bucket_model_disagreement": True,
                "bucket_low_margin": True,
                "bucket_review_flagged": True,
                "bucket_high_conf_iia": False,
                "bucket_matched_myosight_disparity": True,
                "disparity_bucket": "iib_vs_iix",
                "model_confidence": 0.70,
                "model_margin": 0.03,
            },
            {
                "image_id": "train1",
                "label": 3,
                "split": "train",
                "fiber_type": "iix",
                "audit_bucket_count": 1,
                "bucket_model_disagreement": False,
                "bucket_low_margin": False,
                "bucket_review_flagged": False,
                "bucket_high_conf_iia": False,
                "bucket_matched_myosight_disparity": False,
                "disparity_bucket": "",
                "model_confidence": 0.98,
                "model_margin": 0.20,
            },
            {
                "image_id": "train1",
                "label": 4,
                "split": "train",
                "fiber_type": "iia",
                "audit_bucket_count": 2,
                "bucket_model_disagreement": False,
                "bucket_low_margin": False,
                "bucket_review_flagged": False,
                "bucket_high_conf_iia": True,
                "bucket_matched_myosight_disparity": True,
                "disparity_bucket": "iix_vs_iia",
                "model_confidence": 0.94,
                "model_margin": 0.08,
            },
            {
                "image_id": "train2",
                "label": 5,
                "split": "train",
                "fiber_type": "iia",
                "audit_bucket_count": 2,
                "bucket_model_disagreement": False,
                "bucket_low_margin": True,
                "bucket_review_flagged": True,
                "bucket_high_conf_iia": True,
                "bucket_matched_myosight_disparity": True,
                "disparity_bucket": "iia_vs_iix",
                "model_confidence": 0.88,
                "model_margin": 0.03,
            },
            {
                "image_id": "train3",
                "label": 6,
                "split": "train",
                "fiber_type": "iib",
                "audit_bucket_count": 1,
                "bucket_model_disagreement": False,
                "bucket_low_margin": False,
                "bucket_review_flagged": False,
                "bucket_high_conf_iia": False,
                "bucket_matched_myosight_disparity": False,
                "disparity_bucket": "",
                "model_confidence": 0.97,
                "model_margin": 0.15,
            },
        ]
    )

    sampled = sample_manual_labeling_round2(
        df,
        eval_quotas={
            "eval_iia_focus": 1,
            "eval_iib_iix_ambiguity": 1,
            "eval_myosight_disparity": 1,
            "eval_review_flagged_controls": 1,
            "eval_model_disagreement": 1,
            "eval_easy_controls": 1,
        },
        train_quotas={
            "train_iia_focus": 1,
            "train_iia_ambiguity": 1,
            "train_iib_iix_ambiguity": 1,
            "train_myosight_disparity": 1,
            "train_review_flagged_controls": 1,
            "train_easy_controls": 1,
        },
        eval_max_per_image=2,
        train_max_per_image=2,
        seed=0,
    )

    assert set(sampled["manual_round2_pool"]) == {
        "manual_eval_candidate",
        "manual_train_candidate",
    }
    assert sampled[["image_id", "label"]].drop_duplicates().shape[0] == len(sampled)
    assert "manual_round2_stratum" in sampled.columns


def test_round2_sampler_respects_pool_specific_image_caps():
    rows = []
    for label in range(1, 7):
        rows.append(
            {
                "image_id": "img1",
                "label": label,
                "split": "train",
                "fiber_type": "iia" if label % 2 else "iix",
                "audit_bucket_count": 2,
                "bucket_model_disagreement": False,
                "bucket_low_margin": True,
                "bucket_review_flagged": True,
                "bucket_high_conf_iia": label % 2 == 1,
                "bucket_matched_myosight_disparity": False,
                "disparity_bucket": "",
                "model_confidence": 0.95,
                "model_margin": 0.02,
            }
        )
    df = pd.DataFrame(rows)

    sampled = sample_manual_labeling_round2(
        df,
        eval_quotas={
            "eval_iia_focus": 4,
            "eval_iib_iix_ambiguity": 0,
            "eval_myosight_disparity": 0,
            "eval_review_flagged_controls": 0,
            "eval_model_disagreement": 0,
            "eval_easy_controls": 0,
        },
        train_quotas={
            "train_iia_focus": 4,
            "train_iia_ambiguity": 4,
            "train_iib_iix_ambiguity": 0,
            "train_myosight_disparity": 0,
            "train_review_flagged_controls": 0,
            "train_easy_controls": 0,
        },
        eval_max_per_image=2,
        train_max_per_image=3,
        seed=0,
    )

    eval_counts = (
        sampled.loc[sampled["manual_round2_pool"] == "manual_eval_candidate"]
        .groupby("image_id")
        .size()
    )
    train_counts = (
        sampled.loc[sampled["manual_round2_pool"] == "manual_train_candidate"]
        .groupby("image_id")
        .size()
    )

    assert eval_counts.max() <= 2
    assert train_counts.max() <= 3
