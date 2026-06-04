from __future__ import annotations

import pandas as pd

from src.build_combined_audit_set import build_combined_audit_set


def test_build_combined_audit_set_combines_buckets():
    feature_table = pd.DataFrame(
        [
            {
                "_merge_image_id": "img1",
                "image_id": "img1",
                "label": 1,
                "fiber_type": "iia",
                "needs_review": False,
                "model_confidence": 0.99,
                "model_margin": 0.05,
                "split": "dev",
            },
            {
                "_merge_image_id": "img1",
                "image_id": "img1",
                "label": 2,
                "fiber_type": "iix",
                "needs_review": True,
                "model_confidence": 0.97,
                "model_margin": 0.9,
                "split": "dev",
            },
            {
                "_merge_image_id": "img2",
                "image_id": "img2",
                "label": 1,
                "fiber_type": "iib",
                "needs_review": False,
                "model_confidence": 0.4,
                "model_margin": 0.8,
                "split": "train",
            },
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "_merge_image_id": "img1",
                "label": 1,
                "candidate_pred_baseline_rf": "iia",
                "candidate_pred_expanded_rf": "iix",
            },
            {
                "_merge_image_id": "img1",
                "label": 2,
                "candidate_pred_baseline_rf": "iix",
                "candidate_pred_expanded_rf": "iix",
            },
        ]
    )
    matched = pd.DataFrame(
        [
            {
                "_merge_image_id": "img1",
                "label": 2,
                "myosight_label": "iib",
                "disparity_bucket": "iib_vs_iix",
            }
        ]
    )

    out = build_combined_audit_set(feature_table, predictions, matched)

    assert len(out) == 2
    row1 = out.loc[out["label"] == 1].iloc[0]
    assert bool(row1["bucket_model_disagreement"])
    assert bool(row1["bucket_low_margin"])
    assert bool(row1["bucket_high_conf_iia"])

    row2 = out.loc[out["label"] == 2].iloc[0]
    assert bool(row2["bucket_review_flagged"])
    assert bool(row2["bucket_high_conf_iix"])
    assert bool(row2["bucket_matched_myosight_disparity"])
    assert "matched_myosight_disparity" in row2["audit_bucket_list"]


def test_build_combined_audit_set_uses_fiber_type_as_baseline_when_predictions_missing():
    feature_table = pd.DataFrame(
        [
            {
                "_merge_image_id": "img1",
                "image_id": "img1",
                "label": 1,
                "fiber_type": "iix",
                "needs_review": False,
                "model_confidence": 0.3,
                "model_margin": 0.1,
                "split": "dev",
            }
        ]
    )
    out = build_combined_audit_set(feature_table, predictions=None, matched_disparity=None)
    assert len(out) == 1
    assert not bool(out.iloc[0]["bucket_model_disagreement"])
    assert bool(out.iloc[0]["bucket_low_margin"])
