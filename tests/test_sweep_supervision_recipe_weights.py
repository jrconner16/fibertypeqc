from __future__ import annotations

import pandas as pd

from src.quantify_classify import FROZEN_ALPHA_BASELINE_FEATURES
from src.sweep_supervision_recipe_weights import sweep_weights


def _row(
    image_id: str,
    label: int,
    split: str,
    fiber_type: str,
    type1_mean: float,
    type2_mean: float,
) -> dict[str, object]:
    row = {
        "image_id": image_id,
        "label": label,
        "split": split,
        "fiber_type": fiber_type,
        "needs_review": False,
        "model_confidence": 0.9,
        "model_margin": 0.8,
        "type1_mean": type1_mean,
        "type2_mean": type2_mean,
        "type1_coverage": max(0.05, min(0.95, type1_mean / 10)),
        "type2_coverage": max(0.05, min(0.95, type2_mean / 10)),
        "type1_cov_x_snr": type1_mean,
        "type1_snr_mean": type1_mean,
        "type2_cov_x_snr": type2_mean,
        "type2_snr_mean": type2_mean,
    }
    for feature in FROZEN_ALPHA_BASELINE_FEATURES:
        row.setdefault(feature, 0.0)
    return row


def test_sweep_weights_includes_reference_and_candidates() -> None:
    feature_table = pd.DataFrame(
        [
            _row("train1", 1, "train", "iib", 8.0, 1.0),
            _row("train2", 2, "train", "iix", 1.5, 1.2),
            _row("train3", 3, "train", "iia", 0.3, 8.5),
            _row("train4", 4, "train", "iix", 2.5, 1.4),
            _row("eval1", 5, "heldout", "iia", 0.4, 8.0),
            _row("eval2", 6, "heldout", "iix", 2.2, 1.5),
        ]
    )
    manual_split = pd.DataFrame(
        [
            {
                "image_id": "train1",
                "label": 1,
                "audit_final_label": "iib",
                "manual_supervision_split": "manual_train",
            },
            {
                "image_id": "train3",
                "label": 3,
                "audit_final_label": "iia",
                "manual_supervision_split": "manual_train",
            },
            {
                "image_id": "eval1",
                "label": 5,
                "audit_final_label": "iia",
                "manual_supervision_split": "manual_eval_holdout",
            },
            {
                "image_id": "eval2",
                "label": 6,
                "audit_final_label": "iix",
                "manual_supervision_split": "manual_eval_holdout",
            },
        ]
    )
    matched_myosight = pd.DataFrame(
        [
            {"image_id": "train1", "label": 1, "myosight_label": "iib", "split": "train"},
            {"image_id": "train2", "label": 2, "myosight_label": "iix", "split": "train"},
        ]
    )
    true_iia_reviewed = pd.DataFrame(
        [
            {
                "image_id": "hunt1",
                "label": 10,
                "type1_mean": 0.4,
                "type2_mean": 8.4,
                "type1_coverage": 0.08,
                "type2_coverage": 0.9,
                "type1_cov_x_snr": 0.4,
                "type1_snr_mean": 0.4,
                "type2_cov_x_snr": 8.2,
                "type2_snr_mean": 8.2,
                "audit_corrected_type": "iia",
            },
            {
                "image_id": "hunt2",
                "label": 11,
                "type1_mean": 0.5,
                "type2_mean": 8.6,
                "type1_coverage": 0.09,
                "type2_coverage": 0.92,
                "type1_cov_x_snr": 0.5,
                "type1_snr_mean": 0.5,
                "type2_cov_x_snr": 8.4,
                "type2_snr_mean": 8.4,
                "audit_corrected_type": "iia",
            },
        ]
    )
    reference = pd.DataFrame(
        [
            {
                "strategy": "baseline_gb_gated_iia_q0.01",
                "accuracy": 0.8,
                "balanced_accuracy": 0.84,
                "iia_precision": 1.0,
                "iia_recall": 0.97,
            }
        ]
    )

    summary = sweep_weights(
        feature_table,
        manual_split,
        matched_myosight,
        true_iia_reviewed,
        reference,
        manual_weight=8.0,
        myosight_weights=[2.0],
        baseline_weights=[0.5],
    )

    assert "reference_baseline_gb_gated_iia_q0.01" in set(summary["recipe"])
    assert "manual_high_myo_medium_baseline_light_soft_iia_q0.01" in set(summary["recipe"])
