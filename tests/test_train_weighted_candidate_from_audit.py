from __future__ import annotations

import pandas as pd

from src.quantify_classify import FROZEN_ALPHA_BASELINE_FEATURES
from src.train_weighted_candidate_from_audit import train_weighted_candidates


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
        "fiber_type_source": "model_prediction",
        "needs_review": False,
        "model_confidence": 0.9,
        "model_margin": 0.8,
    }
    for col in FROZEN_ALPHA_BASELINE_FEATURES:
        row[col] = 0.0
    row["area"] = 100.0
    row["type1_mean"] = type1_mean
    row["type2_mean"] = type2_mean
    row["type1_p75"] = type1_mean
    row["type2_p75"] = type2_mean
    row["type1_p90"] = type1_mean
    row["type2_p90"] = type2_mean
    row["type1_pctl"] = type1_mean
    row["type2_pctl"] = type2_mean
    row["type1_coverage"] = min(type1_mean / 10.0, 1.0)
    row["type2_coverage"] = min(type2_mean / 10.0, 1.0)
    row["type_ratio"] = type1_mean / max(type2_mean, 1e-6)
    row["type_diff"] = type1_mean - type2_mean
    row["type_pctl_ratio"] = row["type_ratio"]
    row["type_pctl_diff"] = row["type_diff"]
    row["type_p75_ratio"] = row["type_ratio"]
    row["type_p75_diff"] = row["type_diff"]
    row["type_p90_ratio"] = row["type_ratio"]
    row["type_p90_diff"] = row["type_diff"]
    row["type_cov_ratio"] = row["type1_coverage"] / max(row["type2_coverage"], 1e-6)
    row["type_cov_diff"] = row["type1_coverage"] - row["type2_coverage"]
    row["type_cov_sum"] = row["type1_coverage"] + row["type2_coverage"]
    row["type1_snr_mean"] = type1_mean * 2
    row["type2_snr_mean"] = type2_mean * 2
    return row


def test_train_weighted_candidates_uses_manual_train_and_manual_eval_holdout():
    feature_table = pd.DataFrame(
        [
            _row("img_t1", 1, "train", "iib", 8.0, 1.0),
            _row("img_t2", 2, "train", "iia", 1.0, 8.0),
            _row("img_t3", 3, "train", "iix", 1.2, 1.1),
            _row("img_d1", 4, "dev", "iia", 1.1, 7.5),
            _row("img_d2", 5, "heldout", "iix", 1.1, 1.0),
        ]
    )
    manual_split = pd.DataFrame(
        [
            {
                "image_id": "img_d1",
                "label": 4,
                "audit_final_label": "iib",
                "manual_supervision_split": "manual_train",
            },
            {
                "image_id": "img_d2",
                "label": 5,
                "audit_final_label": "iix",
                "manual_supervision_split": "manual_eval_holdout",
            },
        ]
    )

    best_name, results = train_weighted_candidates(
        feature_table,
        manual_split,
        manual_weight=6.0,
        model_families=("rf", "gb"),
        feature_sets_filter=("baseline", "expanded"),
    )

    assert best_name in {"baseline_rf", "expanded_rf", "baseline_gb", "expanded_gb"}
    assert results["manual_train_rows"] == 1
    assert results["eval_rows"] == 1
    assert "candidate_name" in results["predictions_df"].columns
    assert set(results["metrics_df"]["candidate_name"]) == {
        "baseline_rf",
        "expanded_rf",
        "baseline_gb",
        "expanded_gb",
    }
