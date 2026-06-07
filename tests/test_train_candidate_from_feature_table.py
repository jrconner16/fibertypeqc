from __future__ import annotations

import pandas as pd

from src.quantify_classify import FROZEN_ALPHA_BASELINE_FEATURES
from src.train_candidate_from_feature_table import train_and_compare


def _row(
    fiber_type: str,
    split: str,
    image_id: str,
    label: int,
    area: float,
    type1_mean: float,
    type2_mean: float,
) -> dict[str, object]:
    row = {
        "image_id": image_id,
        "label": label,
        "fiber_type": fiber_type,
        "split": split,
        "fiber_type_source": "direct_marker",
        "needs_review": False,
        "model_confidence": 0.9,
        "model_margin": 0.8,
        "type_cov_sum": type1_mean + type2_mean,
        "type1_snr_mean": type1_mean * 2,
        "type2_snr_mean": type2_mean * 2,
    }
    for col in FROZEN_ALPHA_BASELINE_FEATURES:
        row.setdefault(col, 0.0)
    row["area"] = area
    row["type1_mean"] = type1_mean
    row["type2_mean"] = type2_mean
    row["type1_p75"] = type1_mean
    row["type2_p75"] = type2_mean
    row["type1_p90"] = type1_mean
    row["type2_p90"] = type2_mean
    row["type1_pctl"] = type1_mean
    row["type2_pctl"] = type2_mean
    row["type1_coverage"] = min(type1_mean / 10, 1.0)
    row["type2_coverage"] = min(type2_mean / 10, 1.0)
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
    return row


def test_train_and_compare_returns_baseline_and_expanded_candidates():
    rows = [
        _row("iib", "train", "img_train_1", 1, 100, 8.0, 1.0),
        _row("iib", "train", "img_train_2", 2, 110, 7.5, 1.2),
        _row("iia", "train", "img_train_3", 3, 90, 1.0, 8.0),
        _row("iia", "train", "img_train_4", 4, 95, 1.2, 7.2),
        _row("iix", "train", "img_train_5", 5, 85, 1.5, 1.4),
        _row("iix", "train", "img_train_6", 6, 80, 1.1, 1.0),
        _row("iib", "dev", "img_dev_1", 7, 105, 7.8, 1.1),
        _row("iia", "dev", "img_dev_2", 8, 92, 1.1, 7.9),
        _row("iix", "dev", "img_dev_3", 9, 82, 1.2, 1.3),
    ]
    df = pd.DataFrame(rows)

    best_name, results = train_and_compare(df)

    assert best_name in {"baseline_rf", "expanded_rf", "baseline_gb", "expanded_gb"}
    assert set(results["metrics_df"]["candidate_name"]) == {
        "baseline_rf",
        "expanded_rf",
        "baseline_gb",
        "expanded_gb",
    }
    assert results["train_rows"] == 6
    assert results["dev_rows"] == 3
    assert "candidate_name" in results["predictions_df"].columns
    assert "feature_set" in results["predictions_df"].columns
    assert "model_family" in results["predictions_df"].columns
    assert "predicted_fiber_type" in results["predictions_df"].columns
