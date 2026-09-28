from __future__ import annotations

import pandas as pd
import pytest

from src.run_p1_iia_gate_ablation import evaluate


def _features() -> pd.DataFrame:
    rows = []
    class_values = {
        "iia": (9.0, 1.0, 8.0, 1.0, 0.9, 0.1),
        "iix": (1.0, 1.0, 1.0, 1.0, 0.1, 0.1),
        "iib": (1.0, 9.0, 1.0, 8.0, 0.1, 0.9),
    }
    for group_index, group_id in enumerate(["mouse_a", "mouse_b", "mouse_c"]):
        for label_id, (target, values) in enumerate(class_values.items(), start=1):
            rows.append(
                {
                    "image_id": f"image_{group_index}",
                    "group_id": group_id,
                    "pipeline_label_id": label_id,
                    "target_label": target,
                    "type_iia.mean": values[0],
                    "type_iib.mean": values[1],
                    "type_iia.snr_mean": values[2],
                    "type_iib.snr_mean": values[3],
                    "type_iia.coverage_high": values[4],
                    "type_iib.coverage_high": values[5],
                }
            )
    return pd.DataFrame(rows)


def _predictions(features: pd.DataFrame) -> pd.DataFrame:
    result = features.loc[:, ["image_id", "group_id", "pipeline_label_id", "target_label"]].copy()
    result.insert(0, "condition", "notch3_only")
    result.insert(1, "evaluation_scope", "grouped_resampling_notch3")
    result["predicted_label"] = result["target_label"]
    result["model_confidence"] = 0.9
    result["model_margin"] = 0.8
    return result


def test_iia_gate_uses_one_training_threshold_set_per_heldout_group():
    features = _features()

    predictions, metrics, thresholds = evaluate(features, _predictions(features), 0.01)

    assert len(predictions) == len(features)
    assert thresholds["heldout_group_id"].nunique() == 3
    assert set(metrics["strategy"]) == {"raw_logistic", "soft_iia_gate_q0.01"}
    assert set(metrics["scope_type"]) == {"overall", "group"}


def test_iia_gate_rejects_mismatched_oof_rows():
    features = _features()
    predictions = _predictions(features).iloc[:-1]

    with pytest.raises(ValueError, match="row count"):
        evaluate(features, predictions, 0.01)
