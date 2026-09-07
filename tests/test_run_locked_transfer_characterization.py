from __future__ import annotations

import pandas as pd
import pytest

from src.run_locked_transfer_characterization import (
    characterize_feature_set,
    prepare_transfer_table,
)


def _contract() -> dict:
    return {
        "target": {"classes": ["iia", "iix", "iib"]},
        "feature_sets": {"F1": {"columns": ["type_iia.mean", "type_iib.mean"]}},
        "numeric_contract": {"epsilon": 1.0e-6},
        "fixed_model_recipe": {
            "family": "multinomial_logistic_regression",
            "parameters": {
                "C": 1.0,
                "solver": "lbfgs",
                "class_weight": "balanced",
                "max_iter": 500,
                "random_state": 7,
            },
        },
        "evaluation": {
            "descriptive_review_policy": {
                "low_confidence_below": 0.70,
                "low_margin_below": 0.25,
            }
        },
    }


def _table(*, authority: str, role: str) -> pd.DataFrame:
    rows = []
    patterns = {"iia": (9.0, 1.0), "iix": (1.0, 1.0), "iib": (1.0, 9.0)}
    for group_index, group_id in enumerate(["mouse_a", "mouse_b", "mouse_c"]):
        for label_id, (target, (iia, iib)) in enumerate(patterns.items(), start=1):
            rows.append(
                {
                    "image_id": f"image_{group_index}",
                    "group_id": group_id,
                    "pipeline_label_id": label_id,
                    "target_label": target,
                    "label_authority": authority,
                    "evidence_role": role,
                    "type_iia.mean": iia + group_index * 0.01,
                    "type_iib.mean": iib + group_index * 0.01,
                }
            )
    return pd.DataFrame(rows)


def test_prepare_transfer_table_rejects_non_development_evidence():
    table = _table(authority="myosight_derived", role="untouched_evaluation")

    with pytest.raises(ValueError, match="development_transfer"):
        prepare_transfer_table(table, ["iia", "iix", "iib"])


def test_characterize_locked_transfer_emits_image_and_overall_metrics():
    development = _table(authority="reviewed_myosight", role="development_model_selection")
    transfer = prepare_transfer_table(
        _table(authority="myosight_derived", role="development_transfer"),
        ["iia", "iix", "iib"],
    )

    predictions, metrics, _, columns = characterize_feature_set(
        development, transfer, _contract(), "F1"
    )

    assert len(predictions) == len(transfer)
    assert columns == ["type_iia.mean", "type_iib.mean"]
    assert set(metrics["scope_type"]) == {"image", "overall"}
    assert metrics.loc[metrics["scope_type"].eq("image"), "scope_id"].nunique() == 3
