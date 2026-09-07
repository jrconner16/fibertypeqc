from __future__ import annotations

import pandas as pd
import pytest

from src.run_grouped_model_family_comparison import (
    prepare_development_table,
    run_comparison,
)


def _contract() -> dict:
    return {
        "evidence_boundary": {
            "development": {"groups": 3, "images": 3, "eligible_three_class_rows": 9}
        },
        "target": {"classes": ["iia", "iix", "iib"]},
        "feature_set": {"columns": ["type_iia.mean", "type_iib.mean"]},
        "candidate_models": {
            "logistic_f1": {
                "family": "multinomial_logistic_regression",
                "parameters": {
                    "C": 1.0,
                    "solver": "lbfgs",
                    "class_weight": "balanced",
                    "max_iter": 500,
                    "random_state": 11,
                },
            },
            "random_forest_f1": {
                "family": "random_forest",
                "parameters": {
                    "n_estimators": 3,
                    "min_samples_leaf": 1,
                    "class_weight": "balanced",
                    "n_jobs": 1,
                    "random_state": 12,
                },
            },
            "xgboost_f1": {
                "family": "xgboost_multiclass",
                "parameters": {
                    "n_estimators": 3,
                    "max_depth": 2,
                    "learning_rate": 0.1,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "min_child_weight": 1,
                    "reg_lambda": 1.0,
                    "tree_method": "hist",
                    "n_jobs": 1,
                    "random_state": 13,
                    "eval_metric": "mlogloss",
                },
            },
        },
        "evaluation": {
            "descriptive_review_policy": {
                "low_confidence_below": 0.70,
                "low_margin_below": 0.25,
            }
        },
    }


def _table() -> pd.DataFrame:
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
                    "label_authority": "reviewed_myosight",
                    "supervision_role": "development_model_selection",
                    "eligible_for_model_fitting": True,
                    "type_iia.mean": iia + group_index * 0.01,
                    "type_iib.mean": iib + group_index * 0.01,
                }
            )
    return pd.DataFrame(rows)


def test_prepare_model_family_table_rejects_non_development_rows():
    table = _table()
    table.loc[0, "supervision_role"] = "protected_final_evaluation"

    with pytest.raises(ValueError, match="development_model_selection"):
        prepare_development_table(table, _contract())


def test_run_model_family_comparison_emits_grouped_outputs(tmp_path):
    table = prepare_development_table(_table(), _contract())

    predictions, metrics = run_comparison(table, _contract(), tmp_path)

    assert len(predictions) == len(table) * 3
    assert len(metrics) == 9
    assert set(predictions["model_family"]) == {
        "logistic_f1",
        "random_forest_f1",
        "xgboost_f1",
    }
    assert predictions.groupby("model_family").size().eq(len(table)).all()
    assert predictions.groupby(["model_family", "group_id"]).size().eq(3).all()
    assert len(list((tmp_path / "models").glob("*/*.joblib"))) == 9
