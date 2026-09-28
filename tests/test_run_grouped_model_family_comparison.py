from __future__ import annotations

import pandas as pd
import pytest

from src.run_grouped_model_family_comparison import (
    build_shortlist_decision,
    paired_model_comparisons,
    prepare_development_table,
    run_comparison,
    validate_comparison_outputs,
)


def _contract() -> dict:
    return {
        "contract_id": "synthetic_model_family.v1",
        "status": "frozen",
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
            },
            "uncertainty": {"method": "paired_mouse_bootstrap", "repetitions": 50, "seed": 4},
        },
        "development_shortlist_rule": {
            "comparator": "logistic_f1",
            "all_required": {
                "median_macro_f1_delta_at_least": 0.005,
                "nonnegative_mouse_deltas_at_least": 2,
                "iia_recall_delta_at_least": -0.02,
                "iix_iib_confusion_delta_at_most": 0.01,
                "brier_delta_at_most": 0.01,
            },
            "result_interpretation": "Development shortlist only.",
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
    validate_comparison_outputs(predictions, metrics, table, _contract(), tmp_path)
    comparisons = paired_model_comparisons(metrics, _contract())
    decision = build_shortlist_decision(comparisons, _contract())

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
    assert len(comparisons) == 32
    assert set(decision["candidates"]) == {"random_forest_f1", "xgboost_f1"}
    assert decision["promotion_decision"] is False


def test_shortlist_decision_requires_every_predeclared_criterion():
    rows = []
    metric_deltas = {
        "macro_f1": (0.01, 3),
        "iia_recall": (-0.01, 2),
        "iix_iib_symmetric_confusion_rate": (0.005, 2),
        "multiclass_brier_score": (0.009, 2),
    }
    for metric, (delta, nonnegative) in metric_deltas.items():
        rows.append(
            {
                "candidate": "candidate_a",
                "metric": metric,
                "paired_median_delta": delta,
                "nonnegative_group_deltas": nonnegative,
            }
        )

    decision = build_shortlist_decision(pd.DataFrame(rows), _contract())

    assert decision["shortlisted_model_families"] == ["candidate_a"]
    assert decision["candidates"]["candidate_a"]["passes_all_required"] is True


def test_shortlist_decision_fails_candidate_with_brier_regression():
    rows = []
    for metric, delta in {
        "macro_f1": 0.01,
        "iia_recall": 0.0,
        "iix_iib_symmetric_confusion_rate": 0.0,
        "multiclass_brier_score": 0.02,
    }.items():
        rows.append(
            {
                "candidate": "candidate_a",
                "metric": metric,
                "paired_median_delta": delta,
                "nonnegative_group_deltas": 3,
            }
        )

    decision = build_shortlist_decision(pd.DataFrame(rows), _contract())

    assert decision["shortlisted_model_families"] == []
    brier = decision["candidates"]["candidate_a"]["criteria"]["brier_delta_at_most"]
    assert brier["passed"] is False
