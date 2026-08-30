from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from src.run_grouped_feature_ablation import (
    expected_calibration_error,
    feature_columns,
    numeric_feature_table,
    prepare_development_table,
    run_ablation,
    run_feature_set,
)


def _contract() -> dict:
    return {
        "target": {"classes": ["iia", "iix", "iib"]},
        "evidence_roles": {
            "primary_development": {
                "label_authority": "reviewed_myosight",
                "groups": 3,
                "images": 3,
                "eligible_three_class_rows": 9,
            }
        },
        "feature_sets": {
            "F0": {"columns": ["type_iib.mean", "type_iia.mean"]},
            "F1": {"includes": ["F0"], "columns": ["type_iia.p90", "type_iib.p90"]},
        },
        "fixed_model_recipe": {
            "family": "multinomial_logistic_regression",
            "parameters": {
                "C": 1.0,
                "solver": "lbfgs",
                "class_weight": "balanced",
                "max_iter": 5000,
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


def _table() -> pd.DataFrame:
    rows = []
    patterns = {
        "iia": (1.0, 9.0),
        "iix": (1.5, 1.5),
        "iib": (9.0, 1.0),
    }
    for group_index, group_id in enumerate(["mouse_a", "mouse_b", "mouse_c"]):
        for label_index, (target, (iib, iia)) in enumerate(patterns.items(), start=1):
            rows.append(
                {
                    "image_id": f"image_{group_index}",
                    "group_id": group_id,
                    "pipeline_label_id": label_index,
                    "target_label": target,
                    "development_cv_fold": f"fold_{group_index}",
                    "label_authority": "reviewed_myosight",
                    "supervision_role": "development_model_selection",
                    "eligible_for_model_fitting": True,
                    "type_iib.mean": iib + group_index * 0.1,
                    "type_iia.mean": iia + group_index * 0.1,
                    "type_iib.p90": iib + 1.0,
                    "type_iia.p90": iia + 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_feature_columns_resolves_includes_in_stable_order():
    assert feature_columns(_contract(), "F1") == [
        "type_iib.mean",
        "type_iia.mean",
        "type_iia.p90",
        "type_iib.p90",
    ]


def test_prepare_development_table_rejects_group_in_multiple_folds():
    table = _table()
    table.loc[1, "development_cv_fold"] = "different_fold"

    with pytest.raises(ValueError, match="more than one development fold"):
        prepare_development_table(table, _contract())


def test_numeric_feature_table_rejects_nonfinite_values():
    table = _table()
    table.loc[0, "type_iia.mean"] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        numeric_feature_table(table, ["type_iia.mean"])


def test_run_feature_set_requires_every_class_in_each_heldout_group(tmp_path):
    table = _table()
    table.loc[
        (table["group_id"] == "mouse_a") & (table["target_label"] == "iia"), "target_label"
    ] = "iix"

    with pytest.raises(ValueError, match="held-out rows lack classes"):
        run_feature_set(table, _contract(), "F0", tmp_path)


def test_expected_calibration_error_is_zero_for_correct_certain_predictions():
    actual = np.array([0, 1, 2])
    probabilities = np.eye(3)

    assert expected_calibration_error(actual, probabilities) == pytest.approx(0.0)


def test_run_ablation_emits_one_prediction_per_row_and_fold_models(tmp_path):
    output = tmp_path / "experiment"

    predictions, metrics = run_ablation(_table(), _contract(), ["F0", "F1"], output)

    assert len(predictions) == 18
    assert len(metrics) == 6
    assert predictions.groupby("feature_set").size().to_dict() == {"F0": 9, "F1": 9}
    assert metrics.groupby("feature_set").size().to_dict() == {"F0": 3, "F1": 3}
    assert predictions.groupby(["feature_set", "group_id"]).size().eq(3).all()
    assert len(list((output / "models" / "F0").glob("*.joblib"))) == 3
    assert len(list((output / "models" / "F1").glob("*.joblib"))) == 3
    assert (output / "out_of_fold_predictions.csv").is_file()
    assert (output / "per_mouse_metrics.csv").is_file()
    assert (output / "median_mouse_metrics.csv").is_file()

    first_fold = joblib.load(output / "models" / "F0" / "fold_00.joblib")
    training = _table().loc[lambda frame: frame["group_id"] != "mouse_a"]
    expected_mean = training[["type_iib.mean", "type_iia.mean"]].mean().to_numpy()
    np.testing.assert_allclose(first_fold["pipeline"].named_steps["scale"].mean_, expected_mean)


def test_run_ablation_refuses_to_overwrite_output(tmp_path):
    output = tmp_path / "experiment"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        run_ablation(_table(), _contract(), ["F0"], output)
