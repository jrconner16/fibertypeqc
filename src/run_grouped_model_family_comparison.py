"""Run predeclared classifier families on frozen F1 features with grouped Notch3 CV."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.run_grouped_feature_ablation import _fold_metrics, numeric_feature_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_contract(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Model-family contract must be a YAML mapping.")
    return value


def prepare_development_table(table: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    required = {
        "image_id",
        "group_id",
        "pipeline_label_id",
        "target_label",
        "label_authority",
        "supervision_role",
        "eligible_for_model_fitting",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Feature table missing required columns: {missing}")
    if table.duplicated(["image_id", "pipeline_label_id"]).any():
        raise ValueError("Feature table has duplicate image/pipeline-label keys.")
    if not table["label_authority"].astype(str).eq("reviewed_myosight").all():
        raise ValueError("Development labels must be uniformly reviewed_myosight.")
    if not table["supervision_role"].astype(str).eq("development_model_selection").all():
        raise ValueError("Only development_model_selection rows may be read.")
    if not table["eligible_for_model_fitting"].astype(str).str.lower().eq("true").all():
        raise ValueError("All rows must be eligible development fitting rows.")
    classes = [str(value) for value in contract["target"]["classes"]]
    out = table.copy()
    out["target_label"] = out["target_label"].astype(str).str.strip().str.lower()
    if set(out["target_label"]).difference(classes):
        raise ValueError("Development table contains labels outside the frozen target.")
    expected = contract["evidence_boundary"]["development"]
    if out["group_id"].nunique() != int(expected["groups"]):
        raise ValueError("Development table has the wrong number of groups.")
    if out["image_id"].nunique() != int(expected["images"]):
        raise ValueError("Development table has the wrong number of images.")
    if len(out) != int(expected["eligible_three_class_rows"]):
        raise ValueError("Development table has the wrong number of rows.")
    return out.reset_index(drop=True)


def build_model(name: str, contract: dict[str, Any]) -> object:
    recipe = contract["candidate_models"][name]
    params = recipe["parameters"]
    family = recipe["family"]
    if family == "multinomial_logistic_regression":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(params["C"]),
                        solver=str(params["solver"]),
                        class_weight=str(params["class_weight"]),
                        max_iter=int(params["max_iter"]),
                        random_state=int(params["random_state"]),
                    ),
                ),
            ]
        )
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            class_weight=str(params["class_weight"]),
            n_jobs=int(params["n_jobs"]),
            random_state=int(params["random_state"]),
        )
    if family == "xgboost_multiclass":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            min_child_weight=float(params["min_child_weight"]),
            reg_lambda=float(params["reg_lambda"]),
            tree_method=str(params["tree_method"]),
            n_jobs=int(params["n_jobs"]),
            random_state=int(params["random_state"]),
            eval_metric=str(params["eval_metric"]),
        )
    raise ValueError(f"Unsupported model family: {family}")


def _probabilities_in_class_order(
    model: object, x: pd.DataFrame, classes: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(model.predict_proba(x), dtype=np.float64)
    model_classes = getattr(model, "classes_", None)
    if model_classes is None and isinstance(model, Pipeline):
        model_classes = model.named_steps["model"].classes_
    model_classes = list(model_classes)
    if set(model_classes) == set(classes):
        order = [model_classes.index(name) for name in classes]
    elif set(model_classes) == set(range(len(classes))):
        order = list(range(len(classes)))
    else:
        raise ValueError(f"Unexpected fitted class labels: {model_classes}")
    probabilities = raw[:, order]
    return np.asarray(classes)[probabilities.argmax(axis=1)], probabilities


def run_comparison(
    table: pd.DataFrame, contract: dict[str, Any], output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [str(value) for value in contract["feature_set"]["columns"]]
    features = numeric_feature_table(table, columns)
    classes = [str(value) for value in contract["target"]["classes"]]
    groups = sorted(table["group_id"].astype(str).unique())
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for model_name in contract["candidate_models"]:
        model_dir = output_dir / "models" / str(model_name)
        model_dir.mkdir(parents=True)
        for fold_index, heldout_group in enumerate(groups):
            heldout = table["group_id"].astype(str).eq(heldout_group)
            train_y = table.loc[~heldout, "target_label"].astype(str)
            test_y = table.loc[heldout, "target_label"].astype(str)
            if set(train_y) != set(classes) or set(test_y) != set(classes):
                raise ValueError(f"Fold {heldout_group} does not contain all target classes.")
            model = build_model(str(model_name), contract)
            if contract["candidate_models"][model_name]["family"] == "xgboost_multiclass":
                class_index = {name: index for index, name in enumerate(classes)}
                model.fit(features.loc[~heldout], train_y.map(class_index))
            else:
                model.fit(features.loc[~heldout], train_y)
            predicted, probabilities = _probabilities_in_class_order(
                model, features.loc[heldout], classes
            )
            fold_predictions = table.loc[
                heldout, ["image_id", "group_id", "pipeline_label_id", "target_label"]
            ].copy()
            fold_predictions.insert(0, "model_family", model_name)
            fold_predictions["predicted_label"] = predicted
            for class_index, class_name in enumerate(classes):
                fold_predictions[f"prob_{class_name}"] = probabilities[:, class_index]
            fold_predictions["model_confidence"] = probabilities.max(axis=1)
            fold_predictions["model_margin"] = (
                np.sort(probabilities, axis=1)[:, -1] - np.sort(probabilities, axis=1)[:, -2]
            )
            prediction_frames.append(fold_predictions)
            metric_rows.append(
                {
                    "model_family": model_name,
                    "heldout_group_id": heldout_group,
                    "fold_index": fold_index,
                    "train_rows": int((~heldout).sum()),
                    "test_rows": int(heldout.sum()),
                    "feature_count": len(columns),
                    **_fold_metrics(test_y.to_numpy(), predicted, probabilities, classes, contract),
                }
            )
            joblib.dump(
                {
                    "model_family": model_name,
                    "feature_columns": columns,
                    "heldout_group_id": heldout_group,
                    "pipeline": model,
                },
                model_dir / f"fold_{fold_index:02d}.joblib",
            )
            print(
                f"completed {model_name} fold {fold_index + 1}/{len(groups)}: {heldout_group}",
                flush=True,
            )
    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(metric_rows)


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    contract = _load_contract(args.contract)
    table = prepare_development_table(pd.read_csv(args.feature_table, low_memory=False), contract)
    args.output_dir.mkdir(parents=True)
    predictions, metrics = run_comparison(table, contract, args.output_dir)
    predictions.to_csv(args.output_dir / "out_of_fold_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "per_mouse_metrics.csv", index=False)
    metrics.groupby("model_family").median(numeric_only=True).reset_index().to_csv(
        args.output_dir / "median_mouse_metrics.csv", index=False
    )
    shutil.copy2(args.contract, args.output_dir / "resolved_contract.yaml")
    (args.output_dir / "run_provenance.json").write_text(
        json.dumps(
            {
                "contract_sha256": _sha256(args.contract),
                "feature_table_sha256": _sha256(args.feature_table),
                "development_rows": len(table),
                "protected_rows_read": 0,
                "original_jag_rows_read": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
