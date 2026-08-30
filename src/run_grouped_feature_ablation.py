from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fibertypeqc.experimental_features import (
    RELATIVE_CHANNEL_FEATURES,
    add_relative_channel_features,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-recipe leave-one-group-out feature ablations and save fold-level evidence."
        )
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--diagnostics-inventory", type=Path, required=True)
    parser.add_argument("--feature-set", action="append", dest="feature_sets")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Contract must be a YAML mapping: {path}")
    return raw


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state(repo_root: Path) -> tuple[str, bool]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        text=True,
    )
    return revision, bool(status.strip())


def feature_columns(contract: dict[str, Any], feature_set: str) -> list[str]:
    definitions = contract.get("feature_sets")
    if not isinstance(definitions, dict) or feature_set not in definitions:
        raise ValueError(f"Unknown feature set: {feature_set}")

    resolved: list[str] = []
    visiting: set[str] = set()

    def add(name: str) -> None:
        if name in visiting:
            raise ValueError(f"Feature-set inclusion cycle involving {name}")
        entry = definitions.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid feature-set definition: {name}")
        visiting.add(name)
        for included in entry.get("includes", []):
            add(str(included))
        resolved.extend(str(column) for column in entry.get("columns", []))
        for columns in entry.get("families", {}).values():
            resolved.extend(str(column) for column in columns)
        per_marker = entry.get("per_marker_columns", {})
        for marker in per_marker.get("markers", []):
            for suffix in per_marker.get("suffixes", []):
                resolved.append(f"{marker}.{suffix}")
        visiting.remove(name)

    add(feature_set)
    return list(dict.fromkeys(resolved))


def prepare_development_table(table: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    required = {
        "image_id",
        "group_id",
        "pipeline_label_id",
        "target_label",
        "development_cv_fold",
        "label_authority",
        "supervision_role",
        "eligible_for_model_fitting",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Feature table missing required columns: {missing}")
    if table.duplicated(["image_id", "pipeline_label_id"]).any():
        raise ValueError("Feature table has duplicate image/pipeline-label keys.")
    if not table["supervision_role"].astype(str).eq("development_model_selection").all():
        raise ValueError("Feature table contains rows outside development_model_selection.")
    eligible = table["eligible_for_model_fitting"].astype(str).str.lower()
    if not eligible.eq("true").all():
        raise ValueError("Feature table contains rows ineligible for model fitting.")

    expected_authority = str(contract["evidence_roles"]["primary_development"]["label_authority"])
    if not table["label_authority"].astype(str).eq(expected_authority).all():
        raise ValueError(f"Feature table label authority is not uniformly {expected_authority}.")

    out = table.copy()
    out["target_label"] = out["target_label"].astype(str).str.strip().str.lower()
    classes = [str(value) for value in contract["target"]["classes"]]
    out = out.loc[out["target_label"].isin(classes)].copy()
    if out.empty:
        raise ValueError("No rows remain after applying the target-class contract.")

    group_to_fold = out.loc[:, ["group_id", "development_cv_fold"]].drop_duplicates()
    if group_to_fold["group_id"].duplicated().any():
        raise ValueError("A group maps to more than one development fold.")
    if group_to_fold["development_cv_fold"].duplicated().any():
        raise ValueError("A development fold contains more than one group.")

    expected = contract["evidence_roles"]["primary_development"]
    if out["group_id"].nunique() != int(expected["groups"]):
        raise ValueError(
            f"Expected {expected['groups']} development groups; found {out['group_id'].nunique()}."
        )
    if out["image_id"].nunique() != int(expected["images"]):
        raise ValueError(
            f"Expected {expected['images']} development images; found {out['image_id'].nunique()}."
        )
    if len(out) != int(expected["eligible_three_class_rows"]):
        raise ValueError(
            f"Expected {expected['eligible_three_class_rows']} eligible rows; found {len(out)}."
        )
    return out.reset_index(drop=True)


def numeric_feature_table(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = sorted(set(columns).difference(table.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    out = table.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    nonfinite = ~np.isfinite(out.to_numpy(dtype=np.float64))
    if nonfinite.any():
        bad_columns = out.columns[np.any(nonfinite, axis=0)].tolist()
        raise ValueError(f"Required features contain non-finite values: {bad_columns}")
    return out


def expected_calibration_error(
    actual: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == actual
    bin_ids = np.digitize(confidence, np.linspace(0.0, 1.0, bins + 1)[1:-1], right=True)
    error = 0.0
    for bin_id in range(bins):
        selected = bin_ids == bin_id
        if selected.any():
            error += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return error


def _build_model(contract: dict[str, Any]) -> Pipeline:
    recipe = contract["fixed_model_recipe"]
    if recipe["family"] != "multinomial_logistic_regression":
        raise ValueError(f"Unsupported fixed model family: {recipe['family']}")
    parameters = recipe["parameters"]
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(parameters["C"]),
                    solver=str(parameters["solver"]),
                    class_weight=str(parameters["class_weight"]),
                    max_iter=int(parameters["max_iter"]),
                    random_state=int(parameters["random_state"]),
                ),
            ),
        ]
    )


def _fold_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
    contract: dict[str, Any],
) -> dict[str, float]:
    precision, recall, class_f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=np.asarray(classes),
        zero_division=0,
    )
    class_index = {name: index for index, name in enumerate(classes)}
    encoded = np.asarray([class_index[value] for value in actual], dtype=int)
    one_hot = np.eye(len(classes), dtype=np.float64)[encoded]
    confidence = probabilities.max(axis=1)
    sorted_probabilities = np.sort(probabilities, axis=1)
    margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    review = contract["evaluation"]["descriptive_review_policy"]
    iix_iib = np.isin(actual, ["iix", "iib"])
    iix_iib_error = iix_iib & (
        ((actual == "iix") & (predicted == "iib")) | ((actual == "iib") & (predicted == "iix"))
    )
    metrics: dict[str, float] = {
        "macro_f1": float(f1_score(actual, predicted, labels=classes, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "multiclass_log_loss": float(
            -np.log(np.clip(probabilities[np.arange(len(encoded)), encoded], 1.0e-15, 1.0)).mean()
        ),
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "expected_calibration_error": expected_calibration_error(encoded, probabilities),
        "iix_iib_symmetric_confusion_rate": (
            float(iix_iib_error.sum() / iix_iib.sum()) if iix_iib.any() else float("nan")
        ),
        "review_rate": float(
            (
                (confidence < float(review["low_confidence_below"]))
                | (margin < float(review["low_margin_below"]))
            ).mean()
        ),
    }
    for index, name in enumerate(classes):
        metrics[f"{name}_precision"] = float(precision[index])
        metrics[f"{name}_recall"] = float(recall[index])
        metrics[f"{name}_f1"] = float(class_f1[index])
    return metrics


def run_feature_set(
    table: pd.DataFrame,
    contract: dict[str, Any],
    feature_set: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = feature_columns(contract, feature_set)
    features = numeric_feature_table(table, columns)
    classes = [str(value) for value in contract["target"]["classes"]]
    groups = sorted(table["group_id"].astype(str).unique())
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    model_dir = output_dir / "models" / feature_set
    model_dir.mkdir(parents=True, exist_ok=True)

    for fold_index, heldout_group in enumerate(groups):
        heldout = table["group_id"].astype(str) == heldout_group
        train_y = table.loc[~heldout, "target_label"].astype(str)
        test_y = table.loc[heldout, "target_label"].astype(str)
        missing_train_classes = sorted(set(classes).difference(train_y.unique()))
        if missing_train_classes:
            raise ValueError(
                f"Fold {heldout_group} training rows lack classes: {missing_train_classes}"
            )
        missing_test_classes = sorted(set(classes).difference(test_y.unique()))
        if missing_test_classes:
            raise ValueError(
                f"Fold {heldout_group} held-out rows lack classes: {missing_test_classes}"
            )
        fold_id = str(table.loc[heldout, "development_cv_fold"].iloc[0])
        model = _build_model(contract)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(features.loc[~heldout], train_y)

        raw_probabilities = model.predict_proba(features.loc[heldout])
        model_classes = list(model.named_steps["model"].classes_)
        order = [model_classes.index(name) for name in classes]
        probabilities = raw_probabilities[:, order]
        predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
        actual = table.loc[heldout, "target_label"].astype(str).to_numpy()
        confidence = probabilities.max(axis=1)
        sorted_probabilities = np.sort(probabilities, axis=1)
        margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]

        fold_predictions = table.loc[
            heldout,
            [
                "image_id",
                "group_id",
                "development_cv_fold",
                "pipeline_label_id",
                "target_label",
            ],
        ].copy()
        fold_predictions.insert(0, "feature_set", feature_set)
        fold_predictions["predicted_label"] = predicted
        for class_index, name in enumerate(classes):
            fold_predictions[f"prob_{name}"] = probabilities[:, class_index]
        fold_predictions["model_confidence"] = confidence
        fold_predictions["model_margin"] = margin
        prediction_frames.append(fold_predictions)

        metric_rows.append(
            {
                "feature_set": feature_set,
                "fold_index": fold_index,
                "development_cv_fold": fold_id,
                "heldout_group_id": heldout_group,
                "train_rows": int((~heldout).sum()),
                "test_rows": int(heldout.sum()),
                "feature_count": len(columns),
                **_fold_metrics(actual, predicted, probabilities, classes, contract),
            }
        )
        joblib.dump(
            {
                "feature_set": feature_set,
                "feature_columns": columns,
                "classes": classes,
                "heldout_group_id": heldout_group,
                "pipeline": model,
            },
            model_dir / f"fold_{fold_index:02d}.joblib",
        )

    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(metric_rows)


def run_ablation(
    table: pd.DataFrame,
    contract: dict[str, Any],
    feature_sets: list[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    development = prepare_development_table(table, contract)
    requested_columns = {
        column for feature_set in feature_sets for column in feature_columns(contract, feature_set)
    }
    if requested_columns.intersection(RELATIVE_CHANNEL_FEATURES):
        development = add_relative_channel_features(
            development,
            epsilon=float(contract["numeric_contract"]["epsilon"]),
        )
    output_dir.mkdir(parents=True)
    predictions: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    for feature_set in feature_sets:
        feature_predictions, feature_metrics = run_feature_set(
            development, contract, feature_set, output_dir
        )
        predictions.append(feature_predictions)
        metrics.append(feature_metrics)
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_metrics = pd.concat(metrics, ignore_index=True)
    all_predictions.to_csv(output_dir / "out_of_fold_predictions.csv", index=False)
    all_metrics.to_csv(output_dir / "per_mouse_metrics.csv", index=False)
    metric_columns = all_metrics.select_dtypes(include="number").columns.difference(
        ["fold_index", "train_rows", "test_rows", "feature_count"]
    )
    summary = all_metrics.groupby("feature_set")[list(metric_columns)].median().reset_index()
    summary.to_csv(output_dir / "median_mouse_metrics.csv", index=False)
    return all_predictions, all_metrics


def main() -> None:
    args = build_parser().parse_args()
    contract = _load_yaml(args.contract)
    selected = args.feature_sets or ["F0", "F1"]
    table = pd.read_csv(args.feature_table, low_memory=False)
    predictions, metrics = run_ablation(table, contract, selected, args.output_dir)
    shutil.copy2(args.contract, args.output_dir / "resolved_contract.yaml")
    repo_root = Path(__file__).resolve().parents[1]
    revision, dirty = _git_state(repo_root)
    lockfile = repo_root / "uv.lock"
    provenance = {
        "contract_id": contract["contract_id"],
        "contract_sha256": _sha256(args.contract),
        "feature_table_sha256": _sha256(args.feature_table),
        "diagnostics_inventory_sha256": _sha256(args.diagnostics_inventory),
        "git_revision": revision,
        "git_tracked_files_dirty": dirty,
        "uv_lock_sha256": _sha256(lockfile),
        "feature_sets": selected,
        "rows": int(len(table)),
        "out_of_fold_prediction_rows": int(len(predictions)),
        "fold_metric_rows": int(len(metrics)),
    }
    (args.output_dir / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"feature sets: {', '.join(selected)}")
    print(f"out-of-fold predictions: {len(predictions)}")
    print(f"per-mouse metric rows: {len(metrics)}")
    print(f"saved experiment: {args.output_dir}")


if __name__ == "__main__":
    main()
