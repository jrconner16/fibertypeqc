"""Development-only panel-specific baseline training and held-out evaluation.

This does not modify pipeline inference or select a production model. It trains
predeclared baselines from manual-gold labels and evaluates each once on a
separate whole-image holdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

BASELINE_VERSION = "panel_baselines.v1"
SEMANTIC_SUFFIXES = ("mean", "p90", "coverage_high", "snr_mean", "snr_p90")
SEMANTIC_MARKERS = ("type_i", "type_iia", "type_iib", "type_iix")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train development-only panel baselines and evaluate a frozen whole-image holdout."
        )
    )
    parser.add_argument("--panel-id", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--train-reviewed-fibers", type=Path, required=True)
    parser.add_argument("--train-diagnostics", type=Path, required=True)
    parser.add_argument("--heldout-reviewed-fibers", type=Path, required=True)
    parser.add_argument("--heldout-diagnostics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _manual_table(
    reviewed: pd.DataFrame, diagnostics: pd.DataFrame, classes: tuple[str, ...]
) -> pd.DataFrame:
    _require_columns(reviewed, ("label", "final_type", "label_source"), "reviewed fibers")
    _require_columns(diagnostics, ("label",), "diagnostics")
    labels = reviewed.loc[
        reviewed["label_source"].fillna("").eq("manual_gold"), ["label", "final_type"]
    ].copy()
    labels["final_type"] = labels["final_type"].fillna("").astype(str).str.strip().str.lower()
    labels = labels.loc[labels["final_type"].isin(classes)].copy()
    if labels.empty:
        raise ValueError("No eligible manual-gold labels found.")
    if labels["label"].duplicated().any():
        raise ValueError("Manual labels contain duplicate fiber labels.")
    table = labels.merge(diagnostics, on="label", how="inner", validate="one_to_one")
    if len(table) != len(labels):
        raise ValueError("Some manual labels are missing diagnostics.")
    return table.reset_index(drop=True)


def semantic_feature_columns(train: pd.DataFrame, heldout: pd.DataFrame) -> list[str]:
    requested = [
        f"{marker}.{suffix}" for marker in SEMANTIC_MARKERS for suffix in SEMANTIC_SUFFIXES
    ]
    columns = [
        column for column in requested if column in train.columns and column in heldout.columns
    ]
    if not columns:
        raise ValueError(
            "No shared semantic marker features are available between train and holdout."
        )
    return columns


def _numeric_features(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = table.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    if out.isna().any().any():
        missing = out.columns[out.isna().any()].tolist()
        raise ValueError(f"Feature table has missing values in: {', '.join(missing)}")
    return out


def baseline_models() -> dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=0,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=7,
            l2_regularization=1.0,
            random_state=0,
        ),
    }


def _fit(model: object, x: pd.DataFrame, y: pd.Series) -> object:
    sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    if isinstance(model, Pipeline):
        model.fit(x, y, model__sample_weight=sample_weight)
    else:
        model.fit(x, y, sample_weight=sample_weight)
    return model


def _metrics(y_true: pd.Series, y_pred: np.ndarray, classes: tuple[str, ...]) -> dict[str, object]:
    return {
        "n_labels": int(len(y_true)),
        "class_counts": {name: int((y_true == name).sum()) for name in classes},
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": classification_report(
            y_true, y_pred, labels=list(classes), output_dict=True, zero_division=0
        ),
    }


def cross_validated_predictions(
    model: object, x: pd.DataFrame, y: pd.Series, classes: tuple[str, ...]
) -> tuple[np.ndarray, dict[str, object]]:
    folds = min(5, int(y.value_counts().min()))
    if folds < 2:
        raise ValueError("Each training class needs at least two labels for development CV.")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    prediction = np.empty(len(y), dtype=object)
    for train_idx, test_idx in splitter.split(x, y):
        fitted = _fit(clone(model), x.iloc[train_idx], y.iloc[train_idx])
        prediction[test_idx] = fitted.predict(x.iloc[test_idx])
    return prediction.astype(str), _metrics(y, prediction.astype(str), classes)


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output directory already exists: {args.output_dir}")
    classes = tuple(str(value).strip().lower() for value in args.classes)
    if len(set(classes)) != len(classes):
        raise SystemExit("--classes contains duplicates.")

    train = _manual_table(
        pd.read_csv(args.train_reviewed_fibers, low_memory=False),
        pd.read_csv(args.train_diagnostics, low_memory=False),
        classes,
    )
    heldout = _manual_table(
        pd.read_csv(args.heldout_reviewed_fibers, low_memory=False),
        pd.read_csv(args.heldout_diagnostics, low_memory=False),
        classes,
    )
    features = semantic_feature_columns(train, heldout)
    train_x, heldout_x = _numeric_features(train, features), _numeric_features(heldout, features)
    train_y, heldout_y = train["final_type"], heldout["final_type"]

    args.output_dir.mkdir(parents=True)
    manifest = {
        "candidate_version": BASELINE_VERSION,
        "panel_id": args.panel_id,
        "classes": list(classes),
        "features": features,
        "training_evaluation": "fiber-level cross-validation within one development image",
        "heldout_evaluation": "one frozen whole-image holdout; do not select a winner from it",
        "train_label_counts": train_y.value_counts().to_dict(),
        "heldout_label_counts": heldout_y.value_counts().to_dict(),
    }
    (args.output_dir / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    train[["label", "final_type", *features]].to_csv(
        args.output_dir / "training_rows.csv", index=False
    )
    heldout[["label", "final_type", *features]].to_csv(
        args.output_dir / "heldout_rows.csv", index=False
    )

    report: dict[str, object] = {"manifest": manifest, "models": {}}
    for name, model in baseline_models().items():
        cv_pred, cv_metrics = cross_validated_predictions(model, train_x, train_y, classes)
        fitted = _fit(model, train_x, train_y)
        heldout_pred = fitted.predict(heldout_x).astype(str)
        joblib.dump(
            {
                "candidate_version": BASELINE_VERSION,
                "features": features,
                "classes": classes,
                "model": fitted,
            },
            args.output_dir / f"{name}.joblib",
        )
        pd.DataFrame(
            {"label": heldout["label"], "actual": heldout_y, "prediction": heldout_pred}
        ).to_csv(args.output_dir / f"{name}_heldout_predictions.csv", index=False)
        report["models"][name] = {
            "development_cv": cv_metrics,
            "heldout": _metrics(heldout_y, heldout_pred, classes),
        }
    (args.output_dir / "comparison_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Development-only candidates written to: {args.output_dir}")


if __name__ == "__main__":
    main()
