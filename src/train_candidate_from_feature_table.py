from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

from src.quantify_classify import FROZEN_ALPHA_BASELINE_FEATURES

NON_FEATURE_COLUMNS = {
    "image_id",
    "feature_diagnostics_path",
    "label",
    "fiber_type",
    "fiber_type_source",
    "classification_method",
    "available_markers",
    "needs_review",
    "model_confidence",
    "model_margin",
    "prob_iib",
    "prob_iia",
    "prob_iix",
    "split",
    "split_suggested",
    "candidate_included",
    "used_in_alpha_training",
    "has_myosight_comparison",
    "has_manual_labels",
    "input_kind",
    "validation_input_kind",
    "panel_type",
    "genotype",
    "timepoint",
    "source_image",
    "fiber_count_bin",
    "review_rate_bin",
    "signal_warning_bin",
    "myosight_count_diff_bin",
    "difficulty_score",
    "quality_bin_suggested",
    "saturation_proxy_suggested",
    "pipeline_total_fibers",
    "pipeline_needs_review_rate",
    "pipeline_signal_warning_rate",
    "pipeline_iib_pct",
    "pipeline_iia_pct",
    "pipeline_iix_pct",
    "fiber_count_abs_pct_diff",
    "quality_bin_manual",
    "saturation_bin_manual",
    "stain_quality_notes",
    "distribution_notes",
    "split_notes",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the first candidate model from a candidate feature table. "
            "Selection is performed on the dev split only."
        )
    )
    parser.add_argument(
        "--feature-table",
        type=Path,
        required=True,
        help="Candidate feature table CSV assembled from diagnostics outputs.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        required=True,
        help="Output .joblib path for the selected candidate model trained on the train split.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Optional CSV with dev-set comparison metrics.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
        help="Optional CSV with dev-set predictions from each candidate.",
    )
    parser.add_argument(
        "--report-txt",
        type=Path,
        default=None,
        help="Optional dev-set classification report for the selected candidate.",
    )
    return parser


def _load_feature_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {"fiber_type", "split", *FROZEN_ALPHA_BASELINE_FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    return df


def _baseline_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        max_features="sqrt",
        random_state=0,
        n_jobs=-1,
    )


def _boosting_model() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(random_state=0)


def _candidate_model_builders() -> dict[str, callable]:
    return {
        "rf": _baseline_model,
        "gb": _boosting_model,
    }


def _expanded_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=["number", "bool"]).columns
    cols = [col for col in numeric if col not in NON_FEATURE_COLUMNS]
    # Keep target/probability outputs and other downstream metadata out of the candidate features.
    cols = [
        col
        for col in cols
        if not col.startswith("prob_")
        and col not in {"model_confidence", "model_margin", "needs_review"}
    ]
    return sorted(cols)


def _score(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def _fit_predict_with_builder(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    feature_columns: list[str],
    model_builder,
) -> tuple[object, pd.Series]:
    model = model_builder()
    model.fit(train_df[feature_columns], train_df["fiber_type"].astype(str))
    pred = pd.Series(
        model.predict(dev_df[feature_columns]),
        index=dev_df.index,
        name="predicted_fiber_type",
    )
    return model, pred


def train_and_compare(df: pd.DataFrame) -> tuple[str, dict[str, object]]:
    train_df = df.loc[df["split"].astype(str) == "train"].copy()
    dev_df = df.loc[df["split"].astype(str) == "dev"].copy()
    if train_df.empty or dev_df.empty:
        raise ValueError("Feature table must contain both train and dev rows.")

    feature_sets = {
        "baseline": list(FROZEN_ALPHA_BASELINE_FEATURES),
        "expanded": _expanded_feature_columns(df),
    }

    results: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    predictions_frames: list[pd.DataFrame] = []

    for model_suffix, model_builder in _candidate_model_builders().items():
        for feature_prefix, feature_columns in feature_sets.items():
            name = f"{feature_prefix}_{model_suffix}"
            model, pred = _fit_predict_with_builder(
                train_df,
                dev_df,
                feature_columns,
                model_builder,
            )
            metrics = _score(dev_df["fiber_type"].astype(str), pred.astype(str))
            rows.append(
                {
                    "candidate_name": name,
                    "feature_set": feature_prefix,
                    "model_family": model_suffix,
                    "feature_count": len(feature_columns),
                    **metrics,
                }
            )
            pred_df = dev_df[
                [
                    col
                    for col in (
                        "image_id",
                        "label",
                        "fiber_type",
                        "fiber_type_source",
                        "needs_review",
                        "model_confidence",
                        "model_margin",
                    )
                    if col in dev_df.columns
                ]
            ].copy()
            pred_df["candidate_name"] = name
            pred_df["feature_set"] = feature_prefix
            pred_df["model_family"] = model_suffix
            pred_df["predicted_fiber_type"] = pred.astype(str).to_numpy()
            predictions_frames.append(pred_df)
            results[name] = {
                "model": model,
                "feature_columns": feature_columns,
                "predictions": pred_df,
            }

    metrics_df = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "accuracy", "feature_count"],
        ascending=[False, False, True],
        kind="stable",
    )
    best_name = str(metrics_df.iloc[0]["candidate_name"])
    results["metrics_df"] = metrics_df
    results["predictions_df"] = pd.concat(predictions_frames, ignore_index=True)
    results["train_rows"] = len(train_df)
    results["dev_rows"] = len(dev_df)
    return best_name, results


def main() -> None:
    args = build_parser().parse_args()
    df = _load_feature_table(args.feature_table)
    best_name, results = train_and_compare(df)

    best = results[best_name]
    model = best["model"]
    feature_columns = best["feature_columns"]

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_model)

    if args.metrics_csv is not None:
        args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        results["metrics_df"].to_csv(args.metrics_csv, index=False)

    if args.predictions_csv is not None:
        args.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        results["predictions_df"].to_csv(args.predictions_csv, index=False)

    if args.report_txt is not None:
        dev_df = df.loc[df["split"].astype(str) == "dev"].copy()
        pred = pd.Series(model.predict(dev_df[feature_columns]), index=dev_df.index)
        args.report_txt.parent.mkdir(parents=True, exist_ok=True)
        args.report_txt.write_text(
            classification_report(dev_df["fiber_type"].astype(str), pred.astype(str), digits=3)
        )

    print(f"saved model: {args.output_model}")
    print(f"best candidate: {best_name}")
    print(f"train rows: {results['train_rows']}")
    print(f"dev rows: {results['dev_rows']}")
    print("candidate comparison:")
    print(results["metrics_df"].to_string(index=False))
    print("selected feature columns:")
    print("\n".join(feature_columns))


if __name__ == "__main__":
    main()
