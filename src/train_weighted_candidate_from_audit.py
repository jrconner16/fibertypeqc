from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

from src.evaluate_against_audit_benchmark import CANONICAL_CLASSES
from src.quantify_classify import FROZEN_ALPHA_BASELINE_FEATURES
from src.train_candidate_from_feature_table import (
    _candidate_model_builders,
    _expanded_feature_columns,
    _load_feature_table,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train an experimental weighted-supervision candidate model using reviewed "
            "manual labels as a high-weight augmentation."
        )
    )
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--manual-split", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--report-txt", type=Path, required=True)
    parser.add_argument(
        "--manual-weight",
        type=float,
        default=8.0,
        help="Sample weight assigned to reviewed manual-train rows.",
    )
    parser.add_argument(
        "--model-families",
        type=str,
        default="rf,gb",
        help="Comma-separated model families to evaluate.",
    )
    parser.add_argument(
        "--feature-sets",
        type=str,
        default="baseline,expanded",
        help="Comma-separated feature sets to evaluate.",
    )
    return parser


def _load_manual_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {"image_id", "label", "audit_final_label", "manual_supervision_split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    out = df.copy()
    out["image_id"] = out["image_id"].astype(str)
    out["label"] = out["label"].astype(int)
    out["audit_final_label"] = (
        out["audit_final_label"].fillna("").astype(str).str.lower().str.strip()
    )
    out["manual_supervision_split"] = (
        out["manual_supervision_split"].fillna("").astype(str).str.lower().str.strip()
    )
    return out


def _prepare_weighted_training_table(
    feature_table: pd.DataFrame,
    manual_split: pd.DataFrame,
    *,
    manual_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manual_keep = manual_split[
        [
            "image_id",
            "label",
            "audit_final_label",
            "manual_supervision_split",
        ]
    ].copy()
    merged = feature_table.merge(
        manual_keep,
        on=["image_id", "label"],
        how="left",
        validate="one_to_one",
    )

    merged["manual_supervision_split"] = (
        merged["manual_supervision_split"].fillna("").astype(str).str.lower()
    )
    merged["audit_final_label"] = merged["audit_final_label"].fillna("").astype(str).str.lower()

    is_manual_train = (
        merged["manual_supervision_split"].eq("manual_train")
        & merged["audit_final_label"].isin(CANONICAL_CLASSES)
    )
    is_manual_eval = (
        merged["manual_supervision_split"].eq("manual_eval_holdout")
        & merged["audit_final_label"].isin(CANONICAL_CLASSES)
    )

    train_rows = merged.loc[
        ((merged["split"].astype(str) == "train") & ~is_manual_eval) | is_manual_train
    ].copy()
    train_rows["training_label"] = train_rows["fiber_type"].astype(str)
    train_rows.loc[is_manual_train.loc[train_rows.index], "training_label"] = train_rows.loc[
        is_manual_train.loc[train_rows.index], "audit_final_label"
    ].astype(str)
    train_rows["sample_weight"] = 1.0
    train_rows.loc[is_manual_train.loc[train_rows.index], "sample_weight"] = float(manual_weight)

    eval_rows = merged.loc[is_manual_eval].copy()
    if train_rows.empty:
        raise ValueError("No training rows available after applying manual supervision split.")
    if eval_rows.empty:
        raise ValueError("No manual_eval_holdout rows available for evaluation.")
    return train_rows, eval_rows


def _score(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def train_weighted_candidates(
    feature_table: pd.DataFrame,
    manual_split: pd.DataFrame,
    *,
    manual_weight: float = 8.0,
    model_families: tuple[str, ...] = ("rf", "gb"),
    feature_sets_filter: tuple[str, ...] = ("baseline", "expanded"),
) -> tuple[str, dict[str, object]]:
    train_df, eval_df = _prepare_weighted_training_table(
        feature_table, manual_split, manual_weight=manual_weight
    )

    all_feature_sets = {
        "baseline": list(FROZEN_ALPHA_BASELINE_FEATURES),
        "expanded": _expanded_feature_columns(feature_table),
    }
    feature_sets = {
        name: cols for name, cols in all_feature_sets.items() if name in set(feature_sets_filter)
    }
    model_builders = {
        name: builder
        for name, builder in _candidate_model_builders().items()
        if name in set(model_families)
    }
    if not feature_sets:
        raise ValueError("No feature sets selected.")
    if not model_builders:
        raise ValueError("No model families selected.")

    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    results: dict[str, object] = {
        "train_rows": len(train_df),
        "eval_rows": len(eval_df),
        "manual_train_rows": int((train_df["sample_weight"] > 1.0).sum()),
    }

    for model_suffix, model_builder in model_builders.items():
        for feature_prefix, feature_columns in feature_sets.items():
            name = f"{feature_prefix}_{model_suffix}"
            model = model_builder()
            model.fit(
                train_df[feature_columns],
                train_df["training_label"].astype(str),
                sample_weight=train_df["sample_weight"].astype(float),
            )
            pred = pd.Series(
                model.predict(eval_df[feature_columns]),
                index=eval_df.index,
                name="predicted_fiber_type",
            )
            metrics = _score(eval_df["audit_final_label"].astype(str), pred.astype(str))
            rows.append(
                {
                    "candidate_name": name,
                    "feature_set": feature_prefix,
                    "model_family": model_suffix,
                    "feature_count": len(feature_columns),
                    "manual_weight": float(manual_weight),
                    **metrics,
                }
            )
            pred_df = eval_df[
                [
                    col
                    for col in (
                        "image_id",
                        "label",
                        "split",
                        "fiber_type",
                        "audit_final_label",
                        "manual_supervision_split",
                        "needs_review",
                        "model_confidence",
                        "model_margin",
                    )
                    if col in eval_df.columns
                ]
            ].copy()
            pred_df["candidate_name"] = name
            pred_df["feature_set"] = feature_prefix
            pred_df["model_family"] = model_suffix
            pred_df["predicted_fiber_type"] = pred.astype(str).to_numpy()
            prediction_frames.append(pred_df)
            results[name] = {"model": model, "feature_columns": feature_columns}

    metrics_df = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "accuracy", "feature_count"],
        ascending=[False, False, True],
        kind="stable",
    )
    best_name = str(metrics_df.iloc[0]["candidate_name"])
    results["metrics_df"] = metrics_df
    results["predictions_df"] = pd.concat(prediction_frames, ignore_index=True)
    results["eval_df"] = eval_df
    return best_name, results


def main() -> None:
    args = build_parser().parse_args()
    feature_table = _load_feature_table(args.feature_table)
    manual_split = _load_manual_split(args.manual_split)
    model_families = tuple(s.strip() for s in args.model_families.split(",") if s.strip())
    feature_sets = tuple(s.strip() for s in args.feature_sets.split(",") if s.strip())
    best_name, results = train_weighted_candidates(
        feature_table,
        manual_split,
        manual_weight=args.manual_weight,
        model_families=model_families,
        feature_sets_filter=feature_sets,
    )

    best = results[best_name]
    model = best["model"]
    feature_columns = best["feature_columns"]
    eval_df = results["eval_df"]
    pred = pd.Series(model.predict(eval_df[feature_columns]), index=eval_df.index)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_txt.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, args.output_model)
    results["metrics_df"].to_csv(args.metrics_csv, index=False)
    results["predictions_df"].to_csv(args.predictions_csv, index=False)
    args.report_txt.write_text(
        classification_report(
            eval_df["audit_final_label"].astype(str),
            pred.astype(str),
            labels=list(CANONICAL_CLASSES),
            zero_division=0,
            digits=3,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"saved model: {args.output_model}")
    print(f"best candidate: {best_name}")
    print(f"train rows: {results['train_rows']}")
    print(f"manual-train override rows: {results['manual_train_rows']}")
    print(f"manual-eval rows: {results['eval_rows']}")
    print("candidate comparison:")
    print(results["metrics_df"].to_string(index=False))


if __name__ == "__main__":
    main()
