from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

from src.quantify_classify import FROZEN_ALPHA_BASELINE_FEATURES
from src.train_candidate_from_feature_table import (
    _candidate_model_builders,
    _expanded_feature_columns,
    _load_feature_table,
)

CANONICAL_CLASSES = ("iib", "iia", "iix")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the current pipeline label and first-pass candidate models against "
            "the reviewed audit benchmark table."
        )
    )
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser


def _load_benchmark(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {"image_id", "label", "audit_final_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    out = df.copy()
    out["label"] = out["label"].astype(int)
    out["audit_final_label"] = (
        out["audit_final_label"].fillna("").astype(str).str.strip().str.lower()
    )
    return out


def _train_candidates(feature_table: pd.DataFrame) -> dict[str, tuple[object, list[str]]]:
    train_df = feature_table.loc[feature_table["split"].astype(str) == "train"].copy()
    if train_df.empty:
        raise ValueError("Feature table must contain a train split to fit benchmark comparators.")

    feature_sets = {
        "baseline": list(FROZEN_ALPHA_BASELINE_FEATURES),
        "expanded": _expanded_feature_columns(feature_table),
    }
    models: dict[str, tuple[object, list[str]]] = {}
    for model_suffix, model_builder in _candidate_model_builders().items():
        for feature_prefix, feature_columns in feature_sets.items():
            name = f"{feature_prefix}_{model_suffix}"
            model = model_builder()
            model.fit(train_df[feature_columns], train_df["fiber_type"].astype(str))
            models[name] = (model, feature_columns)
    return models


def _benchmark_scope_rows(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if split_name == "all":
        return df
    if split_name == "nontrain":
        return df.loc[df["split"].astype(str) != "train"].copy()
    return df.loc[df["split"].astype(str) == split_name].copy()


def evaluate_against_audit_benchmark(
    feature_table: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    feature_keep = [c for c in feature_table.columns if c not in {"audit_final_label"}]
    merged = benchmark.merge(
        feature_table[feature_keep],
        on=["image_id", "label"],
        how="left",
        suffixes=("", "_feature"),
        validate="one_to_one",
    )

    models = _train_candidates(feature_table)
    merged["candidate_pred_pipeline_current"] = merged["fiber_type"].astype(str)
    for name, (model, feature_columns) in models.items():
        valid = merged[feature_columns].notna().all(axis=1)
        pred = pd.Series("", index=merged.index, dtype=object)
        if valid.any():
            pred.loc[valid] = model.predict(merged.loc[valid, feature_columns]).astype(str)
        merged[f"candidate_pred_{name}"] = pred

    comparator_cols = ["candidate_pred_pipeline_current"] + [
        f"candidate_pred_{name}" for name in sorted(models)
    ]

    metric_rows: list[dict[str, object]] = []
    report_blocks: list[str] = []
    benchmark_certain = merged.loc[merged["audit_final_label"].isin(CANONICAL_CLASSES)].copy()

    for scope in ("all", "nontrain", "dev", "heldout", "train"):
        scope_df = _benchmark_scope_rows(benchmark_certain, scope)
        if scope_df.empty:
            continue
        report_blocks.append(f"[scope={scope}] n={len(scope_df)}")
        for col in comparator_cols:
            pred = scope_df[col].fillna("").astype(str).str.lower()
            valid = pred.isin(CANONICAL_CLASSES)
            if not valid.any():
                continue
            y_true = scope_df.loc[valid, "audit_final_label"].astype(str)
            y_pred = pred.loc[valid]
            metric_rows.append(
                {
                    "scope": scope,
                    "comparator": col.removeprefix("candidate_pred_"),
                    "n_rows": int(len(scope_df)),
                    "n_scored": int(valid.sum()),
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                }
            )
            report_blocks.append(
                f"- {col.removeprefix('candidate_pred_')}: "
                f"accuracy={accuracy_score(y_true, y_pred):.4f}, "
                f"balanced_accuracy={balanced_accuracy_score(y_true, y_pred):.4f}, "
                f"n_scored={int(valid.sum())}"
            )
            if scope in {"all", "nontrain"}:
                report_blocks.append(
                    classification_report(
                        y_true,
                        y_pred,
                        labels=list(CANONICAL_CLASSES),
                        zero_division=0,
                    )
                )

    prediction_columns = [
        "image_id",
        "label",
        "split",
        "fiber_type",
        "audit_final_label",
        "audit_bucket_list",
        "needs_review",
        "model_confidence",
        "model_margin",
        *comparator_cols,
    ]
    predictions = merged.loc[:, [c for c in prediction_columns if c in merged.columns]].copy()
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["scope", "balanced_accuracy", "accuracy"],
        ascending=[True, False, False],
        kind="stable",
    )
    report = "\n\n".join(report_blocks)
    return metrics, predictions, report


def main() -> None:
    args = build_parser().parse_args()
    feature_table = _load_feature_table(args.feature_table)
    benchmark = _load_benchmark(args.benchmark)
    metrics, predictions, report = evaluate_against_audit_benchmark(feature_table, benchmark)

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(args.metrics_output, index=False)
    predictions.to_csv(args.predictions_output, index=False)
    args.report_output.write_text(report + "\n", encoding="utf-8")

    uncertain_rows = int((benchmark["audit_final_label"] == "uncertain").sum())
    print(f"benchmark rows total: {len(benchmark)}")
    scored_rows = int(benchmark["audit_final_label"].isin(CANONICAL_CLASSES).sum())
    print(f"benchmark rows scored: {scored_rows}")
    print(f"benchmark rows uncertain: {uncertain_rows}")
    if not metrics.empty:
        print("top metrics:")
        print(metrics.head(10).to_string(index=False))
    print(f"saved metrics: {args.metrics_output}")
    print(f"saved predictions: {args.predictions_output}")
    print(f"saved report: {args.report_output}")


if __name__ == "__main__":
    main()
