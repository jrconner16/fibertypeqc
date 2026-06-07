from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analyze_iia_gate import _apply_iia_gate, _gate_mask, derive_iia_gate_thresholds
from src.compare_supervision_recipes import (
    _assemble_recipe_training_rows,
    _load_matched_myosight,
    _load_true_iia_reviewed,
    _metric_row,
    _prepare_base_table,
)
from src.quantify_classify import FROZEN_ALPHA_BASELINE_FEATURES
from src.train_candidate_from_feature_table import _boosting_model, _load_feature_table
from src.train_weighted_candidate_from_audit import _load_manual_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep matched MyoSight and baseline-light weights around the current "
            "manual-heavy incumbent and compare gated mixed-recipe variants."
        )
    )
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--manual-split", type=Path, required=True)
    parser.add_argument("--matched-myosight-audit", type=Path, required=True)
    parser.add_argument("--true-iia-reviewed-glob", type=str, required=True)
    parser.add_argument("--reference-metrics", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--manual-weight", type=float, default=8.0)
    parser.add_argument(
        "--myosight-weights",
        type=str,
        default="1.0,2.0,3.0,4.0",
        help="Comma-separated matched MyoSight weights to evaluate.",
    )
    parser.add_argument(
        "--baseline-weights",
        type=str,
        default="0.1,0.25,0.5,1.0",
        help="Comma-separated baseline/self-label weights to evaluate.",
    )
    return parser


def _parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        values.append(float(text))
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def _load_reference(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = df.loc[
        df["strategy"].astype(str).eq("baseline_gb_gated_iia_q0.01")
    ].copy()
    if keep.empty:
        raise ValueError(f"No baseline_gb_gated_iia_q0.01 row found in {path}")
    keep = keep.assign(
        recipe="reference_baseline_gb_gated_iia_q0.01",
        manual_weight=pd.NA,
        myosight_weight=pd.NA,
        baseline_weight=pd.NA,
        source="reference",
    )
    return keep.rename(columns={"strategy": "reference_strategy"})


def sweep_weights(
    feature_table: pd.DataFrame,
    manual_split: pd.DataFrame,
    matched_myosight: pd.DataFrame,
    true_iia_reviewed: pd.DataFrame,
    reference_metrics: pd.DataFrame,
    *,
    manual_weight: float,
    myosight_weights: list[float],
    baseline_weights: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ref = reference_metrics.iloc[0]
    rows.append(
        {
            "recipe": "reference_baseline_gb_gated_iia_q0.01",
            "manual_weight": pd.NA,
            "myosight_weight": pd.NA,
            "baseline_weight": pd.NA,
            "accuracy": float(ref["accuracy"]),
            "balanced_accuracy": float(ref["balanced_accuracy"]),
            "iia_precision": float(ref["iia_precision"]),
            "iia_recall": float(ref["iia_recall"]),
            "delta_vs_reference_balanced_accuracy": 0.0,
            "delta_vs_reference_accuracy": 0.0,
            "source": "reference",
        }
    )
    ref_bacc = float(ref["balanced_accuracy"])
    ref_acc = float(ref["accuracy"])
    merged, eval_df = _prepare_base_table(feature_table, manual_split, matched_myosight)
    thresholds = derive_iia_gate_thresholds(true_iia_reviewed, gate_quantile=0.01)
    gate_ok = _gate_mask(eval_df, thresholds)
    y_true = eval_df["audit_final_label"].astype(str).str.lower()
    feature_columns = list(FROZEN_ALPHA_BASELINE_FEATURES)

    for myo_weight in myosight_weights:
        for base_weight in baseline_weights:
            train_df = _assemble_recipe_training_rows(
                merged,
                recipe_name="manual_high_myo_medium_baseline_light",
                manual_weight=manual_weight,
                myosight_weight=myo_weight,
                baseline_weight=base_weight,
            )
            model = _boosting_model()
            model.fit(
                train_df[feature_columns],
                train_df["training_label"].astype(str),
                sample_weight=train_df["sample_weight"].astype(float),
            )
            pred = pd.Series(
                model.predict(eval_df[feature_columns]),
                index=eval_df.index,
                name="predicted_fiber_type",
            ).astype(str).str.lower()
            gated_pred = _apply_iia_gate(pred, gate_ok)
            target = _metric_row(
                "manual_high_myo_medium_baseline_light_soft_iia_q0.01",
                y_true,
                gated_pred,
            )
            rows.append(
                {
                    "recipe": str(target["recipe"]),
                    "manual_weight": manual_weight,
                    "myosight_weight": myo_weight,
                    "baseline_weight": base_weight,
                    "accuracy": float(target["accuracy"]),
                    "balanced_accuracy": float(target["balanced_accuracy"]),
                    "iia_precision": float(target["iia_precision"]),
                    "iia_recall": float(target["iia_recall"]),
                    "delta_vs_reference_balanced_accuracy": float(
                        target["balanced_accuracy"] - ref_bacc
                    ),
                    "delta_vs_reference_accuracy": float(target["accuracy"] - ref_acc),
                    "source": "sweep",
                }
            )

    out = pd.DataFrame(rows).sort_values(
        [
            "source",
            "balanced_accuracy",
            "accuracy",
            "myosight_weight",
            "baseline_weight",
        ],
        ascending=[True, False, False, True, True],
        kind="stable",
    )
    return out.reset_index(drop=True)


def main() -> None:
    args = build_parser().parse_args()
    feature_table = _load_feature_table(args.feature_table)
    manual_split = _load_manual_split(args.manual_split)
    matched_myosight = _load_matched_myosight(args.matched_myosight_audit)
    true_iia_reviewed = _load_true_iia_reviewed(args.true_iia_reviewed_glob)
    reference_metrics = _load_reference(args.reference_metrics)
    myosight_weights = _parse_float_list(args.myosight_weights)
    baseline_weights = _parse_float_list(args.baseline_weights)
    summary = sweep_weights(
        feature_table,
        manual_split,
        matched_myosight,
        true_iia_reviewed,
        reference_metrics,
        manual_weight=args.manual_weight,
        myosight_weights=myosight_weights,
        baseline_weights=baseline_weights,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    print(summary.to_string(index=False))
    print(f"saved sweep: {args.output_csv}")


if __name__ == "__main__":
    main()
