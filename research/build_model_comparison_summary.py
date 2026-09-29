from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a compact summary table of key manual-benchmark model results."
    )
    parser.add_argument("--weighted-manual-metrics", type=Path, required=True)
    parser.add_argument("--iia-gate-metrics", type=Path, required=True)
    parser.add_argument("--recipe-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def build_summary(
    weighted_manual_metrics: pd.DataFrame,
    iia_gate_metrics: pd.DataFrame,
    recipe_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    baseline_row = weighted_manual_metrics.loc[
        weighted_manual_metrics["candidate_name"].astype(str).eq("baseline_gb")
    ].iloc[0]
    rows.append(
        {
            "experiment": "weighted_manual_baseline_gb",
            "family": "manual_high + baseline_default",
            "soft_iia_gate": "none",
            "accuracy": float(baseline_row["accuracy"]),
            "balanced_accuracy": float(baseline_row["balanced_accuracy"]),
            "iia_precision": pd.NA,
            "iia_recall": pd.NA,
            "notes": "Current weighted-manual baseline_gb incumbent before gate analysis.",
        }
    )

    gate_keep = iia_gate_metrics.loc[
        iia_gate_metrics["strategy"].astype(str).isin(
            ["baseline_gb", "baseline_gb_gated_iia_q0.01"]
        )
    ].copy()
    for _, row in gate_keep.iterrows():
        strategy = str(row["strategy"])
        rows.append(
            {
                "experiment": strategy,
                "family": (
                    "manual_high + baseline_default + soft_iia_gate"
                    if "gated" in strategy
                    else "manual_high + baseline_default"
                ),
                "soft_iia_gate": "q0.01" if "gated" in strategy else "none",
                "accuracy": float(row["accuracy"]),
                "balanced_accuracy": float(row["balanced_accuracy"]),
                "iia_precision": float(row["iia_precision"]),
                "iia_recall": float(row["iia_recall"]),
                "notes": (
                    "Best-performing soft IIa gate candidate on protected manual holdout."
                    if strategy.endswith("q0.01")
                    else "Same weighted baseline_gb row with IIa metrics attached."
                ),
            }
        )

    recipe_order = [
        "manual_only_high",
        "manual_high_myo_medium",
        "manual_high_myo_medium_baseline_light",
        "manual_high_myo_medium_baseline_light_soft_iia_q0.01",
    ]
    recipe_subset = recipe_metrics.loc[
        recipe_metrics["recipe"].astype(str).isin(recipe_order)
    ].copy()
    for _, row in recipe_subset.iterrows():
        recipe = str(row["recipe"])
        rows.append(
            {
                "experiment": recipe,
                "family": recipe,
                "soft_iia_gate": "q0.01" if recipe.endswith("q0.01") else "none",
                "accuracy": float(row["accuracy"]),
                "balanced_accuracy": float(row["balanced_accuracy"]),
                "iia_precision": float(row["iia_precision"]),
                "iia_recall": float(row["iia_recall"]),
                "notes": (
                    f"train_rows={int(row['train_rows'])}, "
                    f"manual_rows={int(row['manual_rows'])}, "
                    f"myosight_rows={int(row['myosight_rows'])}, "
                    f"baseline_rows={int(row['baseline_rows'])}"
                    if pd.notna(row.get("train_rows"))
                    else "Gated variant of the recipe immediately above."
                ),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["balanced_accuracy", "accuracy"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)


def main() -> None:
    args = build_parser().parse_args()
    weighted_manual_metrics = pd.read_csv(args.weighted_manual_metrics)
    iia_gate_metrics = pd.read_csv(args.iia_gate_metrics)
    recipe_metrics = pd.read_csv(args.recipe_metrics)
    summary = build_summary(weighted_manual_metrics, iia_gate_metrics, recipe_metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))
    print(f"saved summary: {args.output}")


if __name__ == "__main__":
    main()
