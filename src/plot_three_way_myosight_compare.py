from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.plot_biological_story import add_biology_metadata
from src.plot_validation_summary import INPUT_LABELS, prep_table

METHOD_COLORS = {
    "Frozen": "#2f6f9f",
    "Candidate": "#b45f06",
    "MyoSight": "#777777",
}


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def load_and_harmonize(path: Path, method_name: str) -> pd.DataFrame:
    df = prep_table(path)
    df = add_biology_metadata(df)
    df["method_name"] = method_name
    df["input_label"] = df["validation_input_kind"].map(INPUT_LABELS).fillna(
        df["validation_input_kind"]
    )
    return df


def build_long_counts(frozen: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method_name, df in [("Frozen", frozen), ("Candidate", candidate)]:
        for _, row in df.iterrows():
            rows.append(
                {
                    "image_id": row["image_id"],
                    "age": row["age"],
                    "age_month": row["age_month"],
                    "genotype": row["genotype"],
                    "input_label": row["input_label"],
                    "training_or_heldout": row["training_or_heldout"],
                    "method": method_name,
                    "total_fibers": row["pipeline_total_fibers"],
                    "needs_review_rate_pct": row["pipeline_needs_review_rate"] * 100,
                    "signal_warning_rate_pct": row["pipeline_signal_warning_rate"] * 100,
                }
            )
            rows.append(
                {
                    "image_id": row["image_id"],
                    "age": row["age"],
                    "age_month": row["age_month"],
                    "genotype": row["genotype"],
                    "input_label": row["input_label"],
                    "training_or_heldout": row["training_or_heldout"],
                    "method": "MyoSight",
                    "total_fibers": row["myosight_total_fibers"],
                    "needs_review_rate_pct": pd.NA,
                    "signal_warning_rate_pct": pd.NA,
                }
            )
    return pd.DataFrame(rows)


def build_long_type_diffs(frozen: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method_name, df in [("Frozen", frozen), ("Candidate", candidate)]:
        for _, row in df.iterrows():
            for fiber_type in ("iib", "iia", "iix"):
                rows.append(
                    {
                        "image_id": row["image_id"],
                        "age": row["age"],
                        "age_month": row["age_month"],
                        "genotype": row["genotype"],
                        "input_label": row["input_label"],
                        "training_or_heldout": row["training_or_heldout"],
                        "method": method_name,
                        "fiber_type": fiber_type.upper().replace("IIB", "IIb").replace(
                            "IIA", "IIa"
                        ).replace("IIX", "IIx"),
                        "myosight_pct": row[f"myosight_{fiber_type}_pct"] * 100,
                        "pipeline_pct": row[f"pipeline_{fiber_type}_pct"] * 100,
                        "pct_diff_points": row[f"{fiber_type}_pct_diff_pipeline_minus_myosight"]
                        * 100,
                        "abs_pct_diff_points": abs(
                            row[f"{fiber_type}_pct_diff_pipeline_minus_myosight"] * 100
                        ),
                    }
                )
    return pd.DataFrame(rows)


def write_summary_tables(
    frozen: pd.DataFrame,
    candidate: pd.DataFrame,
    outdir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts_long = build_long_counts(frozen, candidate)
    type_diffs = build_long_type_diffs(frozen, candidate)

    grouped = (
        type_diffs.groupby(["method", "fiber_type"])
        .agg(
            n_images=("image_id", "nunique"),
            mean_abs_pct_diff_points=("abs_pct_diff_points", "mean"),
            median_abs_pct_diff_points=("abs_pct_diff_points", "median"),
            mean_signed_pct_diff_points=("pct_diff_points", "mean"),
        )
        .reset_index()
    )
    grouped.to_csv(outdir / "three_way_type_difference_summary.csv", index=False)

    review = (
        counts_long[counts_long["method"].isin(["Frozen", "Candidate"])]
        .groupby("method")
        .agg(
            n_images=("image_id", "nunique"),
            mean_total_fibers=("total_fibers", "mean"),
            mean_review_rate_pct=("needs_review_rate_pct", "mean"),
            mean_signal_warning_rate_pct=("signal_warning_rate_pct", "mean"),
        )
        .reset_index()
    )
    review.to_csv(outdir / "three_way_review_summary.csv", index=False)
    return grouped, review


def plot_count_agreement(frozen: pd.DataFrame, candidate: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), sharex=True, sharey=True)
    for ax, (method_name, df) in zip(
        axes,
        [("Frozen", frozen), ("Candidate", candidate)],
        strict=True,
    ):
        ax.scatter(
            df["myosight_total_fibers"],
            df["pipeline_total_fibers"],
            s=72,
            color=METHOD_COLORS[method_name],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9,
        )
        max_count = max(df["myosight_total_fibers"].max(), df["pipeline_total_fibers"].max()) * 1.05
        ax.plot([0, max_count], [0, max_count], color="#333333", linewidth=1.1)
        ax.set_xlim(0, max_count)
        ax.set_ylim(0, max_count)
        ax.set_title(method_name)
        ax.set_xlabel("MyoSight fiber count")
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
    axes[0].set_ylabel("Pipeline fiber count")
    fig.suptitle("Fiber Count Agreement vs MyoSight", y=1.02)
    save_fig(outdir / "three_way_fiber_count_agreement.png")


def plot_type_difference_bars(type_diffs: pd.DataFrame, outdir: Path) -> None:
    plot_df = (
        type_diffs.groupby(["method", "fiber_type"])
        .agg(mean_abs_pct_diff_points=("abs_pct_diff_points", "mean"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    sns.barplot(
        data=plot_df,
        x="fiber_type",
        y="mean_abs_pct_diff_points",
        hue="method",
        palette={k: METHOD_COLORS[k] for k in ["Frozen", "Candidate"]},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean absolute difference vs MyoSight (pct points)")
    ax.set_title("Class Proportion Error vs MyoSight")
    ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)
    ax.legend(frameon=False, title="")
    save_fig(outdir / "three_way_type_abs_difference_bar.png")


def plot_type_difference_strip(type_diffs: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.8), sharey=True)
    for ax, fiber_type in zip(axes, ["IIb", "IIa", "IIx"], strict=True):
        sub = type_diffs[type_diffs["fiber_type"].eq(fiber_type)].copy()
        sns.stripplot(
            data=sub,
            x="method",
            y="pct_diff_points",
            order=["Frozen", "Candidate"],
            palette={k: METHOD_COLORS[k] for k in ["Frozen", "Candidate"]},
            jitter=0.18,
            size=7,
            alpha=0.85,
            ax=ax,
        )
        ax.axhline(0, color="#333333", linewidth=1.0)
        ax.set_title(fiber_type)
        ax.set_xlabel("")
        ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)
    axes[0].set_ylabel("Pipeline minus MyoSight (pct points)")
    fig.suptitle("Per-image Class Proportion Differences", y=1.02)
    save_fig(outdir / "three_way_type_difference_strip.png")


def plot_review_burden(frozen: pd.DataFrame, candidate: pd.DataFrame, outdir: Path) -> None:
    rows = []
    for method_name, df in [("Frozen", frozen), ("Candidate", candidate)]:
        for _, row in df.iterrows():
            rows.append(
                {
                    "image_id": row["image_id"],
                    "method": method_name,
                    "metric": "Needs review",
                    "rate_pct": row["pipeline_needs_review_rate"] * 100,
                }
            )
            rows.append(
                {
                    "image_id": row["image_id"],
                    "method": method_name,
                    "metric": "Signal warnings",
                    "rate_pct": row["pipeline_signal_warning_rate"] * 100,
                }
            )
    plot_df = pd.DataFrame(rows)
    grid = sns.catplot(
        data=plot_df,
        x="method",
        y="rate_pct",
        col="metric",
        kind="box",
        palette={k: METHOD_COLORS[k] for k in ["Frozen", "Candidate"]},
        height=4.8,
        aspect=0.9,
        sharey=False,
    )
    grid.set_axis_labels("", "Rate (%)")
    grid.set_titles("{col_name}")
    for ax in grid.axes.flat:
        ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)
    save_fig(outdir / "three_way_review_burden_box.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a 3-way Frozen vs Candidate vs MyoSight comparison report."
    )
    parser.add_argument(
        "--frozen-summary",
        type=Path,
        default=Path("outputs/validation/myosight_validation_consolidated_image_summary.csv"),
    )
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation/figures_three_way_compare"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
    frozen = load_and_harmonize(args.frozen_summary, "Frozen")
    candidate = load_and_harmonize(args.candidate_summary, "Candidate")
    common = sorted(set(frozen["image_id"]) & set(candidate["image_id"]))
    frozen = frozen[frozen["image_id"].isin(common)].copy()
    candidate = candidate[candidate["image_id"].isin(common)].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    type_summary, review_summary = write_summary_tables(frozen, candidate, args.output_dir)
    plot_count_agreement(frozen, candidate, args.output_dir)
    type_diffs = build_long_type_diffs(frozen, candidate)
    plot_type_difference_bars(type_diffs, args.output_dir)
    plot_type_difference_strip(type_diffs, args.output_dir)
    plot_review_burden(frozen, candidate, args.output_dir)
    print(f"saved figures to {args.output_dir}")
    print("type difference summary:")
    print(type_summary.to_string(index=False))
    print("review summary:")
    print(review_summary.to_string(index=False))


if __name__ == "__main__":
    main()
