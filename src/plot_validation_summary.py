from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE = {
    "direct_czi": "#2f6f9f",
    "section_series_export": "#b45f06",
}
INPUT_LABELS = {
    "direct_czi": "Direct CZI",
    "section_series_export": "Section export",
}
PLOT_PALETTE = {INPUT_LABELS[key]: color for key, color in PALETTE.items()}
TYPE_COLORS = {
    "IIb": "#8b4a35",
    "IIa": "#179c52",
    "IIx": "#327c8a",
}


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def prep_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "fiber_count_pct_diff" not in df.columns:
        df["fiber_count_pct_diff"] = (
            (df["pipeline_total_fibers"] - df["myosight_total_fibers"])
            / df["myosight_total_fibers"]
            * 100
        )
    if "fiber_count_abs_pct_diff" not in df.columns:
        df["fiber_count_abs_pct_diff"] = df["fiber_count_pct_diff"].abs()
    return df


def plot_count_scatter(df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    sns.scatterplot(
        data=df,
        x="myosight_total_fibers",
        y="pipeline_total_fibers",
        hue="input_label",
        palette=PLOT_PALETTE,
        s=88,
        edgecolor="white",
        linewidth=0.8,
        ax=ax,
    )
    max_count = max(df["myosight_total_fibers"].max(), df["pipeline_total_fibers"].max()) * 1.05
    ax.plot([0, max_count], [0, max_count], color="#333333", linewidth=1.2, label="identity")
    ax.set_xlim(0, max_count)
    ax.set_ylim(0, max_count)
    ax.set_xlabel("MyoSight fiber count")
    ax.set_ylabel("Pipeline fiber count")
    ax.set_title("Fiber Count Agreement")
    ax.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left", fontsize=9, title="")
    save_fig(outdir / "fiber_count_agreement.png")


def plot_count_scatter_lab(df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.scatter(
        df["myosight_total_fibers"],
        df["pipeline_total_fibers"],
        s=84,
        color="#2f6f9f",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.92,
    )
    max_count = max(df["myosight_total_fibers"].max(), df["pipeline_total_fibers"].max()) * 1.05
    ax.plot([0, max_count], [0, max_count], color="#333333", linewidth=1.2)
    ax.set_xlim(0, max_count)
    ax.set_ylim(0, max_count)
    ax.set_xlabel("MyoSight fiber count")
    ax.set_ylabel("Pipeline fiber count")
    ax.set_title("Fiber Counts Track MyoSight")
    ax.grid(True, color="#e5e5e5", linewidth=0.8)
    save_fig(outdir / "lab_fiber_count_agreement.png")


def plot_count_error_lab(df: pd.DataFrame, outdir: Path) -> None:
    plot_df = df.sort_values("fiber_count_pct_diff")
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    colors = np.where(plot_df["fiber_count_pct_diff"].abs() <= 20, "#2f6f9f", "#9f4f2f")
    ax.barh(plot_df["image_id"], plot_df["fiber_count_pct_diff"], color=colors)
    ax.axvline(0, color="#333333", linewidth=1.0)
    ax.axvspan(-20, 20, color="#4c956c", alpha=0.10, zorder=0)
    ax.set_xlabel("Pipeline count difference vs MyoSight (%)")
    ax.set_ylabel("")
    ax.set_title("Most Images Are Within 20% of MyoSight Count")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, axis="x", color="#e5e5e5", linewidth=0.8)
    save_fig(outdir / "lab_fiber_count_percent_difference.png")


def plot_count_error(df: pd.DataFrame, outdir: Path) -> None:
    order = ["direct_czi", "section_series_export"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sns.stripplot(
        data=df,
        x="input_label",
        y="fiber_count_pct_diff",
        order=[INPUT_LABELS[k] for k in order],
        hue="training_or_heldout",
        dodge=True,
        jitter=0.18,
        size=7,
        alpha=0.85,
        ax=ax,
    )
    medians = df.groupby("input_label")["fiber_count_pct_diff"].median()
    for i, key in enumerate(order):
        label = INPUT_LABELS[key]
        if label in medians:
            ax.hlines(medians[label], i - 0.28, i + 0.28, color="#111111", linewidth=2.0)
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.axhspan(-10, 10, color="#4c956c", alpha=0.10, zorder=0)
    ax.set_xlabel("")
    ax.set_ylabel("Pipeline count difference vs MyoSight (%)")
    ax.set_title("Fiber Count Difference by Input Type")
    ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)
    ax.legend(frameon=False, title="", loc="lower left", fontsize=8)
    save_fig(outdir / "fiber_count_difference_by_input.png")


def type_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for typ in ("iib", "iia", "iix"):
            rows.append(
                {
                    "image_id": row["image_id"],
                    "validation_input_kind": row["validation_input_kind"],
                    "input_label": row["input_label"],
                    "training_or_heldout": row["training_or_heldout"],
                    "fiber_type": typ.upper().replace("II", "II"),
                    "MyoSight": row[f"myosight_{typ}_pct"] * 100,
                    "Pipeline": row[f"pipeline_{typ}_pct"] * 100,
                    "Difference": row[f"{typ}_pct_diff_pipeline_minus_myosight"] * 100,
                }
            )
    out = pd.DataFrame(rows)
    out["fiber_type"] = (
        out["fiber_type"]
        .str.replace("IIB", "IIb")
        .str.replace("IIA", "IIa")
        .str.replace("IIX", "IIx")
    )
    return out


def plot_type_scatter(df: pd.DataFrame, outdir: Path) -> None:
    long = type_long(df)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharex=True, sharey=True)
    for ax, typ in zip(axes, ["IIb", "IIa", "IIX"], strict=True):
        label = "IIx" if typ == "IIX" else typ
        sub = long[long["fiber_type"].eq(label)]
        sns.scatterplot(
            data=sub,
            x="MyoSight",
            y="Pipeline",
            hue="input_label",
            palette=PLOT_PALETTE,
            s=58,
            edgecolor="white",
            linewidth=0.6,
            legend=(ax is axes[0]),
            ax=ax,
        )
        ax.plot([0, 100], [0, 100], color="#333333", linewidth=1.0)
        ax.set_title(label)
        ax.set_xlabel("MyoSight (%)")
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        if ax is axes[0]:
            ax.set_ylabel("Pipeline (%)")
            ax.legend(frameon=False, fontsize=8, loc="upper left")
        else:
            ax.set_ylabel("")
    fig.suptitle("Fiber Type Proportion Agreement", y=1.02)
    save_fig(outdir / "fiber_type_proportion_agreement.png")


def plot_type_scatter_lab(df: pd.DataFrame, outdir: Path) -> None:
    long = type_long(df)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharex=True, sharey=True)
    for ax, label in zip(axes, ["IIb", "IIa", "IIx"], strict=True):
        sub = long[long["fiber_type"].eq(label)]
        ax.scatter(
            sub["MyoSight"],
            sub["Pipeline"],
            s=52,
            color=TYPE_COLORS[label],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
        )
        ax.plot([0, 100], [0, 100], color="#333333", linewidth=1.0)
        ax.set_title(label)
        ax.set_xlabel("MyoSight (%)")
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        if ax is axes[0]:
            ax.set_ylabel("Pipeline (%)")
        else:
            ax.set_ylabel("")
    fig.suptitle("Fiber Type Proportions Compared With MyoSight", y=1.02)
    save_fig(outdir / "lab_fiber_type_proportion_agreement.png")


def plot_type_profile_lab(df: pd.DataFrame, outdir: Path) -> None:
    rows = []
    for typ, label in [("iib", "IIb"), ("iia", "IIa"), ("iix", "IIx")]:
        rows.append(
            {
                "fiber_type": label,
                "method": "MyoSight",
                "mean_pct": df[f"myosight_{typ}_pct"].mean() * 100,
                "sem_pct": df[f"myosight_{typ}_pct"].sem() * 100,
            }
        )
        rows.append(
            {
                "fiber_type": label,
                "method": "Pipeline",
                "mean_pct": df[f"pipeline_{typ}_pct"].mean() * 100,
                "sem_pct": df[f"pipeline_{typ}_pct"].sem() * 100,
            }
        )
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    sns.barplot(
        data=plot_df,
        x="fiber_type",
        y="mean_pct",
        hue="method",
        palette={"MyoSight": "#777777", "Pipeline": "#2f6f9f"},
        ax=ax,
    )
    for container, method in zip(ax.containers, ["MyoSight", "Pipeline"], strict=False):
        sem = plot_df[plot_df["method"].eq(method)]["sem_pct"].to_numpy()
        for bar, err in zip(container, sem, strict=False):
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            ax.errorbar(x, y, yerr=err, color="#222222", linewidth=1, capsize=3)
    ax.set_xlabel("")
    ax.set_ylabel("Mean fiber type proportion (%)")
    ax.set_title("Cohort-Level Fiber Type Profile")
    ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)
    ax.legend(frameon=False, title="")
    save_fig(outdir / "lab_cohort_fiber_type_profile.png")


def plot_type_differences(df: pd.DataFrame, outdir: Path) -> None:
    long = type_long(df)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    sns.boxplot(
        data=long,
        x="fiber_type",
        y="Difference",
        hue="input_label",
        palette=PLOT_PALETTE,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=long,
        x="fiber_type",
        y="Difference",
        hue="input_label",
        dodge=True,
        palette=PLOT_PALETTE,
        alpha=0.55,
        size=4,
        legend=False,
        ax=ax,
    )
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Pipeline minus MyoSight (percentage points)")
    ax.set_title("Fiber Type Proportion Differences")
    ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)
    ax.legend(frameon=False, title="", fontsize=8)
    save_fig(outdir / "fiber_type_percentage_point_differences.png")


def plot_review_burden(df: pd.DataFrame, outdir: Path) -> None:
    plot_df = df.sort_values("pipeline_needs_review_rate")
    colors = plot_df["validation_input_kind"].map(PALETTE)
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    ax.barh(plot_df["image_id"], plot_df["pipeline_needs_review_rate"] * 100, color=colors)
    ax.set_xlabel("Fibers flagged for review (%)")
    ax.set_ylabel("")
    ax.set_title("Manual Review Burden")
    ax.grid(True, axis="x", color="#e5e5e5", linewidth=0.8)
    ax.tick_params(axis="y", labelsize=7)
    median = plot_df["pipeline_needs_review_rate"].median() * 100
    ax.axvline(median, color="#222222", linewidth=1.2)
    ax.text(median + 0.5, 0.5, f"median {median:.1f}%", va="center", fontsize=8)
    save_fig(outdir / "manual_review_burden.png")


def plot_review_burden_lab(df: pd.DataFrame, outdir: Path) -> None:
    plot_df = df.sort_values("pipeline_needs_review_rate")
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    ax.barh(plot_df["image_id"], plot_df["pipeline_needs_review_rate"] * 100, color="#2f6f9f")
    ax.set_xlabel("Fibers flagged for review (%)")
    ax.set_ylabel("")
    ax.set_title("Review Queue Is Focused, Not Full Manual Recounting")
    ax.grid(True, axis="x", color="#e5e5e5", linewidth=0.8)
    ax.tick_params(axis="y", labelsize=7)
    median = plot_df["pipeline_needs_review_rate"].median() * 100
    ax.axvline(median, color="#222222", linewidth=1.2)
    ax.text(median + 0.5, 0.5, f"median {median:.1f}%", va="center", fontsize=8)
    save_fig(outdir / "lab_manual_review_burden.png")


def plot_csa_boundary_effect_lab(df: pd.DataFrame, outdir: Path) -> None:
    ratio_cols = [
        ("Full label", "area_median_ratio_pipeline_to_myosight"),
        ("Erode 3 px", "area_erode_3px_median_ratio_pipeline_to_myosight"),
        ("Erode 4 px", "area_erode_4px_median_ratio_pipeline_to_myosight"),
        ("Erode 5 px", "area_erode_5px_median_ratio_pipeline_to_myosight"),
    ]
    rows = []
    for label, col in ratio_cols:
        if col not in df.columns:
            continue
        for _, row in df.iterrows():
            rows.append(
                {
                    "measurement": label,
                    "ratio": row[col],
                    "validation_input_kind": row["validation_input_kind"],
                }
            )
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    sns.boxplot(
        data=plot_df,
        x="measurement",
        y="ratio",
        color="#8fb3c9",
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="measurement",
        y="ratio",
        color="#2f6f9f",
        alpha=0.45,
        size=4,
        jitter=0.22,
        ax=ax,
    )
    ax.axhline(1.0, color="#333333", linewidth=1.1)
    ax.set_xlabel("")
    ax.set_ylabel("Median CSA ratio vs MyoSight")
    ax.set_title("CSA Depends on Measurement Boundary")
    ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)
    save_fig(outdir / "lab_csa_boundary_effect.png")


def plot_csa_agreement_lab(df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.scatter(
        df["myosight_area_median"],
        df["pipeline_area_erode_5px_median"],
        s=78,
        color="#2f6f9f",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
    )
    max_area = max(df["myosight_area_median"].max(), df["pipeline_area_erode_5px_median"].max())
    max_area *= 1.05
    ax.plot([0, max_area], [0, max_area], color="#333333", linewidth=1.1)
    ax.set_xlim(0, max_area)
    ax.set_ylim(0, max_area)
    ax.set_xlabel("MyoSight median CSA (um2)")
    ax.set_ylabel("Pipeline eroded median CSA (um2)")
    ax.set_title("Eroded CSA Is More MyoSight-Comparable")
    ax.grid(True, color="#e5e5e5", linewidth=0.8)
    save_fig(outdir / "lab_csa_eroded_agreement.png")


def write_summary(df: pd.DataFrame, outdir: Path) -> None:
    rows = []
    groups = {
        "all": df,
        "direct_czi": df[df["validation_input_kind"].eq("direct_czi")],
        "direct_czi_heldout": df[
            df["validation_input_kind"].eq("direct_czi")
            & df["training_or_heldout"].eq("heldout")
        ],
        "section_series_export": df[df["validation_input_kind"].eq("section_series_export")],
    }
    for name, sub in groups.items():
        rows.append(
            {
                "group": name,
                "n": len(sub),
                "median_abs_count_pct_diff": sub["fiber_count_abs_pct_diff"].median(),
                "within_10pct_count": int((sub["fiber_count_abs_pct_diff"] <= 10).sum()),
                "within_20pct_count": int((sub["fiber_count_abs_pct_diff"] <= 20).sum()),
                "median_review_rate_pct": sub["pipeline_needs_review_rate"].median() * 100,
                "median_signal_warning_rate_pct": sub[
                    "pipeline_signal_warning_rate"
                ].median()
                * 100,
            }
        )
    pd.DataFrame(rows).to_csv(outdir / "validation_figure_summary.csv", index=False)


def write_lab_summary(df: pd.DataFrame, outdir: Path) -> None:
    summary = {
        "n_images": len(df),
        "median_abs_count_pct_diff": df["fiber_count_abs_pct_diff"].median(),
        "within_10pct_count": int((df["fiber_count_abs_pct_diff"] <= 10).sum()),
        "within_20pct_count": int((df["fiber_count_abs_pct_diff"] <= 20).sum()),
        "median_review_rate_pct": df["pipeline_needs_review_rate"].median() * 100,
        "median_signal_warning_rate_pct": df["pipeline_signal_warning_rate"].median() * 100,
        "direct_czi_n": int(df["validation_input_kind"].eq("direct_czi").sum()),
        "section_export_n": int(df["validation_input_kind"].eq("section_series_export").sum()),
        "median_full_csa_ratio": df["area_median_ratio_pipeline_to_myosight"].median(),
        "median_erode_4px_csa_ratio": df[
            "area_erode_4px_median_ratio_pipeline_to_myosight"
        ].median(),
        "median_erode_5px_csa_ratio": df[
            "area_erode_5px_median_ratio_pipeline_to_myosight"
        ].median(),
    }
    pd.DataFrame([summary]).to_csv(outdir / "lab_deck_summary.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Make validation summary figures.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/validation/myosight_validation_consolidated_image_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation/figures"),
    )
    parser.add_argument(
        "--lab-output-dir",
        type=Path,
        default=Path("outputs/validation/figures_lab_deck"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
    df = prep_table(args.input)
    df["input_label"] = df["validation_input_kind"].map(INPUT_LABELS).fillna(
        df["validation_input_kind"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_count_scatter(df, args.output_dir)
    plot_count_error(df, args.output_dir)
    plot_type_scatter(df, args.output_dir)
    plot_type_differences(df, args.output_dir)
    plot_review_burden(df, args.output_dir)
    write_summary(df, args.output_dir)
    args.lab_output_dir.mkdir(parents=True, exist_ok=True)
    plot_count_scatter_lab(df, args.lab_output_dir)
    plot_count_error_lab(df, args.lab_output_dir)
    plot_type_scatter_lab(df, args.lab_output_dir)
    plot_type_profile_lab(df, args.lab_output_dir)
    plot_review_burden_lab(df, args.lab_output_dir)
    plot_csa_boundary_effect_lab(df, args.lab_output_dir)
    plot_csa_agreement_lab(df, args.lab_output_dir)
    write_lab_summary(df, args.lab_output_dir)
    print(f"saved figures to {args.output_dir}")
    print(f"saved lab deck figures to {args.lab_output_dir}")


if __name__ == "__main__":
    main()
