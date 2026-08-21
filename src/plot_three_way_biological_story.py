from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.biology_metadata import (
    AGE_ORDER,
    GENOTYPE_COLORS,
    GENOTYPE_ORDER,
    TYPE_COLORS,
    add_biology_metadata,
)

METHOD_ORDER = ["MyoSight", "Frozen", "Candidate"]
METHOD_COLORS = {
    "MyoSight": "#777777",
    "Frozen": "#7d8ea3",
    "Candidate": "#2f6f9f",
}


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def _long_rows_for_method(row: pd.Series, method: str, prefix: str) -> list[dict[str, object]]:
    metric_specs = [
        ("total_fibers", "Total fibers", "count", "total_fibers"),
        ("iib_n", "IIb fibers", "count", "iib_n"),
        ("iia_n", "IIa fibers", "count", "iia_n"),
        ("iix_n", "IIx fibers", "count", "iix_n"),
        ("iib_pct", "IIb proportion", "percent", "iib_pct"),
        ("iia_pct", "IIa proportion", "percent", "iia_pct"),
        ("iix_pct", "IIx proportion", "percent", "iix_pct"),
        ("area_median", "Median CSA", "um2", "area_median"),
    ]
    out: list[dict[str, object]] = []
    for metric, label, unit, suffix in metric_specs:
        col = f"{prefix}_{suffix}"
        if col not in row.index or pd.isna(row[col]):
            continue
        value = row[col]
        if unit == "percent":
            value *= 100
        out.append(
            {
                "image_id": row["image_id"],
                "age_month": row["age_month"],
                "age": row["age"],
                "genotype": row["genotype"],
                "method": method,
                "metric": metric,
                "metric_label": label,
                "unit": unit,
                "value": value,
            }
        )
    return out


def build_long_table(frozen: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    frozen = add_biology_metadata(frozen)
    candidate = add_biology_metadata(candidate)
    common = sorted(set(frozen["image_id"]) & set(candidate["image_id"]))
    frozen = frozen.set_index("image_id").loc[common].reset_index()
    candidate = candidate.set_index("image_id").loc[common].reset_index()
    rows: list[dict[str, object]] = []
    for _, frow in frozen.iterrows():
        rows.extend(_long_rows_for_method(frow, "MyoSight", "myosight"))
        rows.extend(_long_rows_for_method(frow, "Frozen", "pipeline"))
    for _, crow in candidate.iterrows():
        rows.extend(_long_rows_for_method(crow, "Candidate", "pipeline"))
    out = pd.DataFrame(rows)
    out["age"] = pd.Categorical(out["age"], categories=AGE_ORDER, ordered=True)
    out["genotype"] = pd.Categorical(out["genotype"], categories=GENOTYPE_ORDER, ordered=True)
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    return out


def write_group_summary(long: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    group = (
        long.groupby(["method", "age", "age_month", "genotype", "metric", "metric_label", "unit"])
        .agg(n_images=("value", "size"), mean=("value", "mean"), sem=("value", "sem"))
        .reset_index()
    )
    group.to_csv(outdir / "three_way_biological_story_group_summary.csv", index=False)
    return group


def write_directional_checks(group: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        for metric in ["total_fibers", "iib_n", "iia_n", "iix_n"]:
            for age_month in [1, 4, 12]:
                sub = group[
                    group["method"].eq(method)
                    & group["metric"].eq(metric)
                    & group["age_month"].eq(age_month)
                ].set_index("genotype")
                if not {"mdx5cv", "mdx5cv-JAG1"}.issubset(sub.index):
                    continue
                mdx = sub.loc["mdx5cv", "mean"]
                jag = sub.loc["mdx5cv-JAG1", "mean"]
                rows.append(
                    {
                        "method": method,
                        "metric": metric,
                        "age_month": age_month,
                        "mdx_mean": mdx,
                        "jag_mean": jag,
                        "jag_minus_mdx": jag - mdx,
                        "jag_vs_mdx_pct": ((jag - mdx) / mdx * 100) if mdx else pd.NA,
                    }
                )
    checks = pd.DataFrame(rows)
    checks.to_csv(outdir / "three_way_biological_story_directional_checks.csv", index=False)
    return checks


def plot_total_fiber_count(long: pd.DataFrame, outdir: Path) -> None:
    plot_df = long[long["metric"].eq("total_fibers")].copy()
    grid = sns.relplot(
        data=plot_df,
        x="age",
        y="value",
        hue="genotype",
        style="method",
        row="method",
        kind="line",
        marker="o",
        errorbar="se",
        palette=GENOTYPE_COLORS,
        height=3.2,
        aspect=1.25,
        facet_kws={"sharey": True},
    )
    grid.set_axis_labels("", "Mean fibers per TA section")
    grid.set_titles("{row_name}")
    for ax in grid.axes.flat:
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
    grid.figure.suptitle("Total Fiber Count Trend Across Methods", y=1.02)
    save_fig(outdir / "three_way_story_total_fiber_count.png")


def plot_type_proportions(long: pd.DataFrame, outdir: Path) -> None:
    plot_df = long[long["metric"].isin(["iib_pct", "iia_pct", "iix_pct"])].copy()
    grid = sns.relplot(
        data=plot_df,
        x="age",
        y="value",
        hue="metric_label",
        col="method",
        row="genotype",
        kind="line",
        marker="o",
        errorbar="se",
        palette={
            "IIb proportion": TYPE_COLORS["IIb"],
            "IIa proportion": TYPE_COLORS["IIa"],
            "IIx proportion": TYPE_COLORS["IIx"],
        },
        height=3.4,
        aspect=1.05,
        facet_kws={"sharey": True},
    )
    grid.set_axis_labels("", "Mean fiber type proportion (%)")
    grid.set_titles("{row_name} | {col_name}")
    for ax in grid.axes.flat:
        ax.set_ylim(0, 100)
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
    grid.figure.suptitle("Fiber Type Composition Trends", y=1.02)
    save_fig(outdir / "three_way_story_fiber_type_composition.png")


def plot_type_counts(long: pd.DataFrame, outdir: Path) -> None:
    plot_df = long[long["metric"].isin(["iib_n", "iia_n", "iix_n"])].copy()
    grid = sns.relplot(
        data=plot_df,
        x="age",
        y="value",
        hue="genotype",
        row="metric_label",
        col="method",
        kind="line",
        marker="o",
        errorbar="se",
        palette=GENOTYPE_COLORS,
        height=3.0,
        aspect=1.05,
        facet_kws={"sharey": False},
    )
    grid.set_axis_labels("", "Mean fiber count")
    grid.set_titles("{row_name} | {col_name}")
    for ax in grid.axes.flat:
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
    grid.figure.suptitle("Type-Specific Fiber Counts", y=1.01)
    save_fig(outdir / "three_way_story_type_specific_counts.png")


def plot_jag_effects(checks: pd.DataFrame, outdir: Path) -> None:
    plot_df = checks[
        checks["metric"].isin(["total_fibers", "iib_n", "iia_n", "iix_n"])
        & checks["age_month"].isin([1, 4, 12])
    ].copy()
    plot_df["metric"] = pd.Categorical(
        plot_df["metric"],
        categories=["total_fibers", "iib_n", "iia_n", "iix_n"],
        ordered=True,
    )
    grid = sns.catplot(
        data=plot_df,
        x="age_month",
        y="jag_vs_mdx_pct",
        hue="method",
        col="metric",
        kind="bar",
        palette=METHOD_COLORS,
        height=3.8,
        aspect=0.95,
        sharey=False,
    )
    grid.set_axis_labels("Age (months)", "JAG1 vs mdx difference (%)")
    grid.set_titles("{col_name}")
    for ax in grid.axes.flat:
        ax.axhline(0, color="#333333", linewidth=1.0)
        ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)
    grid.figure.suptitle("Does Each Method Tell the Same JAG1 Story?", y=1.05)
    save_fig(outdir / "three_way_story_jag1_effects.png")


def plot_csa(long: pd.DataFrame, outdir: Path) -> None:
    plot_df = long[long["metric"].eq("area_median")].copy()
    grid = sns.relplot(
        data=plot_df,
        x="age",
        y="value",
        hue="genotype",
        row="method",
        kind="line",
        marker="o",
        errorbar="se",
        palette=GENOTYPE_COLORS,
        height=3.2,
        aspect=1.25,
        facet_kws={"sharey": False},
    )
    grid.set_axis_labels("", "Median fiber CSA (um2)")
    grid.set_titles("{row_name}")
    for ax in grid.axes.flat:
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
    grid.figure.suptitle("CSA Trend Across Methods", y=1.02)
    save_fig(outdir / "three_way_story_csa.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot three-way biology story: MyoSight, Frozen, Candidate."
    )
    parser.add_argument(
        "--frozen-summary",
        type=Path,
        default=Path("outputs/validation/myosight_validation_consolidated_image_summary.csv"),
    )
    parser.add_argument(
        "--candidate-summary",
        type=Path,
        default=Path("outputs/validation/myosight_candidate_baseline_gb_soft_iia_q001_hybrid_image_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation/figures_three_way_biological_story"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.78)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen = pd.read_csv(args.frozen_summary)
    candidate = pd.read_csv(args.candidate_summary)
    long = build_long_table(frozen, candidate)
    long.to_csv(args.output_dir / "three_way_biological_story_long.csv", index=False)
    group = write_group_summary(long, args.output_dir)
    checks = write_directional_checks(group, args.output_dir)
    plot_total_fiber_count(long, args.output_dir)
    plot_type_proportions(long, args.output_dir)
    plot_type_counts(long, args.output_dir)
    plot_jag_effects(checks, args.output_dir)
    plot_csa(long, args.output_dir)
    print(f"saved three-way biological story to {args.output_dir}")


if __name__ == "__main__":
    main()
