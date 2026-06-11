from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.plot_biological_story import AGE_ORDER, GENOTYPE_ORDER, add_biology_metadata
from src.plot_validation_summary import prep_table

DEFAULT_METHOD_SPECS = [
    (
        "Frozen",
        Path("outputs/validation/myosight_validation_consolidated_image_summary.csv"),
    ),
    (
        "Candidate",
        Path("outputs/validation/myosight_candidate_baseline_gb_soft_iia_q001_hybrid_image_summary.csv"),
    ),
]

METHOD_COLORS = {
    "MyoSight": "#6f6f6f",
    "Frozen": "#355f8c",
    "Candidate": "#c06a2b",
    "Model": "#3e8f63",
}
TYPE_ORDER = ["iib", "iia", "iix"]
TYPE_LABELS = {"iib": "IIb", "iia": "IIa", "iix": "IIx"}


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def parse_method_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"Invalid method spec {spec!r}. Use LABEL=/path/to/summary.csv"
        )
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise argparse.ArgumentTypeError(
            f"Invalid method spec {spec!r}. Use LABEL=/path/to/summary.csv"
        )
    return label, Path(raw_path)


def method_palette(methods: list[str]) -> dict[str, str]:
    fallback = ["#355f8c", "#c06a2b", "#3e8f63", "#8d4f8b", "#7a7a2a", "#2f8f8f"]
    palette: dict[str, str] = {}
    fallback_iter = iter(fallback)
    for method in methods:
        palette[method] = METHOD_COLORS.get(method, next(fallback_iter, "#555555"))
    return palette


def load_pipeline_summary(path: Path, method_name: str) -> pd.DataFrame:
    df = prep_table(path)
    df = add_biology_metadata(df)
    df["method_name"] = method_name
    return df


def build_method_long(summaries: list[pd.DataFrame]) -> pd.DataFrame:
    metric_specs = [
        ("total_fibers", "Total fibers", "count", "total_fibers"),
        ("area_median", "Median CSA", "um2", "area_median"),
    ]
    rows: list[dict[str, object]] = []
    seen_myo_ids: set[str] = set()
    for df in summaries:
        method_name = str(df["method_name"].iloc[0])
        for _, row in df.iterrows():
            image_id = str(row["image_id"])
            base = {
                "image_id": image_id,
                "age": row["age"],
                "age_month": row["age_month"],
                "genotype": row["genotype"],
                "training_or_heldout": row.get("training_or_heldout", "unknown"),
                "validation_input_kind": row.get("validation_input_kind", "unknown"),
            }
            if image_id not in seen_myo_ids:
                for metric, metric_label, unit, suffix in metric_specs:
                    col = f"myosight_{suffix}"
                    if col in row.index and not pd.isna(row[col]):
                        rows.append(
                            {
                                **base,
                                "method": "MyoSight",
                                "metric": metric,
                                "metric_label": metric_label,
                                "fiber_type": "all",
                                "value": row[col],
                                "unit": unit,
                            }
                        )
                for fiber_type in TYPE_ORDER:
                    rows.append(
                        {
                            **base,
                            "method": "MyoSight",
                            "metric": "fiber_count_by_type",
                            "metric_label": f"{TYPE_LABELS[fiber_type]} fibers",
                            "fiber_type": fiber_type,
                            "value": row[f"myosight_{fiber_type}_n"],
                            "unit": "count",
                        }
                    )
                    rows.append(
                        {
                            **base,
                            "method": "MyoSight",
                            "metric": "fiber_pct_by_type",
                            "metric_label": f"{TYPE_LABELS[fiber_type]} proportion",
                            "fiber_type": fiber_type,
                            "value": row[f"myosight_{fiber_type}_pct"] * 100.0,
                            "unit": "percent",
                        }
                    )
                seen_myo_ids.add(image_id)
            for metric, metric_label, unit, suffix in metric_specs:
                col = f"pipeline_{suffix}"
                if col in row.index and not pd.isna(row[col]):
                    rows.append(
                        {
                            **base,
                            "method": method_name,
                            "metric": metric,
                            "metric_label": metric_label,
                            "fiber_type": "all",
                            "value": row[col],
                            "unit": unit,
                        }
                    )
            for fiber_type in TYPE_ORDER:
                rows.append(
                    {
                        **base,
                        "method": method_name,
                        "metric": "fiber_count_by_type",
                        "metric_label": f"{TYPE_LABELS[fiber_type]} fibers",
                        "fiber_type": fiber_type,
                        "value": row[f"pipeline_{fiber_type}_n"],
                        "unit": "count",
                    }
                )
                rows.append(
                    {
                        **base,
                        "method": method_name,
                        "metric": "fiber_pct_by_type",
                        "metric_label": f"{TYPE_LABELS[fiber_type]} proportion",
                        "fiber_type": fiber_type,
                        "value": row[f"pipeline_{fiber_type}_pct"] * 100.0,
                        "unit": "percent",
                    }
                )
    long = pd.DataFrame(rows)
    long["age"] = pd.Categorical(long["age"], categories=AGE_ORDER, ordered=True)
    long["genotype"] = pd.Categorical(long["genotype"], categories=GENOTYPE_ORDER, ordered=True)
    method_order = ["MyoSight", *[str(df["method_name"].iloc[0]) for df in summaries]]
    long["method"] = pd.Categorical(long["method"], categories=method_order, ordered=True)
    return long


def build_review_long(summaries: list[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for df in summaries:
        method_name = str(df["method_name"].iloc[0])
        for _, row in df.iterrows():
            total = pd.to_numeric(row.get("pipeline_total_fibers"), errors="coerce")
            if pd.isna(total) or total <= 0:
                continue
            for metric, n_col, rate_col in [
                ("Needs review", "pipeline_needs_review_n", "pipeline_needs_review_rate"),
                ("Signal warnings", "pipeline_signal_warning_n", "pipeline_signal_warning_rate"),
            ]:
                rate = row.get(rate_col)
                if pd.isna(rate):
                    n_value = pd.to_numeric(row.get(n_col), errors="coerce")
                    rate = n_value / total if pd.notna(n_value) else pd.NA
                rows.append(
                    {
                        "image_id": row["image_id"],
                        "age": row["age"],
                        "age_month": row["age_month"],
                        "genotype": row["genotype"],
                        "method": method_name,
                        "metric": metric,
                        "count": pd.to_numeric(row.get(n_col), errors="coerce"),
                        "rate_pct": pd.to_numeric(rate, errors="coerce") * 100.0,
                    }
                )
    review = pd.DataFrame(rows)
    if not review.empty:
        review["age"] = pd.Categorical(review["age"], categories=AGE_ORDER, ordered=True)
        review["genotype"] = pd.Categorical(
            review["genotype"], categories=GENOTYPE_ORDER, ordered=True
        )
    return review


def write_group_summary(long: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    summary = (
        long.groupby(
            ["method", "age", "age_month", "genotype", "metric", "metric_label", "fiber_type", "unit"]
        )
        .agg(n_images=("value", "size"), mean=("value", "mean"), sem=("value", "sem"))
        .reset_index()
    )
    summary.to_csv(outdir / "method_group_summary.csv", index=False)
    return summary


def write_review_summary(review: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    summary = (
        review.groupby(["method", "metric", "age", "age_month", "genotype"])
        .agg(
            n_images=("image_id", "nunique"),
            mean_count=("count", "mean"),
            mean_rate_pct=("rate_pct", "mean"),
            median_rate_pct=("rate_pct", "median"),
        )
        .reset_index()
    )
    summary.to_csv(outdir / "review_signal_group_summary.csv", index=False)
    return summary


def plot_metric_trend(
    long: pd.DataFrame,
    metric: str,
    ylabel: str,
    outpath: Path,
    sharey: bool = True,
) -> None:
    plot_df = long[long["metric"].eq(metric)].copy()
    if plot_df.empty:
        return
    methods = [str(v) for v in plot_df["method"].cat.categories if v in set(plot_df["method"].astype(str))]
    grid = sns.relplot(
        data=plot_df,
        x="age",
        y="value",
        hue="method",
        style="method",
        row="genotype",
        kind="line",
        marker="o",
        errorbar="se",
        palette=method_palette(methods),
        height=3.6,
        aspect=1.2,
        facet_kws={"sharey": sharey},
    )
    grid.set_axis_labels("", ylabel)
    grid.set_titles("{row_name}")
    for ax in grid.axes.flat:
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
    save_fig(outpath)


def plot_type_trend(
    long: pd.DataFrame,
    metric: str,
    ylabel: str,
    outpath: Path,
) -> None:
    plot_df = long[long["metric"].eq(metric)].copy()
    if plot_df.empty:
        return
    plot_df["metric_label"] = pd.Categorical(
        plot_df["metric_label"],
        categories=[f"{TYPE_LABELS[t]} {'fibers' if metric == 'fiber_count_by_type' else 'proportion'}" for t in TYPE_ORDER],
        ordered=True,
    )
    methods = [str(v) for v in plot_df["method"].cat.categories if v in set(plot_df["method"].astype(str))]
    grid = sns.relplot(
        data=plot_df,
        x="age",
        y="value",
        hue="method",
        style="method",
        row="genotype",
        col="metric_label",
        kind="line",
        marker="o",
        errorbar="se",
        palette=method_palette(methods),
        height=3.5,
        aspect=1.0,
        facet_kws={"sharey": metric == "fiber_pct_by_type"},
    )
    grid.set_axis_labels("", ylabel)
    grid.set_titles("{row_name} | {col_name}")
    for ax in grid.axes.flat:
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        if metric == "fiber_pct_by_type":
            ax.set_ylim(0, 100)
    save_fig(outpath)


def plot_group_bars(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    outpath: Path,
) -> None:
    plot_df = summary[summary["metric"].eq(metric)].copy()
    if plot_df.empty:
        return
    plot_df["group"] = plot_df["genotype"].astype(str) + " | " + plot_df["age"].astype(str)
    methods = list(dict.fromkeys(plot_df["method"].astype(str)))
    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    sns.barplot(
        data=plot_df,
        x="group",
        y="mean",
        hue="method",
        hue_order=methods,
        palette=method_palette(methods),
        errorbar=None,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False, title="")
    save_fig(outpath)


def plot_review_burden_by_image(review: pd.DataFrame, outdir: Path) -> None:
    plot_df = review.copy()
    if plot_df.empty:
        return
    for metric, filename in [
        ("Needs review", "review_burden_by_image.png"),
        ("Signal warnings", "signal_warning_by_image.png"),
    ]:
        sub = plot_df[plot_df["metric"].eq(metric)].copy()
        order = (
            sub.groupby("image_id")["rate_pct"].max().sort_values(ascending=False).index.tolist()
        )
        fig, ax = plt.subplots(figsize=(13.0, 7.0))
        sns.stripplot(
            data=sub,
            x="rate_pct",
            y="image_id",
            hue="method",
            order=order,
            dodge=True,
            size=7,
            alpha=0.9,
            palette=method_palette(sorted(sub["method"].astype(str).unique().tolist())),
            ax=ax,
        )
        ax.set_xlabel(f"{metric} rate (%)")
        ax.set_ylabel("")
        ax.grid(True, axis="x", color="#e5e5e5", linewidth=0.8)
        ax.legend(frameon=False, title="")
        save_fig(outdir / filename)


def plot_review_signal_trends(review: pd.DataFrame, outdir: Path) -> None:
    if review.empty:
        return
    methods = sorted(review["method"].astype(str).unique().tolist())
    grid = sns.relplot(
        data=review,
        x="age",
        y="rate_pct",
        hue="method",
        style="method",
        row="metric",
        col="genotype",
        kind="line",
        marker="o",
        errorbar="se",
        palette=method_palette(methods),
        height=3.4,
        aspect=1.05,
        facet_kws={"sharey": False},
    )
    grid.set_axis_labels("", "Rate (%)")
    grid.set_titles("{row_name} | {col_name}")
    for ax in grid.axes.flat:
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
    save_fig(outdir / "review_signal_trends_by_age_genotype.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot direct MyoSight vs pipeline method comparisons across age, genotype, and image."
    )
    parser.add_argument(
        "--method-summary",
        dest="method_summaries",
        action="append",
        type=parse_method_spec,
        help="Method summary in LABEL=path form. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation/figures_multimethod_myosight_compare"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    method_specs = args.method_summaries or DEFAULT_METHOD_SPECS
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.76)
    summaries = [load_pipeline_summary(path, label) for label, path in method_specs]
    common_ids = set(summaries[0]["image_id"].astype(str))
    for df in summaries[1:]:
        common_ids &= set(df["image_id"].astype(str))
    common_ids = sorted(common_ids)
    summaries = [
        df[df["image_id"].astype(str).isin(common_ids)].sort_values("image_id").reset_index(drop=True)
        for df in summaries
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    long = build_method_long(summaries)
    review = build_review_long(summaries)
    long.to_csv(args.output_dir / "method_long.csv", index=False)
    review.to_csv(args.output_dir / "review_signal_long.csv", index=False)
    group_summary = write_group_summary(long, args.output_dir)
    write_review_summary(review, args.output_dir)

    plot_metric_trend(
        long,
        metric="total_fibers",
        ylabel="Mean fibers per TA section",
        outpath=args.output_dir / "total_fibers_trend.png",
        sharey=True,
    )
    plot_metric_trend(
        long,
        metric="area_median",
        ylabel="Median CSA (um2)",
        outpath=args.output_dir / "median_csa_trend.png",
        sharey=False,
    )
    plot_type_trend(
        long,
        metric="fiber_count_by_type",
        ylabel="Mean fiber count",
        outpath=args.output_dir / "fiber_type_count_trends.png",
    )
    plot_type_trend(
        long,
        metric="fiber_pct_by_type",
        ylabel="Mean fiber proportion (%)",
        outpath=args.output_dir / "fiber_type_proportion_trends.png",
    )
    plot_group_bars(
        group_summary,
        metric="total_fibers",
        ylabel="Mean fibers per TA section",
        outpath=args.output_dir / "total_fibers_group_bars.png",
    )
    plot_group_bars(
        group_summary,
        metric="area_median",
        ylabel="Median CSA (um2)",
        outpath=args.output_dir / "median_csa_group_bars.png",
    )
    plot_review_signal_trends(review, args.output_dir)
    plot_review_burden_by_image(review, args.output_dir)
    print(f"saved multi-method comparison figures to {args.output_dir}")


if __name__ == "__main__":
    main()
