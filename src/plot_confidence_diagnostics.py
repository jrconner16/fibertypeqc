from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.biology_metadata import (
    AGE_ORDER,
    GENOTYPE_ORDER,
    TYPE_COLORS,
    add_biology_metadata,
    remove_unused_categories,
)

TYPE_ORDER = ["iib", "iia", "iix"]
TYPE_LABELS = {"iib": "IIb", "iia": "IIa", "iix": "IIx"}
SUBSET_ORDER = [
    "MyoSight",
    "Pipeline all",
    "Pipeline soft",
    "Pipeline unflagged",
    "Pipeline clean high-conf",
    "Pipeline review-only",
]
SUBSET_COLORS = {
    "MyoSight": "#777777",
    "Pipeline all": "#2f6f9f",
    "Pipeline soft": "#6c71c4",
    "Pipeline unflagged": "#4c956c",
    "Pipeline clean high-conf": "#1b7f79",
    "Pipeline review-only": "#b45f06",
}
PROB_COLS = {"iib": "prob_iib", "iia": "prob_iia", "iix": "prob_iix"}


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].fillna(False).astype(bool)


def load_pipeline_fibers(summary: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, row in summary.iterrows():
        path = Path(row["pipeline_path"])
        if not path.exists():
            raise FileNotFoundError(path)
        fibers = pd.read_csv(path)
        fibers["image_id"] = row["image_id"]
        fibers["age"] = row["age"]
        fibers["age_month"] = row["age_month"]
        fibers["genotype"] = row["genotype"]
        frames.append(fibers)
    out = pd.concat(frames, ignore_index=True)
    out["fiber_type"] = out["fiber_type"].str.lower()
    out["needs_review"] = bool_col(out, "needs_review")
    out["has_signal_qc_warning"] = bool_col(out, "has_signal_qc_warning")
    out["model_confidence"] = pd.to_numeric(out["model_confidence"], errors="coerce")
    out["model_margin"] = pd.to_numeric(out["model_margin"], errors="coerce")
    for col in PROB_COLS.values():
        out[col] = pd.to_numeric(out[col], errors="coerce")
    probs = out[list(PROB_COLS.values())].clip(lower=0.0).fillna(0.0)
    prob_sum = probs.sum(axis=1).replace(0, np.nan)
    normalized = probs.div(prob_sum, axis=0).fillna(0.0)
    entropy = -(normalized * np.log(normalized.replace(0, np.nan))).sum(axis=1)
    out["model_entropy"] = entropy.fillna(0.0)
    out["model_entropy_norm"] = out["model_entropy"] / np.log(len(PROB_COLS))
    return out


def summarize_one_image(
    image_id: str,
    age: str,
    age_month: int,
    genotype: str,
    subset: str,
    labels: pd.Series,
) -> list[dict[str, object]]:
    labels = labels[labels.isin(TYPE_ORDER)]
    total = int(len(labels))
    rows = []
    for typ in TYPE_ORDER:
        count = int(labels.eq(typ).sum())
        rows.append(
            {
                "image_id": image_id,
                "age": age,
                "age_month": age_month,
                "genotype": genotype,
                "subset": subset,
                "fiber_type": TYPE_LABELS[typ],
                "count": count,
                "total": total,
                "proportion": count / total * 100 if total else pd.NA,
            }
        )
    return rows


def summarize_soft_image(
    image_id: str,
    age: str,
    age_month: int,
    genotype: str,
    fibers: pd.DataFrame,
) -> list[dict[str, object]]:
    total = int(len(fibers))
    rows = []
    for typ, col in PROB_COLS.items():
        count = float(fibers[col].sum())
        rows.append(
            {
                "image_id": image_id,
                "age": age,
                "age_month": age_month,
                "genotype": genotype,
                "subset": "Pipeline soft",
                "fiber_type": TYPE_LABELS[typ],
                "count": count,
                "total": total,
                "proportion": count / total * 100 if total else pd.NA,
            }
        )
    return rows


def build_subset_summary(
    image_summary: pd.DataFrame,
    fibers: pd.DataFrame,
    confidence_threshold: float,
    margin_threshold: float,
) -> pd.DataFrame:
    rows = []
    for _, row in image_summary.iterrows():
        image_id = row["image_id"]
        for typ in TYPE_ORDER:
            rows.append(
                {
                    "image_id": image_id,
                    "age": row["age"],
                    "age_month": row["age_month"],
                    "genotype": row["genotype"],
                    "subset": "MyoSight",
                    "fiber_type": TYPE_LABELS[typ],
                    "count": int(row[f"myosight_{typ}_n"]),
                    "total": int(row["myosight_total_fibers"]),
                    "proportion": row[f"myosight_{typ}_pct"] * 100,
                }
            )
        sub = fibers[fibers["image_id"].eq(image_id)]
        rows.extend(
            summarize_soft_image(
                image_id=image_id,
                age=row["age"],
                age_month=int(row["age_month"]),
                genotype=row["genotype"],
                fibers=sub,
            )
        )
        masks = {
            "Pipeline all": pd.Series(True, index=sub.index),
            "Pipeline unflagged": ~sub["needs_review"],
            "Pipeline review-only": sub["needs_review"],
            "Pipeline clean high-conf": (
                ~sub["needs_review"]
                & ~sub["has_signal_qc_warning"]
                & sub["model_confidence"].ge(confidence_threshold)
                & sub["model_margin"].ge(margin_threshold)
            ),
        }
        for subset, mask in masks.items():
            rows.extend(
                summarize_one_image(
                    image_id=image_id,
                    age=row["age"],
                    age_month=int(row["age_month"]),
                    genotype=row["genotype"],
                    subset=subset,
                    labels=sub.loc[mask, "fiber_type"],
                )
            )
    out = pd.DataFrame(rows)
    out["age"] = pd.Categorical(out["age"], categories=AGE_ORDER, ordered=True)
    out["genotype"] = pd.Categorical(out["genotype"], categories=GENOTYPE_ORDER, ordered=True)
    out["subset"] = pd.Categorical(out["subset"], categories=SUBSET_ORDER, ordered=True)
    out["fiber_type"] = pd.Categorical(
        out["fiber_type"],
        categories=[TYPE_LABELS[t] for t in TYPE_ORDER],
        ordered=True,
    )
    return out


def group_subset_summary(subsets: pd.DataFrame) -> pd.DataFrame:
    return (
        subsets.groupby(["subset", "age", "age_month", "genotype", "fiber_type"])
        .agg(
            n_images=("image_id", "nunique"),
            mean_count=("count", "mean"),
            mean_proportion=("proportion", "mean"),
            sem_proportion=("proportion", "sem"),
            mean_total=("total", "mean"),
        )
        .reset_index()
    )


def genotype_effects(grouped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in grouped.groupby(["subset", "age", "age_month", "fiber_type"]):
        subset, age, age_month, fiber_type = keys
        values = sub.set_index("genotype")
        if not {"mdx5cv", "mdx5cv-JAG1"}.issubset(values.index):
            continue
        mdx_prop = values.loc["mdx5cv", "mean_proportion"]
        jag_prop = values.loc["mdx5cv-JAG1", "mean_proportion"]
        mdx_count = values.loc["mdx5cv", "mean_count"]
        jag_count = values.loc["mdx5cv-JAG1", "mean_count"]
        rows.append(
            {
                "subset": subset,
                "age": age,
                "age_month": age_month,
                "fiber_type": fiber_type,
                "mdx5cv_mean_proportion": mdx_prop,
                "mdx5cv_jag1_mean_proportion": jag_prop,
                "jag1_minus_mdx_percentage_points": jag_prop - mdx_prop,
                "mdx5cv_mean_count": mdx_count,
                "mdx5cv_jag1_mean_count": jag_count,
                "jag1_vs_mdx_count_pct": (jag_count - mdx_count) / mdx_count * 100
                if mdx_count
                else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def plot_subset_type_proportions(grouped: pd.DataFrame, outdir: Path) -> None:
    plot_df = grouped[
        grouped["subset"].isin(
            [
                "MyoSight",
                "Pipeline all",
                "Pipeline soft",
                "Pipeline unflagged",
                "Pipeline clean high-conf",
            ]
        )
    ].copy()
    plot_df = remove_unused_categories(plot_df)
    grid = sns.relplot(
        data=plot_df,
        x="age",
        y="mean_proportion",
        hue="fiber_type",
        row="genotype",
        col="subset",
        kind="line",
        marker="o",
        palette=TYPE_COLORS,
        height=3.4,
        aspect=1.0,
        facet_kws={"sharey": True},
    )
    grid.set_axis_labels("", "Mean fiber type proportion (%)")
    grid.set_titles("{row_name} | {col_name}")
    for ax in grid.axes.flat:
        ax.set_ylim(0, 100)
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
    grid.figure.suptitle("Fiber Type Proportions After Confidence/Flag Filtering", y=1.03)
    save_fig(outdir / "confidence_filtered_fiber_type_proportions.png")


def plot_jag1_effects(effects: pd.DataFrame, outdir: Path) -> None:
    plot_df = effects[
        effects["fiber_type"].isin(["IIb", "IIx"])
        & effects["subset"].isin(
            [
                "MyoSight",
                "Pipeline all",
                "Pipeline soft",
                "Pipeline unflagged",
                "Pipeline clean high-conf",
            ]
        )
    ].copy()
    plot_df = remove_unused_categories(plot_df)
    grid = sns.catplot(
        data=plot_df,
        x="age",
        y="jag1_minus_mdx_percentage_points",
        hue="subset",
        col="fiber_type",
        kind="bar",
        palette=SUBSET_COLORS,
        height=4.2,
        aspect=1.0,
        sharey=False,
    )
    grid.set_axis_labels("", "JAG1 minus mdx (percentage points)")
    grid.set_titles("{col_name} proportion effect")
    for ax in grid.axes.flat:
        ax.axhline(0, color="#333333", linewidth=1.0)
        ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)
    grid.figure.suptitle("Does Filtering Change the JAG1 Fiber-Type Effect?", y=1.05)
    save_fig(outdir / "confidence_jag1_effect_by_filter.png")


def plot_hard_soft_jag1_effect(effects: pd.DataFrame, outdir: Path) -> None:
    plot_df = effects[
        effects["fiber_type"].isin(["IIb", "IIa", "IIx"])
        & effects["subset"].isin(["MyoSight", "Pipeline all", "Pipeline soft"])
    ].copy()
    plot_df = remove_unused_categories(plot_df)
    grid = sns.catplot(
        data=plot_df,
        x="age",
        y="jag1_minus_mdx_percentage_points",
        hue="subset",
        col="fiber_type",
        kind="bar",
        palette=SUBSET_COLORS,
        height=4.1,
        aspect=0.92,
        sharey=False,
    )
    grid.set_axis_labels("", "JAG1 minus mdx (percentage points)")
    grid.set_titles("{col_name} proportion effect")
    for ax in grid.axes.flat:
        ax.axhline(0, color="#333333", linewidth=1.0)
        ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)
    grid.figure.suptitle("Hard Calls vs Probability-Weighted Calls", y=1.05)
    save_fig(outdir / "confidence_hard_vs_soft_jag1_effect.png")


def plot_flag_composition(subsets: pd.DataFrame, outdir: Path) -> None:
    plot_df = subsets[subsets["subset"].isin(["Pipeline all", "Pipeline review-only"])].copy()
    grouped = group_subset_summary(plot_df)
    grouped = remove_unused_categories(grouped)
    grid = sns.catplot(
        data=grouped,
        x="fiber_type",
        y="mean_proportion",
        hue="subset",
        row="genotype",
        col="age",
        kind="bar",
        palette=SUBSET_COLORS,
        height=3.2,
        aspect=0.95,
        sharey=True,
    )
    grid.set_axis_labels("", "Mean within-subset proportion (%)")
    grid.set_titles("{row_name} | {col_name}")
    for ax in grid.axes.flat:
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)
    grid.figure.suptitle("Where Review-Flagged Fibers Sit by Type", y=1.03)
    save_fig(outdir / "confidence_review_flag_composition.png")


def plot_confidence_margin(fibers: pd.DataFrame, outdir: Path) -> None:
    plot_df = fibers[fibers["fiber_type"].isin(TYPE_ORDER)].copy()
    plot_df["fiber_type_label"] = plot_df["fiber_type"].map(TYPE_LABELS)
    plot_df["review_status"] = plot_df["needs_review"].map(
        {True: "Needs review", False: "Not flagged"}
    )
    plot_df["fiber_type_label"] = pd.Categorical(
        plot_df["fiber_type_label"],
        categories=[TYPE_LABELS[t] for t in TYPE_ORDER],
        ordered=True,
    )
    plot_df = remove_unused_categories(plot_df)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True)
    for ax, metric, ylabel in [
        (axes[0], "model_confidence", "Model confidence"),
        (axes[1], "model_margin", "Model margin"),
    ]:
        sns.boxplot(
            data=plot_df,
            x="fiber_type_label",
            y=metric,
            hue="review_status",
            showfliers=False,
            palette={"Not flagged": "#2f6f9f", "Needs review": "#b45f06"},
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)
        ax.legend(frameon=False, title="")
    fig.suptitle("Confidence and Margin by Pipeline Call")
    save_fig(outdir / "confidence_margin_by_type_and_flag.png")


def entropy_image_summary(fibers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in fibers.groupby(["image_id", "age", "age_month", "genotype"]):
        image_id, age, age_month, genotype = keys
        rows.append(
            {
                "image_id": image_id,
                "age": age,
                "age_month": age_month,
                "genotype": genotype,
                "n_fibers": int(len(sub)),
                "entropy_mean": float(sub["model_entropy_norm"].mean()),
                "entropy_median": float(sub["model_entropy_norm"].median()),
                "high_entropy_rate": float(sub["model_entropy_norm"].ge(0.60).mean() * 100),
                "low_margin_rate": float(sub["model_margin"].lt(0.15).mean() * 100),
                "low_confidence_rate": float(sub["model_confidence"].lt(0.55).mean() * 100),
                "needs_review_rate": float(sub["needs_review"].mean() * 100),
            }
        )
    out = pd.DataFrame(rows)
    out["age"] = pd.Categorical(out["age"], categories=AGE_ORDER, ordered=True)
    out["genotype"] = pd.Categorical(out["genotype"], categories=GENOTYPE_ORDER, ordered=True)
    return out


def entropy_group_summary(image_entropy: pd.DataFrame) -> pd.DataFrame:
    return (
        image_entropy.groupby(["age", "age_month", "genotype"])
        .agg(
            n_images=("image_id", "nunique"),
            entropy_mean=("entropy_mean", "mean"),
            entropy_median=("entropy_median", "mean"),
            high_entropy_rate=("high_entropy_rate", "mean"),
            low_margin_rate=("low_margin_rate", "mean"),
            low_confidence_rate=("low_confidence_rate", "mean"),
            needs_review_rate=("needs_review_rate", "mean"),
        )
        .reset_index()
    )


def plot_entropy_by_group(image_entropy: pd.DataFrame, outdir: Path) -> None:
    plot_df = remove_unused_categories(image_entropy)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=True)
    sns.lineplot(
        data=plot_df,
        x="age",
        y="entropy_mean",
        hue="genotype",
        marker="o",
        errorbar="se",
        ax=axes[0],
    )
    axes[0].set_ylabel("Mean normalized entropy")
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, color="#e5e5e5", linewidth=0.8)
    axes[0].legend(frameon=False, title="")
    sns.lineplot(
        data=plot_df,
        x="age",
        y="high_entropy_rate",
        hue="genotype",
        marker="o",
        errorbar="se",
        ax=axes[1],
    )
    axes[1].set_ylabel("High-entropy fibers (%)")
    axes[1].set_xlabel("")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, color="#e5e5e5", linewidth=0.8)
    axes[1].legend(frameon=False, title="")
    fig.suptitle("Model Uncertainty by Age and Genotype")
    save_fig(outdir / "confidence_entropy_by_age_genotype.png")


def plot_entropy_by_type(fibers: pd.DataFrame, outdir: Path) -> None:
    plot_df = fibers[fibers["fiber_type"].isin(TYPE_ORDER)].copy()
    plot_df["fiber_type_label"] = plot_df["fiber_type"].map(TYPE_LABELS)
    plot_df["fiber_type_label"] = pd.Categorical(
        plot_df["fiber_type_label"],
        categories=[TYPE_LABELS[t] for t in TYPE_ORDER],
        ordered=True,
    )
    plot_df = remove_unused_categories(plot_df)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sns.boxplot(
        data=plot_df,
        x="fiber_type_label",
        y="model_entropy_norm",
        hue="needs_review",
        showfliers=False,
        palette={False: "#2f6f9f", True: "#b45f06"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Normalized entropy")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", color="#e5e5e5", linewidth=0.8)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Not flagged", "Needs review"], frameon=False, title="")
    ax.set_title("Entropy by Pipeline Call")
    save_fig(outdir / "confidence_entropy_by_type_and_flag.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot confidence/flag diagnostics for validation fiber typing."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/validation/myosight_validation_consolidated_image_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation/figures_lab_deck"),
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--margin-threshold", type=float, default=0.40)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.72)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = add_biology_metadata(pd.read_csv(args.summary))
    fibers = load_pipeline_fibers(summary)
    subsets = build_subset_summary(
        summary,
        fibers,
        confidence_threshold=args.confidence_threshold,
        margin_threshold=args.margin_threshold,
    )
    grouped = group_subset_summary(subsets)
    effects = genotype_effects(grouped)
    image_entropy = entropy_image_summary(fibers)
    group_entropy = entropy_group_summary(image_entropy)
    subsets.to_csv(args.output_dir / "confidence_subset_image_summary.csv", index=False)
    grouped.to_csv(args.output_dir / "confidence_subset_group_summary.csv", index=False)
    effects.to_csv(args.output_dir / "confidence_subset_genotype_effects.csv", index=False)
    image_entropy.to_csv(args.output_dir / "confidence_entropy_image_summary.csv", index=False)
    group_entropy.to_csv(args.output_dir / "confidence_entropy_group_summary.csv", index=False)
    plot_subset_type_proportions(grouped, args.output_dir)
    plot_jag1_effects(effects, args.output_dir)
    plot_hard_soft_jag1_effect(effects, args.output_dir)
    plot_flag_composition(subsets, args.output_dir)
    plot_confidence_margin(fibers, args.output_dir)
    plot_entropy_by_group(image_entropy, args.output_dir)
    plot_entropy_by_type(fibers, args.output_dir)
    print(f"saved confidence diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
