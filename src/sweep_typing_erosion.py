from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tifffile

from src.io_utils import extract_pixel_size_um, load_multichannel_image
from src.plot_biological_story import (
    AGE_ORDER,
    GENOTYPE_ORDER,
    TYPE_COLORS,
    add_biology_metadata,
    remove_unused_categories,
)
from src.quantify_classify import QuantifyConfig, quantify_labels
from src.run_batch import V0_PARAMS

TYPE_ORDER = ("iib", "iia", "iix")
TYPE_LABELS = {"iib": "IIb", "iia": "IIa", "iix": "IIx"}


def parse_erodes(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def find_labels_path(pipeline_path: Path) -> Path:
    matches = sorted(pipeline_path.parent.glob("*_cellpose_labels.tif"))
    if not matches:
        raise FileNotFoundError(f"No label TIFF found next to {pipeline_path}")
    if len(matches) > 1:
        raise ValueError(f"Multiple label TIFFs found next to {pipeline_path}: {matches}")
    return matches[0]


def count_types(fibers: pd.DataFrame) -> dict[str, int]:
    labels = fibers["fiber_type"].astype(str).str.lower()
    counts = labels.value_counts().to_dict()
    return {typ: int(counts.get(typ, 0)) for typ in TYPE_ORDER}


def summarize_fibers(
    row: pd.Series,
    erode_px: int,
    fibers: pd.DataFrame,
) -> dict[str, object]:
    total = int(len(fibers))
    counts = count_types(fibers)
    out: dict[str, object] = {
        "image_id": row["image_id"],
        "age": row["age"],
        "age_month": int(row["age_month"]),
        "genotype": row["genotype"],
        "typing_erode_px": int(erode_px),
        "pipeline_total_fibers": total,
        "needs_review_n": int(fibers["needs_review"].fillna(False).astype(bool).sum()),
        "signal_warning_n": int(
            fibers["has_signal_qc_warning"].fillna(False).astype(bool).sum()
        ),
        "model_confidence_median": float(fibers["model_confidence"].median()),
        "model_margin_median": float(fibers["model_margin"].median()),
    }
    for typ in TYPE_ORDER:
        out[f"pipeline_{typ}_n"] = counts[typ]
        out[f"pipeline_{typ}_pct"] = counts[typ] / total * 100 if total else pd.NA
        out[f"myosight_{typ}_pct"] = float(row[f"myosight_{typ}_pct"]) * 100
        out[f"{typ}_pct_diff_vs_myosight"] = out[f"pipeline_{typ}_pct"] - out[
            f"myosight_{typ}_pct"
        ]
    out["needs_review_rate"] = out["needs_review_n"] / total * 100 if total else pd.NA
    out["signal_warning_rate"] = out["signal_warning_n"] / total * 100 if total else pd.NA
    return out


def sweep_image(row: pd.Series, erodes: list[int]) -> list[dict[str, object]]:
    image_path = Path(row["source_image"])
    labels_path = find_labels_path(Path(row["pipeline_path"]))
    print(f"{row['image_id']}: loading image + labels", flush=True)
    image = load_multichannel_image(image_path)
    labels = tifffile.imread(labels_path)
    pixel_size_x_um, pixel_size_y_um = extract_pixel_size_um(image_path)
    rows = []
    for erode_px in erodes:
        print(f"  typing_erode_px={erode_px}", flush=True)
        cfg = QuantifyConfig(
            type1_channel=V0_PARAMS["type1_channel"],
            type2_channel=V0_PARAMS["type2_channel"],
            typing_preprocess=V0_PARAMS["typing_preprocess"],
            typing_tile_size=V0_PARAMS["typing_tile_size"],
            typing_erode_px=erode_px,
            model_confidence_threshold=V0_PARAMS["model_confidence_threshold"],
            model_margin_threshold=V0_PARAMS["model_margin_threshold"],
            pixel_size_x_um=pixel_size_x_um,
            pixel_size_y_um=pixel_size_y_um,
            csa_erode_px=(),
            classifier_path=str(Path(V0_PARAMS["classifier_path"]).resolve()),
        )
        fibers = quantify_labels(labels, image, cfg)
        rows.append(summarize_fibers(row, erode_px, fibers))
    return rows


def group_summary(image_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = []
    for _, row in image_summary.iterrows():
        for typ in TYPE_ORDER:
            metrics.append(
                {
                    "image_id": row["image_id"],
                    "age": row["age"],
                    "age_month": row["age_month"],
                    "genotype": row["genotype"],
                    "typing_erode_px": row["typing_erode_px"],
                    "fiber_type": TYPE_LABELS[typ],
                    "pipeline_pct": row[f"pipeline_{typ}_pct"],
                    "myosight_pct": row[f"myosight_{typ}_pct"],
                    "pct_diff_vs_myosight": row[f"{typ}_pct_diff_vs_myosight"],
                    "pipeline_count": row[f"pipeline_{typ}_n"],
                }
            )
    long = pd.DataFrame(metrics)
    long["age"] = pd.Categorical(long["age"], categories=AGE_ORDER, ordered=True)
    long["genotype"] = pd.Categorical(long["genotype"], categories=GENOTYPE_ORDER, ordered=True)
    grouped = (
        long.groupby(["typing_erode_px", "age", "age_month", "genotype", "fiber_type"])
        .agg(
            n_images=("image_id", "nunique"),
            pipeline_pct_mean=("pipeline_pct", "mean"),
            myosight_pct_mean=("myosight_pct", "mean"),
            pct_diff_vs_myosight_mean=("pct_diff_vs_myosight", "mean"),
            pipeline_count_mean=("pipeline_count", "mean"),
        )
        .reset_index()
    )
    return grouped


def genotype_effects(grouped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in grouped.groupby(["typing_erode_px", "age", "age_month", "fiber_type"]):
        erode_px, age, age_month, fiber_type = keys
        values = sub.set_index("genotype")
        if not {"mdx5cv", "mdx5cv-JAG1"}.issubset(values.index):
            continue
        mdx_pct = values.loc["mdx5cv", "pipeline_pct_mean"]
        jag_pct = values.loc["mdx5cv-JAG1", "pipeline_pct_mean"]
        mdx_count = values.loc["mdx5cv", "pipeline_count_mean"]
        jag_count = values.loc["mdx5cv-JAG1", "pipeline_count_mean"]
        rows.append(
            {
                "typing_erode_px": erode_px,
                "age": age,
                "age_month": age_month,
                "fiber_type": fiber_type,
                "jag1_minus_mdx_percentage_points": jag_pct - mdx_pct,
                "jag1_vs_mdx_count_pct": (jag_count - mdx_count) / mdx_count * 100
                if mdx_count
                else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def plot_type_proportions(grouped: pd.DataFrame, outdir: Path) -> None:
    plot_df = grouped[grouped["fiber_type"].isin(["IIb", "IIa", "IIx"])].copy()
    plot_df = remove_unused_categories(plot_df)
    grid = sns.relplot(
        data=plot_df,
        x="typing_erode_px",
        y="pipeline_pct_mean",
        hue="fiber_type",
        row="genotype",
        col="age",
        kind="line",
        marker="o",
        palette=TYPE_COLORS,
        height=3.1,
        aspect=0.95,
        facet_kws={"sharey": True},
    )
    grid.set_axis_labels("Typing erosion (px)", "Pipeline mean proportion (%)")
    grid.set_titles("{row_name} | {col_name}")
    for ax in grid.axes.flat:
        ax.set_ylim(0, 100)
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
    grid.figure.suptitle("Fiber Type Calls Across Typing Erosion", y=1.03)
    grid.figure.savefig(outdir / "typing_erosion_fiber_type_proportions.png", dpi=220)
    plt.close(grid.figure)


def plot_diff_vs_myosight(grouped: pd.DataFrame, outdir: Path) -> None:
    plot_df = grouped[grouped["fiber_type"].isin(["IIb", "IIa", "IIx"])].copy()
    plot_df = remove_unused_categories(plot_df)
    grid = sns.relplot(
        data=plot_df,
        x="typing_erode_px",
        y="pct_diff_vs_myosight_mean",
        hue="fiber_type",
        row="genotype",
        col="age",
        kind="line",
        marker="o",
        palette=TYPE_COLORS,
        height=3.1,
        aspect=0.95,
        facet_kws={"sharey": True},
    )
    grid.set_axis_labels("Typing erosion (px)", "Pipeline minus MyoSight (percentage points)")
    grid.set_titles("{row_name} | {col_name}")
    for ax in grid.axes.flat:
        ax.axhline(0, color="#333333", linewidth=1.0)
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
    grid.figure.suptitle("MyoSight Difference Across Typing Erosion", y=1.03)
    grid.figure.savefig(outdir / "typing_erosion_difference_vs_myosight.png", dpi=220)
    plt.close(grid.figure)


def plot_jag1_effects(effects: pd.DataFrame, outdir: Path) -> None:
    plot_df = effects[effects["fiber_type"].isin(["IIb", "IIx"])].copy()
    grid = sns.relplot(
        data=plot_df,
        x="typing_erode_px",
        y="jag1_minus_mdx_percentage_points",
        hue="fiber_type",
        col="age",
        kind="line",
        marker="o",
        palette={k: TYPE_COLORS[k] for k in ["IIb", "IIx"]},
        height=3.7,
        aspect=0.95,
        facet_kws={"sharey": False},
    )
    grid.set_axis_labels("Typing erosion (px)", "JAG1 minus mdx (percentage points)")
    grid.set_titles("{col_name}")
    for ax in grid.axes.flat:
        ax.axhline(0, color="#333333", linewidth=1.0)
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
    grid.figure.suptitle("JAG1 Fiber-Type Effect Across Typing Erosion", y=1.06)
    grid.figure.savefig(outdir / "typing_erosion_jag1_effect.png", dpi=220)
    plt.close(grid.figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep typing erosion using existing labels.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/validation/myosight_validation_consolidated_image_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation/typing_erosion_sweep"),
    )
    parser.add_argument("--erodes", type=parse_erodes, default=parse_erodes("0,1,2,3,4,5"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed image/erosion rows from the output CSV if present.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.72)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = add_biology_metadata(pd.read_csv(args.summary))
    if args.limit is not None:
        summary = summary.head(args.limit)
    image_summary_path = args.output_dir / "typing_erosion_image_summary.csv"
    rows = []
    completed: set[tuple[str, int]] = set()
    if args.resume and image_summary_path.exists():
        previous = pd.read_csv(image_summary_path)
        rows = previous.to_dict("records")
        completed = {
            (str(row["image_id"]), int(row["typing_erode_px"]))
            for _, row in previous.iterrows()
        }
        print(f"resuming from {len(previous)} completed image/erosion rows", flush=True)
    for _, row in summary.iterrows():
        remaining_erodes = [
            erode_px
            for erode_px in args.erodes
            if (str(row["image_id"]), int(erode_px)) not in completed
        ]
        if not remaining_erodes:
            print(f"{row['image_id']}: already complete", flush=True)
            continue
        rows.extend(sweep_image(row, remaining_erodes))
        pd.DataFrame(rows).to_csv(image_summary_path, index=False)
    image_summary = pd.DataFrame(rows)
    grouped = group_summary(image_summary)
    effects = genotype_effects(grouped)
    image_summary.to_csv(image_summary_path, index=False)
    grouped.to_csv(args.output_dir / "typing_erosion_group_summary.csv", index=False)
    effects.to_csv(args.output_dir / "typing_erosion_genotype_effects.csv", index=False)
    plot_type_proportions(grouped, args.output_dir)
    plot_diff_vs_myosight(grouped, args.output_dir)
    plot_jag1_effects(effects, args.output_dir)
    print(f"saved typing erosion sweep to {args.output_dir}")


if __name__ == "__main__":
    main()
