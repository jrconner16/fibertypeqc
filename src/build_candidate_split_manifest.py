from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT_MANIFEST = Path("outputs/validation/myosight_validation_input_manifest.csv")
DEFAULT_CONSOLIDATED_SUMMARY = Path(
    "outputs/validation/myosight_validation_consolidated_image_summary.csv"
)
DEFAULT_OUTPUT = Path("outputs/validation/candidate_split_manifest.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an image-level candidate split manifest seeded from existing "
            "MyoSight/FiberTypeQC comparison outputs. This manifest is intended "
            "for manual split curation."
        )
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=DEFAULT_INPUT_MANIFEST,
        help="Path to the validation input manifest.",
    )
    parser.add_argument(
        "--consolidated-summary",
        type=Path,
        default=DEFAULT_CONSOLIDATED_SUMMARY,
        help="Path to the consolidated MyoSight/FiberTypeQC image summary.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path for the candidate split manifest.",
    )
    return parser


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _infer_timepoint(image_id: str) -> str:
    text = image_id.lower()
    if "1month" in text or "1mo" in text:
        return "1mo"
    if "4mo" in text:
        return "4mo"
    if "12mo" in text:
        return "12mo"
    if "1yo" in text or "1yr" in text:
        return "12mo"
    return "unknown"


def _infer_timepoint_from_row(image_id: str, source_image: str) -> str:
    for text in (str(image_id), str(source_image)):
        inferred = _infer_timepoint(text)
        if inferred != "unknown":
            return inferred
    return "unknown"


def _infer_genotype(image_id: str) -> str:
    text = image_id.lower()
    if "mdxjag" in text or "mdx.jag1" in text or "jag1" in text:
        return "mdxJag"
    if "mdx" in text:
        return "mdx"
    return "unknown"


def _infer_panel_type(input_kind: str) -> str:
    if input_kind in {"direct_czi", "section_tiff_export"}:
        return "baseline_iib_iia_membrane_residual_iix"
    return "unknown"


def _bin_series(values: pd.Series, labels: tuple[str, str, str]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(["unknown"] * len(values), index=values.index, dtype="object")
    q1 = float(valid.quantile(0.33))
    q2 = float(valid.quantile(0.66))
    bins = pd.Series(index=values.index, dtype="object")
    bins.loc[numeric <= q1] = labels[0]
    bins.loc[(numeric > q1) & (numeric <= q2)] = labels[1]
    bins.loc[numeric > q2] = labels[2]
    bins = bins.fillna("unknown")
    return bins


def _compute_difficulty_score(df: pd.DataFrame) -> pd.Series:
    review = pd.to_numeric(df["pipeline_needs_review_rate"], errors="coerce").fillna(0.0)
    warning = pd.to_numeric(df["pipeline_signal_warning_rate"], errors="coerce").fillna(0.0)
    count_diff = pd.to_numeric(df["fiber_count_abs_pct_diff"], errors="coerce").fillna(0.0) / 100.0
    return (0.45 * review + 0.35 * warning + 0.20 * count_diff).round(6)


def _suggest_saturation_proxy(row: pd.Series) -> str:
    dominant = max(
        float(row.get("pipeline_iib_pct", 0.0) or 0.0),
        float(row.get("pipeline_iia_pct", 0.0) or 0.0),
        float(row.get("pipeline_iix_pct", 0.0) or 0.0),
    )
    warning = float(row.get("pipeline_signal_warning_rate", 0.0) or 0.0)
    review = float(row.get("pipeline_needs_review_rate", 0.0) or 0.0)

    if warning >= 0.30 and dominant >= 0.60:
        return "possible_saturation_or_overstain"
    if warning >= 0.25 or review >= 0.18:
        return "possible_dynamic_range_issue"
    return "no_obvious_issue"


def _suggest_split(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(index=df.index, dtype="object")
    out[:] = "train"

    # Keep known alpha-development images in train by default.
    locked_train = df["used_in_alpha_training"].astype(bool)
    out.loc[locked_train] = "train"

    group_cols = ["timepoint", "genotype"]
    for _, idx in df.groupby(group_cols, sort=False).groups.items():
        group = df.loc[idx].sort_values("difficulty_score", ascending=False, kind="stable")
        candidates = group.loc[~group["used_in_alpha_training"].astype(bool)]
        if candidates.empty:
            continue

        # Put one harder non-alpha image into heldout per biological bucket when possible.
        heldout_idx = candidates.index[0]
        out.loc[heldout_idx] = "heldout"

        # Put one easier/mid non-alpha image into dev if the bucket has enough remaining images.
        remaining = candidates.drop(index=heldout_idx)
        if len(remaining) >= 2:
            dev_idx = remaining.sort_values("difficulty_score", kind="stable").index[
                len(remaining) // 2
            ]
            out.loc[dev_idx] = "dev"
    return out


def _round_if_present(df: pd.DataFrame, columns: list[str], digits: int = 4) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(digits)


def build_candidate_split_manifest(
    input_manifest: pd.DataFrame,
    consolidated_summary: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        input_manifest,
        {"image_id", "input_kind", "training_image", "source_image"},
        "input manifest",
    )
    _require_columns(
        consolidated_summary,
        {
            "image_id",
            "validation_input_kind",
            "pipeline_total_fibers",
            "pipeline_needs_review_rate",
            "pipeline_signal_warning_rate",
            "pipeline_iib_pct",
            "pipeline_iia_pct",
            "pipeline_iix_pct",
            "fiber_count_abs_pct_diff",
        },
        "consolidated summary",
    )

    base = input_manifest.copy()
    base["used_in_alpha_training"] = base["training_image"].astype(bool)
    base["has_myosight_comparison"] = True
    base["panel_type"] = base["input_kind"].map(_infer_panel_type)
    base["genotype"] = base["image_id"].map(_infer_genotype)
    base["timepoint"] = base.apply(
        lambda row: _infer_timepoint_from_row(row["image_id"], row["source_image"]),
        axis=1,
    )

    merged = base.merge(
        consolidated_summary,
        on="image_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_summary"),
    )

    merged["split"] = ""
    merged["candidate_included"] = True
    merged["has_manual_labels"] = ""
    merged["quality_bin_manual"] = ""
    merged["saturation_bin_manual"] = ""
    merged["stain_quality_notes"] = ""
    merged["distribution_notes"] = ""
    merged["split_notes"] = ""

    merged["fiber_count_bin"] = _bin_series(
        merged["pipeline_total_fibers"], ("small", "medium", "large")
    )
    merged["review_rate_bin"] = _bin_series(
        merged["pipeline_needs_review_rate"], ("low", "medium", "high")
    )
    merged["signal_warning_bin"] = _bin_series(
        merged["pipeline_signal_warning_rate"], ("low", "medium", "high")
    )
    merged["myosight_count_diff_bin"] = _bin_series(
        merged["fiber_count_abs_pct_diff"], ("low", "medium", "high")
    )
    merged["difficulty_score"] = _compute_difficulty_score(merged)
    merged["quality_bin_suggested"] = _bin_series(
        merged["difficulty_score"], ("cleaner", "mixed", "harder")
    )
    merged["saturation_proxy_suggested"] = merged.apply(_suggest_saturation_proxy, axis=1)
    merged["split_suggested"] = _suggest_split(merged)

    select_cols = [
        "image_id",
        "split",
        "split_suggested",
        "candidate_included",
        "used_in_alpha_training",
        "has_myosight_comparison",
        "has_manual_labels",
        "input_kind",
        "validation_input_kind",
        "panel_type",
        "genotype",
        "timepoint",
        "source_image",
        "fiber_count_bin",
        "review_rate_bin",
        "signal_warning_bin",
        "myosight_count_diff_bin",
        "difficulty_score",
        "quality_bin_suggested",
        "saturation_proxy_suggested",
        "pipeline_total_fibers",
        "pipeline_needs_review_rate",
        "pipeline_signal_warning_rate",
        "pipeline_iib_pct",
        "pipeline_iia_pct",
        "pipeline_iix_pct",
        "fiber_count_abs_pct_diff",
        "quality_bin_manual",
        "saturation_bin_manual",
        "stain_quality_notes",
        "distribution_notes",
        "split_notes",
    ]
    out = merged.loc[:, select_cols].copy()

    _round_if_present(
        out,
        [
            "pipeline_needs_review_rate",
            "pipeline_signal_warning_rate",
            "pipeline_iib_pct",
            "pipeline_iia_pct",
            "pipeline_iix_pct",
            "fiber_count_abs_pct_diff",
            "difficulty_score",
        ],
    )
    out = out.sort_values(
        by=["timepoint", "genotype", "input_kind", "image_id"],
        kind="stable",
    ).reset_index(drop=True)
    return out


def main() -> None:
    args = build_parser().parse_args()
    input_manifest = pd.read_csv(args.input_manifest)
    consolidated_summary = pd.read_csv(args.consolidated_summary)
    out = build_candidate_split_manifest(input_manifest, consolidated_summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"rows: {len(out)}")
    print(f"used_in_alpha_training: {int(out['used_in_alpha_training'].sum())}")
    print(
        "input kinds: "
        + ", ".join(
            f"{kind}={count}"
            for kind, count in out["input_kind"].value_counts().sort_index().items()
        )
    )
    print(
        "timepoints: "
        + ", ".join(
            f"{tp}={count}"
            for tp, count in out["timepoint"].value_counts().sort_index().items()
        )
    )
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
