from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a combined audit set from feature-table metadata, candidate disagreements, "
            "and optional matched MyoSight disparity rows."
        )
    )
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--matched-disparity", type=Path, default=None)
    parser.add_argument("--low-margin-threshold", type=float, default=0.20)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.95)
    parser.add_argument(
        "--baseline-candidate-name",
        type=str,
        default="baseline_rf",
        help="Candidate name used as the default baseline comparator in the predictions table.",
    )
    parser.add_argument(
        "--comparison-candidate-name",
        type=str,
        default="expanded_rf",
        help="Candidate name used for the default disagreement bucket comparison.",
    )
    return parser


def _normalize_image_id(value: str) -> str:
    return str(value).strip().lower().replace(" ", "").replace("-", "_")


def _load_feature_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {"image_id", "label", "fiber_type", "needs_review"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    out = df.copy()
    out["_merge_image_id"] = out["image_id"].map(_normalize_image_id)
    return out


def _load_predictions(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    df = pd.read_csv(path)
    required = {"image_id", "label", "candidate_name", "predicted_fiber_type"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    wide = (
        df.pivot_table(
            index=["image_id", "label"],
            columns="candidate_name",
            values="predicted_fiber_type",
            aggfunc="first",
        )
        .rename(columns=lambda c: f"candidate_pred_{c}")
        .reset_index()
    )
    wide["_merge_image_id"] = wide["image_id"].map(_normalize_image_id)
    return wide.drop(columns=["image_id"])


def _load_matched_disparity(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    df = pd.read_csv(path)
    required = {"image_id", "label", "myosight_label", "disparity_bucket"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    keep = [
        col
        for col in [
            "image_id",
            "label",
            "myosight_label",
            "disparity_bucket",
            "centroid_distance_px",
            "area_ratio",
            "matched_fibers_path",
        ]
        if col in df.columns
    ]
    out = df.loc[:, keep].drop_duplicates(subset=["image_id", "label"]).copy()
    out["_merge_image_id"] = out["image_id"].map(_normalize_image_id)
    return out.drop(columns=["image_id"])


def _best_available(series: pd.Series, fallback: pd.Series | None = None) -> pd.Series:
    out = series.copy()
    if fallback is not None:
        out = out.where(out.notna(), fallback)
    return out


def build_combined_audit_set(
    feature_table: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    matched_disparity: pd.DataFrame | None = None,
    *,
    low_margin_threshold: float = 0.20,
    high_conf_threshold: float = 0.95,
    baseline_candidate_name: str = "baseline_rf",
    comparison_candidate_name: str = "expanded_rf",
) -> pd.DataFrame:
    base_cols = [
        col
        for col in [
            "_merge_image_id",
            "image_id",
            "label",
            "split",
            "fiber_type",
            "fiber_type_source",
            "needs_review",
            "model_confidence",
            "model_margin",
            "prob_iib",
            "prob_iia",
            "prob_iix",
            "genotype",
            "timepoint",
            "input_kind",
            "panel_type",
            "used_in_alpha_training",
            "candidate_included",
            "myosight_count_diff_bin",
            "review_rate_bin",
            "signal_warning_bin",
            "quality_bin_suggested",
            "quality_bin_manual",
            "saturation_proxy_suggested",
            "saturation_bin_manual",
            "feature_diagnostics_path",
        ]
        if col in feature_table.columns
    ]
    out = (
        feature_table.loc[:, base_cols]
        .drop_duplicates(subset=["_merge_image_id", "label"])
        .copy()
    )

    if predictions is not None:
        out = out.merge(
            predictions,
            on=["_merge_image_id", "label"],
            how="left",
            validate="one_to_one",
        )

    if matched_disparity is not None:
        out = out.merge(
            matched_disparity,
            on=["_merge_image_id", "label"],
            how="left",
            validate="one_to_one",
        )

    baseline_col = f"candidate_pred_{baseline_candidate_name}"
    comparison_col = f"candidate_pred_{comparison_candidate_name}"
    if baseline_col not in out.columns and "fiber_type" in out.columns:
        out[baseline_col] = out["fiber_type"]

    baseline_pred = (
        out[baseline_col] if baseline_col in out.columns else pd.Series("", index=out.index)
    )
    comparison_pred = (
        out[comparison_col] if comparison_col in out.columns else pd.Series("", index=out.index)
    )

    out["bucket_model_disagreement"] = (
        baseline_pred.notna()
        & comparison_pred.notna()
        & (baseline_pred.astype(str) != "")
        & (comparison_pred.astype(str) != "")
        & (baseline_pred.astype(str) != comparison_pred.astype(str))
    )
    out["bucket_low_margin"] = (
        pd.to_numeric(out.get("model_margin"), errors="coerce") <= low_margin_threshold
    ).fillna(False)
    out["bucket_review_flagged"] = out["needs_review"].fillna(False).astype(bool)
    out["bucket_high_conf_iia"] = (
        (out["fiber_type"].astype(str) == "iia")
        & (pd.to_numeric(out.get("model_confidence"), errors="coerce") >= high_conf_threshold)
    ).fillna(False)
    out["bucket_high_conf_iix"] = (
        (out["fiber_type"].astype(str) == "iix")
        & (pd.to_numeric(out.get("model_confidence"), errors="coerce") >= high_conf_threshold)
    ).fillna(False)
    out["bucket_matched_myosight_disparity"] = (
        out.get("disparity_bucket", pd.Series("", index=out.index)).fillna("").astype(str) != ""
    )

    bucket_cols = [col for col in out.columns if col.startswith("bucket_")]
    out["audit_bucket_count"] = out.loc[:, bucket_cols].sum(axis=1)
    out["audit_bucket_list"] = out.loc[:, bucket_cols].apply(
        lambda row: "|".join(
            col.removeprefix("bucket_") for col, flag in row.items() if bool(flag)
        ),
        axis=1,
    )

    selected = out.loc[out["audit_bucket_count"] > 0].copy()
    selected = selected.sort_values(
        ["audit_bucket_count", "split", "image_id", "label"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    return selected


def main() -> None:
    args = build_parser().parse_args()
    feature_table = _load_feature_table(args.feature_table)
    predictions = _load_predictions(args.predictions)
    matched_disparity = _load_matched_disparity(args.matched_disparity)

    audit = build_combined_audit_set(
        feature_table,
        predictions,
        matched_disparity,
        low_margin_threshold=args.low_margin_threshold,
        high_conf_threshold=args.high_confidence_threshold,
        baseline_candidate_name=args.baseline_candidate_name,
        comparison_candidate_name=args.comparison_candidate_name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output, index=False)

    bucket_cols = [col for col in audit.columns if col.startswith("bucket_")]
    print(f"rows: {len(audit)}")
    if "image_id" in audit.columns:
        print(f"images: {audit['image_id'].nunique()}")
    for col in bucket_cols:
        print(f"{col}: {int(audit[col].sum())}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
