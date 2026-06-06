from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_KEEP_COLUMNS: tuple[str, ...] = (
    "image_id",
    "label",
    "split",
    "fiber_type",
    "needs_review",
    "model_confidence",
    "model_margin",
    "quality_bin_suggested",
    "review_rate_bin",
    "signal_warning_bin",
    "saturation_proxy_suggested",
    "type1_mean",
    "type2_mean",
    "type1_coverage",
    "type2_coverage",
    "type1_cov_x_snr",
    "type1_snr_mean",
    "type1_snr_p90",
    "type2_cov_x_snr",
    "type2_snr_mean",
    "type2_snr_p90",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a positive-hunt CSV for likely true IIa fibers using evidence features "
            "rather than model confidence."
        )
    )
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reviewed",
        type=Path,
        default=Path("outputs/validation/reviewed_audit_all.csv"),
        help="Optional reviewed-audit CSV used to exclude already-reviewed fibers.",
    )
    parser.add_argument(
        "--max-per-image",
        type=int,
        default=20,
        help="Maximum number of candidates to keep per image.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=250,
        help="Maximum total number of candidates to emit after scoring.",
    )
    parser.add_argument(
        "--min-type2-snr-quantile",
        type=float,
        default=0.85,
        help="Minimum quantile threshold for type2_snr_mean.",
    )
    parser.add_argument(
        "--min-type2-coverage-quantile",
        type=float,
        default=0.70,
        help="Minimum quantile threshold for type2_coverage.",
    )
    parser.add_argument(
        "--max-type1-snr-quantile",
        type=float,
        default=0.60,
        help="Maximum quantile threshold for type1_snr_mean.",
    )
    parser.add_argument(
        "--allow-saturation-issues",
        action="store_true",
        help="Keep images flagged with dynamic-range/saturation warnings.",
    )
    parser.add_argument(
        "--allow-high-review-images",
        action="store_true",
        help="Keep candidates from images with high review-rate bins.",
    )
    return parser


def _load_feature_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {
        "image_id",
        "label",
        "type1_mean",
        "type2_mean",
        "type1_coverage",
        "type2_coverage",
        "type1_snr_mean",
        "type2_snr_mean",
        "type1_cov_x_snr",
        "type2_cov_x_snr",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    out = df.copy()
    out["image_id"] = out["image_id"].astype(str)
    out["label"] = out["label"].astype(int)
    return out


def _exclude_reviewed(df: pd.DataFrame, reviewed_path: Path | None) -> pd.DataFrame:
    if reviewed_path is None or not reviewed_path.exists():
        return df
    reviewed = pd.read_csv(reviewed_path, low_memory=False)
    if not {"image_id", "label"}.issubset(reviewed.columns):
        raise ValueError(f"{reviewed_path} must include image_id and label columns.")
    reviewed_keys = set(
        zip(
            reviewed["image_id"].astype(str),
            reviewed["label"].astype(int),
            strict=False,
        )
    )
    mask = ~df.apply(lambda row: (row["image_id"], int(row["label"])) in reviewed_keys, axis=1)
    return df.loc[mask].copy()


def _rank_pct(series: pd.Series, ascending: bool = True) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rank(pct=True, ascending=ascending)


def build_true_iia_hunt(
    df: pd.DataFrame,
    *,
    max_per_image: int = 20,
    top_k: int = 250,
    min_type2_snr_quantile: float = 0.85,
    min_type2_coverage_quantile: float = 0.70,
    max_type1_snr_quantile: float = 0.60,
    allow_saturation_issues: bool = False,
    allow_high_review_images: bool = False,
) -> pd.DataFrame:
    out = df.copy()

    type2_snr_cut = pd.to_numeric(out["type2_snr_mean"], errors="coerce").quantile(
        min_type2_snr_quantile
    )
    type2_cov_cut = pd.to_numeric(out["type2_coverage"], errors="coerce").quantile(
        min_type2_coverage_quantile
    )
    type1_snr_cut = pd.to_numeric(out["type1_snr_mean"], errors="coerce").quantile(
        max_type1_snr_quantile
    )

    mask = (
        pd.to_numeric(out["type2_snr_mean"], errors="coerce").ge(type2_snr_cut)
        & pd.to_numeric(out["type2_coverage"], errors="coerce").ge(type2_cov_cut)
        & pd.to_numeric(out["type1_snr_mean"], errors="coerce").le(type1_snr_cut)
        & pd.to_numeric(out["type2_mean"], errors="coerce")
        .gt(pd.to_numeric(out["type1_mean"], errors="coerce"))
    )

    if "saturation_proxy_suggested" in out.columns and not allow_saturation_issues:
        mask &= out["saturation_proxy_suggested"].fillna("").astype(str).isin(
            ["", "no_obvious_issue"]
        )

    if "review_rate_bin" in out.columns and not allow_high_review_images:
        mask &= ~out["review_rate_bin"].fillna("").astype(str).str.lower().eq("high")

    candidates = out.loc[mask].copy()
    if candidates.empty:
        return candidates

    candidates["iia_evidence_score"] = (
        _rank_pct(candidates["type2_snr_mean"], ascending=True)
        + _rank_pct(candidates["type2_cov_x_snr"], ascending=True)
        + _rank_pct(candidates["type2_coverage"], ascending=True)
        + _rank_pct(candidates["type2_mean"], ascending=True)
        - _rank_pct(candidates["type1_snr_mean"], ascending=True)
        - _rank_pct(candidates["type1_cov_x_snr"], ascending=True)
        - _rank_pct(candidates["type1_coverage"], ascending=True)
    )

    if "needs_review" in candidates.columns:
        candidates["iia_evidence_score"] -= (
            candidates["needs_review"].fillna(False).astype(bool).astype(float) * 0.25
        )

    candidates = candidates.sort_values(
        ["iia_evidence_score", "type2_snr_mean", "type2_coverage"],
        ascending=[False, False, False],
        kind="stable",
    )
    candidates = candidates.groupby("image_id", group_keys=False).head(max_per_image)
    candidates = candidates.head(top_k).copy()

    keep_cols = [col for col in DEFAULT_KEEP_COLUMNS if col in candidates.columns]
    extra_cols = [
        col
        for col in ("iia_evidence_score",)
        if col in candidates.columns and col not in keep_cols
    ]
    return candidates.loc[:, keep_cols + extra_cols]


def main() -> None:
    args = build_parser().parse_args()
    feature_table = _load_feature_table(args.feature_table)
    feature_table = _exclude_reviewed(feature_table, args.reviewed)
    out = build_true_iia_hunt(
        feature_table,
        max_per_image=args.max_per_image,
        top_k=args.top_k,
        min_type2_snr_quantile=args.min_type2_snr_quantile,
        min_type2_coverage_quantile=args.min_type2_coverage_quantile,
        max_type1_snr_quantile=args.max_type1_snr_quantile,
        allow_saturation_issues=args.allow_saturation_issues,
        allow_high_review_images=args.allow_high_review_images,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"rows: {len(out)}")
    if not out.empty:
        print("source fiber_type counts:")
        print(out["fiber_type"].value_counts().to_string())
        print("top images:")
        print(out["image_id"].value_counts().head(15).to_string())
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
