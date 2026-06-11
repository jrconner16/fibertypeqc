from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from src.build_candidate_feature_table import (
    _load_manifest,
    assemble_candidate_feature_table,
    discover_diagnostics_files,
)

KEEP_SAMPLE_COLUMNS: tuple[str, ...] = (
    "image_id",
    "label",
    "split",
    "supervision_pool",
    "supervision_stratum",
    "proposed_manual_split",
    "fiber_type",
    "fiber_type_pre_iia_gate",
    "needs_review",
    "model_confidence",
    "model_margin",
    "prob_iib",
    "prob_iia",
    "prob_iix",
    "type1_mean",
    "type2_mean",
    "type1_coverage",
    "type2_coverage",
    "type1_snr_mean",
    "type2_snr_mean",
    "type1_cov_x_snr",
    "type2_cov_x_snr",
    "genotype",
    "timepoint",
    "input_kind",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit current candidate IIx predictions for suspicious IIb-like evidence and "
            "build a focused manual-review tranche."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--reviewed", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--sample-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument(
        "--diagnostics-glob",
        type=str,
        default="**/*_feature_diagnostics.csv",
    )
    parser.add_argument(
        "--iib-anchor-quantile",
        type=float,
        default=0.25,
        help="Lower quantile of current IIb predictions used as a soft anchor for IIb evidence.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=250,
        help="Maximum total IIx boundary rows to emit for manual review.",
    )
    parser.add_argument(
        "--max-per-image",
        type=int,
        default=25,
        help="Maximum emitted IIx boundary rows per image.",
    )
    parser.add_argument(
        "--per-image-top-fraction",
        type=float,
        default=0.15,
        help="Top within-image fraction of predicted IIx rows to treat as the broader IIb-boundary band.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=10,
        help="Maximum number of images to include in the emitted review tranche.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _load_candidate_table(
    input_root: Path,
    diagnostics_glob: str,
    manifest_path: Path | None,
) -> pd.DataFrame:
    diagnostics_files = discover_diagnostics_files(input_root, diagnostics_glob)
    manifest = _load_manifest(manifest_path)
    table = assemble_candidate_feature_table(diagnostics_files, manifest)
    required = {
        "image_id",
        "label",
        "fiber_type",
        "prob_iib",
        "prob_iix",
        "type1_mean",
        "type2_mean",
        "type1_coverage",
        "type2_coverage",
        "type1_snr_mean",
        "type2_snr_mean",
        "type1_cov_x_snr",
        "type2_cov_x_snr",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Candidate diagnostics table is missing columns: {', '.join(missing)}")
    out = table.copy()
    out["image_id"] = out["image_id"].astype(str)
    out["label"] = out["label"].astype(int)
    out["fiber_type"] = out["fiber_type"].fillna("").astype(str).str.lower().str.strip()
    if "split" in out.columns:
        out["split"] = out["split"].fillna("").astype(str).str.lower().str.strip()
    else:
        out["split"] = ""
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
    keep_mask = ~df.apply(lambda row: (row["image_id"], int(row["label"])) in reviewed_keys, axis=1)
    return df.loc[keep_mask].copy()


def _rank_pct(series: pd.Series, ascending: bool = True) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rank(pct=True, ascending=ascending)


def derive_iib_anchor_thresholds(
    table: pd.DataFrame,
    *,
    anchor_quantile: float = 0.25,
) -> dict[str, float]:
    if not 0.0 < anchor_quantile < 1.0:
        raise ValueError("anchor_quantile must be between 0 and 1.")
    iib = table.loc[table["fiber_type"].eq("iib")].copy()
    if iib.empty:
        raise ValueError("No current IIb predictions available to derive IIb anchor thresholds.")
    upper_quantile = 1.0 - anchor_quantile
    return {
        "min_prob_iib": float(pd.to_numeric(iib["prob_iib"], errors="coerce").quantile(anchor_quantile)),
        "min_type1_coverage": float(
            pd.to_numeric(iib["type1_coverage"], errors="coerce").quantile(anchor_quantile)
        ),
        "min_type1_snr_mean": float(
            pd.to_numeric(iib["type1_snr_mean"], errors="coerce").quantile(anchor_quantile)
        ),
        "min_type1_cov_x_snr": float(
            pd.to_numeric(iib["type1_cov_x_snr"], errors="coerce").quantile(anchor_quantile)
        ),
        "max_type2_coverage": float(
            pd.to_numeric(iib["type2_coverage"], errors="coerce").quantile(upper_quantile)
        ),
        "max_type2_snr_mean": float(
            pd.to_numeric(iib["type2_snr_mean"], errors="coerce").quantile(upper_quantile)
        ),
    }


def score_iix_iib_boundary(
    table: pd.DataFrame,
    *,
    anchor_quantile: float = 0.25,
) -> tuple[pd.DataFrame, dict[str, float]]:
    thresholds = derive_iib_anchor_thresholds(table, anchor_quantile=anchor_quantile)
    out = table.copy()

    out["iib_evidence_score"] = (
        0.30 * _rank_pct(out["prob_iib"], ascending=True)
        + 0.20 * _rank_pct(out["type1_cov_x_snr"], ascending=True)
        + 0.15 * _rank_pct(out["type1_snr_mean"], ascending=True)
        + 0.15 * _rank_pct(out["type1_coverage"], ascending=True)
        + 0.10 * _rank_pct(out["type1_mean"], ascending=True)
        + 0.10 * _rank_pct(out["type1_mean"] - out["type2_mean"], ascending=True)
    ).round(6)

    out["flag_prob_iib_anchor"] = pd.to_numeric(out["prob_iib"], errors="coerce").ge(
        thresholds["min_prob_iib"]
    )
    out["flag_type1_coverage_anchor"] = pd.to_numeric(
        out["type1_coverage"], errors="coerce"
    ).ge(thresholds["min_type1_coverage"])
    out["flag_type1_snr_anchor"] = pd.to_numeric(out["type1_snr_mean"], errors="coerce").ge(
        thresholds["min_type1_snr_mean"]
    )
    out["flag_type1_cov_x_snr_anchor"] = pd.to_numeric(
        out["type1_cov_x_snr"], errors="coerce"
    ).ge(thresholds["min_type1_cov_x_snr"])
    out["flag_low_type2_coverage"] = pd.to_numeric(
        out["type2_coverage"], errors="coerce"
    ).le(thresholds["max_type2_coverage"])
    out["flag_low_type2_snr"] = pd.to_numeric(out["type2_snr_mean"], errors="coerce").le(
        thresholds["max_type2_snr_mean"]
    )
    out["flag_type1_gt_type2"] = pd.to_numeric(out["type1_mean"], errors="coerce").gt(
        pd.to_numeric(out["type2_mean"], errors="coerce")
    )

    flag_columns = [
        "flag_prob_iib_anchor",
        "flag_type1_coverage_anchor",
        "flag_type1_snr_anchor",
        "flag_type1_cov_x_snr_anchor",
        "flag_low_type2_coverage",
        "flag_low_type2_snr",
        "flag_type1_gt_type2",
    ]
    out["iib_anchor_votes"] = out[flag_columns].fillna(False).astype(int).sum(axis=1)
    out["is_suspicious_iix_iib_boundary"] = out["fiber_type"].eq("iix") & (
        (out["iib_anchor_votes"] >= 5)
        | ((out["iib_anchor_votes"] >= 4) & out["iib_evidence_score"].ge(0.85))
    )
    return out, thresholds


def build_iix_iib_summary(scored: pd.DataFrame) -> pd.DataFrame:
    predicted_iix = scored.loc[scored["fiber_type"].eq("iix")].copy()
    if predicted_iix.empty:
        return pd.DataFrame()

    group_cols = [
        col
        for col in ["image_id", "split", "genotype", "timepoint", "input_kind"]
        if col in predicted_iix.columns
    ]
    summary = (
        predicted_iix.groupby(group_cols, dropna=False)
        .agg(
            predicted_iix_n=("label", "size"),
            suspicious_iix_n=("is_suspicious_iix_iib_boundary", "sum"),
            mean_prob_iib=("prob_iib", "mean"),
            mean_prob_iix=("prob_iix", "mean"),
            mean_iib_evidence_score=("iib_evidence_score", "mean"),
            max_iib_evidence_score=("iib_evidence_score", "max"),
            mean_iib_anchor_votes=("iib_anchor_votes", "mean"),
        )
        .reset_index()
    )
    summary["suspicious_iix_rate"] = (
        summary["suspicious_iix_n"] / summary["predicted_iix_n"].clip(lower=1)
    ).round(6)
    return summary.sort_values(
        ["suspicious_iix_n", "suspicious_iix_rate", "mean_iib_evidence_score"],
        ascending=[False, False, False],
        kind="stable",
    )


def sample_suspicious_iix_rows(
    scored: pd.DataFrame,
    *,
    sample_size: int = 250,
    max_per_image: int = 25,
    per_image_top_fraction: float = 0.15,
    max_images: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    if not 0.0 < per_image_top_fraction < 1.0:
        raise ValueError("per_image_top_fraction must be between 0 and 1.")
    if max_images <= 0:
        raise ValueError("max_images must be positive.")

    predicted_iix = scored.loc[scored["fiber_type"].eq("iix")].copy()
    if predicted_iix.empty:
        return predicted_iix

    predicted_iix["supervision_pool"] = "boundary_enrichment"
    predicted_iix["supervision_stratum"] = "iix_iib_boundary"
    predicted_iix["proposed_manual_split"] = predicted_iix["split"].map(
        {
            "heldout": "manual_eval_candidate",
            "dev": "manual_eval_candidate",
            "train": "manual_train_candidate",
        }
    ).fillna("manual_train_candidate")
    predicted_iix["boundary_score_within_image_pct"] = predicted_iix.groupby("image_id")[
        "iib_evidence_score"
    ].rank(method="average", pct=True, ascending=True)
    threshold = 1.0 - per_image_top_fraction
    predicted_iix["in_broader_iix_iib_band"] = predicted_iix[
        "boundary_score_within_image_pct"
    ].ge(threshold)

    broader = predicted_iix.loc[predicted_iix["in_broader_iix_iib_band"]].copy()
    if broader.empty:
        return broader

    image_priority = (
        broader.groupby("image_id", dropna=False)
        .agg(
            band_rows=("label", "size"),
            suspicious_rows=("is_suspicious_iix_iib_boundary", "sum"),
            max_score=("iib_evidence_score", "max"),
            mean_score=("iib_evidence_score", "mean"),
        )
        .reset_index()
        .sort_values(
            ["suspicious_rows", "band_rows", "max_score", "mean_score", "image_id"],
            ascending=[False, False, False, False, True],
            kind="stable",
        )
    )
    chosen_images = set(image_priority.head(max_images)["image_id"].astype(str))
    broader = broader.loc[broader["image_id"].astype(str).isin(chosen_images)].copy()

    split_rank = {"heldout": 0, "dev": 1, "train": 2, "": 3}
    ranked = broader.sample(frac=1.0, random_state=seed).sort_values(
        [
            "split",
            "is_suspicious_iix_iib_boundary",
            "iib_anchor_votes",
            "iib_evidence_score",
            "prob_iib",
            "model_confidence",
        ],
        ascending=[True, False, False, False, False, True],
        key=lambda col: col.map(split_rank) if col.name == "split" else col,
        kind="stable",
    )

    selected: list[int] = []
    image_counts: Counter[str] = Counter()
    for row in ranked.itertuples():
        image_id = str(row.image_id)
        if image_counts[image_id] >= max_per_image:
            continue
        selected.append(int(row.Index))
        image_counts[image_id] += 1
        if len(selected) >= sample_size:
            break

    keep_cols = [col for col in KEEP_SAMPLE_COLUMNS if col in ranked.columns]
    extra_cols = [
        "iib_evidence_score",
        "boundary_score_within_image_pct",
        "in_broader_iix_iib_band",
        "iib_anchor_votes",
        "is_suspicious_iix_iib_boundary",
        "flag_prob_iib_anchor",
        "flag_type1_coverage_anchor",
        "flag_type1_snr_anchor",
        "flag_type1_cov_x_snr_anchor",
        "flag_low_type2_coverage",
        "flag_low_type2_snr",
        "flag_type1_gt_type2",
    ]
    return ranked.loc[selected, keep_cols + extra_cols].reset_index(drop=True)


def build_report(
    scored: pd.DataFrame,
    summary: pd.DataFrame,
    sample: pd.DataFrame,
    thresholds: dict[str, float],
) -> str:
    predicted_iix = scored.loc[scored["fiber_type"].eq("iix")].copy()
    suspicious = predicted_iix.loc[predicted_iix["is_suspicious_iix_iib_boundary"]].copy()

    lines = [
        "IIx/IIb boundary audit for current candidate outputs",
        "",
        "IIb anchor thresholds derived from current IIb predictions:",
    ]
    for key, value in thresholds.items():
        lines.append(f"- {key}={value:.6f}")

    lines.extend(
        [
            "",
            f"predicted_iix_rows={len(predicted_iix)}",
            f"suspicious_iix_rows={len(suspicious)}",
            (
                f"suspicious_share_of_iix={len(suspicious) / max(len(predicted_iix), 1):.4f}"
            ),
            f"sample_rows_emitted={len(sample)}",
            "",
            "Top image concentrations:",
        ]
    )
    if summary.empty:
        lines.append("none")
    else:
        lines.append(summary.head(10).to_string(index=False))
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    table = _load_candidate_table(args.input_root, args.diagnostics_glob, args.manifest)
    table = _exclude_reviewed(table, args.reviewed)
    scored, thresholds = score_iix_iib_boundary(
        table,
        anchor_quantile=args.iib_anchor_quantile,
    )
    summary = build_iix_iib_summary(scored)
    sample = sample_suspicious_iix_rows(
        scored,
        sample_size=args.sample_size,
        max_per_image=args.max_per_image,
        per_image_top_fraction=args.per_image_top_fraction,
        max_images=args.max_images,
        seed=args.seed,
    )
    report = build_report(scored, summary, sample, thresholds)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.sample_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(args.summary_output, index=False)
    sample.to_csv(args.sample_output, index=False)
    args.report_output.write_text(report, encoding="utf-8")

    print(f"rows_scored: {len(scored)}")
    print(f"predicted_iix_rows: {int(scored['fiber_type'].eq('iix').sum())}")
    print(f"suspicious_iix_rows: {int(scored['is_suspicious_iix_iib_boundary'].sum())}")
    print(f"saved summary: {args.summary_output}")
    print(f"saved sample: {args.sample_output}")
    print(f"saved report: {args.report_output}")


if __name__ == "__main__":
    main()
