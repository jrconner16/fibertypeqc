from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

AMBIGUITY_TOKENS = ("iib_iix", "iix_iib")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Snapshot the current reviewed benchmark as a conservative label-policy baseline "
            "and prepare an IIb/IIx liberal re-review queue."
        )
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--benchmark-split", type=Path, required=True)
    parser.add_argument("--conservative-benchmark-output", type=Path, required=True)
    parser.add_argument("--conservative-split-output", type=Path, required=True)
    parser.add_argument("--review-queue-output", type=Path, required=True)
    parser.add_argument(
        "--label-policy",
        type=str,
        default="conservative_iib_iix_v1",
        help="Policy name recorded on the conservative snapshot files.",
    )
    return parser


def _load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False).copy()
    required = {"image_id", "label", "audit_final_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    df["image_id"] = df["image_id"].astype(str)
    df["label"] = df["label"].astype(int)
    df["audit_final_label"] = df["audit_final_label"].fillna("").astype(str).str.lower().str.strip()
    return df


def _contains_ambiguity_token(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.lower()
    out = pd.Series(False, index=series.index)
    for token in AMBIGUITY_TOKENS:
        out = out | text.str.contains(token, regex=False)
    return out


def annotate_conservative_snapshot(df: pd.DataFrame, *, policy_name: str) -> pd.DataFrame:
    out = df.copy()
    out["label_policy"] = policy_name
    out["conservative_audit_final_label"] = out["audit_final_label"].astype(str)
    out["label_policy_changed"] = False
    return out


def build_liberal_review_queue(benchmark_split: pd.DataFrame) -> pd.DataFrame:
    out = benchmark_split.copy()
    is_iib_iix = out["audit_final_label"].isin({"iib", "iix"})
    ambiguity_mask = pd.Series(False, index=out.index)
    for col in ("supervision_stratum", "manual_round2_stratum", "audit_bucket_list"):
        if col in out.columns:
            ambiguity_mask = ambiguity_mask | _contains_ambiguity_token(out[col])
    if "supervision_pool" in out.columns:
        ambiguity_mask = ambiguity_mask | (
            out["supervision_pool"].fillna("").astype(str).str.lower().eq("boundary_enrichment")
        )
    if "is_suspicious_iix_iib_boundary" in out.columns:
        ambiguity_mask = ambiguity_mask | out["is_suspicious_iix_iib_boundary"].fillna(False).astype(bool)

    queue = out.loc[is_iib_iix & ambiguity_mask].copy()
    if queue.empty:
        raise ValueError("No ambiguous IIb/IIx rows found for liberal relabel queue.")

    queue["conservative_audit_final_label"] = queue["audit_final_label"].astype(str)
    queue["label_policy_source"] = "conservative"
    queue["label_policy_target"] = "liberal_iib_iix_v1"
    queue["liberal_review_target"] = "iib_iix_ambiguity_recheck"

    # Clear prior adjudication fields so Napari review records a fresh liberal call.
    queue["audit_corrected_type"] = ""
    queue["audit_is_uncertain"] = False
    queue["audit_is_excluded"] = False
    queue["audit_notes"] = ""

    keep = [
        "image_id",
        "label",
        "audit_final_label",
        "conservative_audit_final_label",
        "manual_supervision_split",
        "split",
        "fiber_type",
        "fiber_type_pre_iia_gate",
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
        "supervision_pool",
        "supervision_stratum",
        "manual_round2_pool",
        "manual_round2_stratum",
        "audit_bucket_list",
        "iib_evidence_score",
        "boundary_score_within_image_pct",
        "iib_anchor_votes",
        "is_suspicious_iix_iib_boundary",
        "review_file",
        "label_policy_source",
        "label_policy_target",
        "liberal_review_target",
        "audit_corrected_type",
        "audit_is_uncertain",
        "audit_is_excluded",
        "audit_notes",
    ]
    keep = [col for col in keep if col in queue.columns]
    return queue.loc[:, keep].sort_values(
        ["manual_supervision_split", "image_id", "label"], kind="stable"
    )


def main() -> None:
    args = build_parser().parse_args()
    benchmark = _load_table(args.benchmark)
    benchmark_split = _load_table(args.benchmark_split)

    conservative_benchmark = annotate_conservative_snapshot(
        benchmark, policy_name=args.label_policy
    )
    conservative_split = annotate_conservative_snapshot(
        benchmark_split, policy_name=args.label_policy
    )
    review_queue = build_liberal_review_queue(benchmark_split)

    for path in (
        args.conservative_benchmark_output,
        args.conservative_split_output,
        args.review_queue_output,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    conservative_benchmark.to_csv(args.conservative_benchmark_output, index=False)
    conservative_split.to_csv(args.conservative_split_output, index=False)
    review_queue.to_csv(args.review_queue_output, index=False)

    print(f"saved conservative benchmark: {args.conservative_benchmark_output}")
    print(f"saved conservative split: {args.conservative_split_output}")
    print(f"saved liberal review queue: {args.review_queue_output}")
    print(f"review queue rows: {len(review_queue)}")
    print("review queue split counts:")
    print(review_queue["manual_supervision_split"].value_counts().to_string())


if __name__ == "__main__":
    main()
