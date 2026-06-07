from __future__ import annotations

import argparse
from collections import Counter
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
    "prob_iib",
    "prob_iia",
    "prob_iix",
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
    "genotype",
    "timepoint",
    "input_kind",
    "difficulty_score",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build next-step manual benchmark and supervision enrichment pools from the "
            "candidate feature table."
        )
    )
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument(
        "--reviewed",
        type=Path,
        default=Path("outputs/validation/reviewed_audit_all.csv"),
        help="Reviewed rows to exclude from all new pools.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--benchmark-quota",
        type=int,
        default=100,
        help="Total size of the broad benchmark enrichment pool.",
    )
    parser.add_argument(
        "--iia-positive-quota",
        type=int,
        default=100,
        help="Total size of the likely-true-IIa supervision pool.",
    )
    parser.add_argument(
        "--iia-iix-ambiguity-quota",
        type=int,
        default=100,
        help="Total size of the IIa/IIx ambiguity supervision pool.",
    )
    parser.add_argument(
        "--iib-positive-quota",
        type=int,
        default=100,
        help="Total size of the likely-true-IIb supervision pool.",
    )
    parser.add_argument(
        "--benchmark-max-per-image",
        type=int,
        default=8,
        help="Maximum benchmark-enrichment rows per image.",
    )
    parser.add_argument(
        "--supervision-max-per-image",
        type=int,
        default=20,
        help="Maximum supervision-enrichment rows per image and stratum.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _load_feature_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {
        "image_id",
        "label",
        "split",
        "fiber_type",
        "model_confidence",
        "model_margin",
        "prob_iib",
        "prob_iia",
        "prob_iix",
        "type1_mean",
        "type2_mean",
        "type1_coverage",
        "type2_coverage",
        "type1_cov_x_snr",
        "type1_snr_mean",
        "type2_cov_x_snr",
        "type2_snr_mean",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    out = df.copy()
    out["image_id"] = out["image_id"].astype(str)
    out["label"] = out["label"].astype(int)
    out["fiber_type"] = out["fiber_type"].fillna("").astype(str).str.lower().str.strip()
    out["split"] = out["split"].fillna("").astype(str).str.lower().str.strip()
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


def _sample_ranked(
    df: pd.DataFrame,
    *,
    score_col: str,
    quota: int,
    max_per_image: int,
    selected_keys: set[tuple[str, int]],
    image_counts: Counter[str],
    split_priority: list[str],
    seed: int,
) -> pd.DataFrame:
    if df.empty or quota <= 0:
        return df.iloc[0:0].copy()
    split_rank = {name: i for i, name in enumerate(split_priority)}
    ranked = df.copy()
    ranked["_split_rank"] = ranked["split"].map(split_rank).fillna(len(split_priority))
    ranked = ranked.sample(frac=1.0, random_state=seed).sort_values(
        ["_split_rank", score_col, "model_confidence"],
        ascending=[True, False, False],
        kind="stable",
    )

    chosen_rows: list[int] = []
    for row in ranked.itertuples():
        key = (str(row.image_id), int(row.label))
        if key in selected_keys:
            continue
        if image_counts[str(row.image_id)] >= max_per_image:
            continue
        chosen_rows.append(int(row.Index))
        selected_keys.add(key)
        image_counts[str(row.image_id)] += 1
        if len(chosen_rows) >= quota:
            break
    return ranked.loc[chosen_rows].copy() if chosen_rows else ranked.iloc[0:0].copy()


def _sample_balanced_by_fiber_type(
    df: pd.DataFrame,
    *,
    score_col: str,
    quota: int,
    max_per_image: int,
    selected_keys: set[tuple[str, int]],
    image_counts: Counter[str],
    split_priority: list[str],
    seed: int,
    target_types: tuple[str, ...],
) -> pd.DataFrame:
    if df.empty or quota <= 0:
        return df.iloc[0:0].copy()

    frames: list[pd.DataFrame] = []
    base_quota = quota // len(target_types)
    remainder = quota % len(target_types)
    consumed = 0

    for idx, fiber_type in enumerate(target_types):
        type_quota = base_quota + (1 if idx < remainder else 0)
        subset = df.loc[df["fiber_type"].astype(str).str.lower().eq(fiber_type)].copy()
        chosen = _sample_ranked(
            subset,
            score_col=score_col,
            quota=type_quota,
            max_per_image=max_per_image,
            selected_keys=selected_keys,
            image_counts=image_counts,
            split_priority=split_priority,
            seed=seed + idx,
        )
        consumed += len(chosen)
        if not chosen.empty:
            frames.append(chosen)

    remaining = quota - consumed
    if remaining > 0:
        chosen = _sample_ranked(
            df,
            score_col=score_col,
            quota=remaining,
            max_per_image=max_per_image,
            selected_keys=selected_keys,
            image_counts=image_counts,
            split_priority=split_priority,
            seed=seed + 999,
        )
        if not chosen.empty:
            frames.append(chosen)

    if not frames:
        return df.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=False)


def _build_benchmark_pool(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["benchmark_score"] = (
        0.60 * _rank_pct(out["model_margin"], ascending=False)
        + 0.20 * _rank_pct(out["difficulty_score"], ascending=True)
        + 0.20 * _rank_pct(out["pipeline_needs_review_rate"], ascending=True)
    )
    return out


def _build_iia_positive_pool(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mask = (
        pd.to_numeric(out["type2_snr_mean"], errors="coerce").ge(
            pd.to_numeric(out["type2_snr_mean"], errors="coerce").quantile(0.80)
        )
        & pd.to_numeric(out["type2_coverage"], errors="coerce").ge(
            pd.to_numeric(out["type2_coverage"], errors="coerce").quantile(0.65)
        )
        & pd.to_numeric(out["type1_snr_mean"], errors="coerce").le(
            pd.to_numeric(out["type1_snr_mean"], errors="coerce").quantile(0.65)
        )
        & pd.to_numeric(out["type2_mean"], errors="coerce").gt(
            pd.to_numeric(out["type1_mean"], errors="coerce")
        )
    )
    out = out.loc[mask].copy()
    out["pool_score"] = (
        _rank_pct(out["type2_snr_mean"], ascending=True)
        + _rank_pct(out["type2_cov_x_snr"], ascending=True)
        + _rank_pct(out["type2_coverage"], ascending=True)
        + _rank_pct(out["type2_mean"], ascending=True)
        - _rank_pct(out["type1_snr_mean"], ascending=True)
    )
    return out


def _build_iia_iix_ambiguity_pool(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prob_iia = pd.to_numeric(out["prob_iia"], errors="coerce").fillna(0.0)
    prob_iix = pd.to_numeric(out["prob_iix"], errors="coerce").fillna(0.0)
    prob_iib = pd.to_numeric(out["prob_iib"], errors="coerce").fillna(0.0)
    ambiguity = (prob_iia - prob_iix).abs()
    mask = (
        out["fiber_type"].isin(["iia", "iix"])
        & ambiguity.le(0.20)
        & prob_iib.le(0.35)
    ) | (
        out["fiber_type"].isin(["iia", "iix"])
        & pd.to_numeric(out["model_margin"], errors="coerce").fillna(1.0).le(0.10)
    )
    out = out.loc[mask].copy()
    out["pool_score"] = (
        _rank_pct(ambiguity, ascending=False)
        + _rank_pct(out["model_margin"], ascending=False)
        + _rank_pct(out["needs_review"].fillna(False).astype(bool).astype(int), ascending=True)
    )
    return out


def _build_iib_positive_pool(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mask = (
        pd.to_numeric(out["type1_snr_mean"], errors="coerce").ge(
            pd.to_numeric(out["type1_snr_mean"], errors="coerce").quantile(0.80)
        )
        & pd.to_numeric(out["type1_coverage"], errors="coerce").ge(
            pd.to_numeric(out["type1_coverage"], errors="coerce").quantile(0.65)
        )
        & pd.to_numeric(out["type2_snr_mean"], errors="coerce").le(
            pd.to_numeric(out["type2_snr_mean"], errors="coerce").quantile(0.65)
        )
        & pd.to_numeric(out["type1_mean"], errors="coerce").gt(
            pd.to_numeric(out["type2_mean"], errors="coerce")
        )
    )
    out = out.loc[mask].copy()
    out["pool_score"] = (
        _rank_pct(out["type1_snr_mean"], ascending=True)
        + _rank_pct(out["type1_cov_x_snr"], ascending=True)
        + _rank_pct(out["type1_coverage"], ascending=True)
        + _rank_pct(out["type1_mean"], ascending=True)
        - _rank_pct(out["type2_snr_mean"], ascending=True)
    )
    return out


def build_next_supervision_pools(
    df: pd.DataFrame,
    *,
    benchmark_quota: int = 100,
    iia_positive_quota: int = 100,
    iia_iix_ambiguity_quota: int = 100,
    iib_positive_quota: int = 100,
    benchmark_max_per_image: int = 8,
    supervision_max_per_image: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    selected_keys: set[tuple[str, int]] = set()
    benchmark_image_counts: Counter[str] = Counter()
    supervision_image_counts: Counter[str] = Counter()
    out_frames: list[pd.DataFrame] = []

    benchmark = _sample_ranked(
        _build_benchmark_pool(df),
        score_col="benchmark_score",
        quota=benchmark_quota,
        max_per_image=benchmark_max_per_image,
        selected_keys=selected_keys,
        image_counts=benchmark_image_counts,
        split_priority=["heldout", "dev", "train"],
        seed=seed,
    )
    if not benchmark.empty:
        benchmark["supervision_pool"] = "benchmark_enrichment"
        benchmark["supervision_stratum"] = "broad_random_controls"
        benchmark["proposed_manual_split"] = "manual_eval_holdout"
        out_frames.append(benchmark)

    supervision_specs = [
        ("iia_positive", _build_iia_positive_pool(df), iia_positive_quota, False),
        ("iia_iix_ambiguity", _build_iia_iix_ambiguity_pool(df), iia_iix_ambiguity_quota, True),
        ("iib_positive", _build_iib_positive_pool(df), iib_positive_quota, False),
    ]
    for offset, (stratum, pool_df, quota, balance_types) in enumerate(supervision_specs, start=1):
        if balance_types:
            chosen = _sample_balanced_by_fiber_type(
                pool_df,
                score_col="pool_score",
                quota=quota,
                max_per_image=supervision_max_per_image,
                selected_keys=selected_keys,
                image_counts=supervision_image_counts,
                split_priority=["train", "dev", "heldout"],
                seed=seed + 1000 * offset,
                target_types=("iia", "iix"),
            )
        else:
            chosen = _sample_ranked(
                pool_df,
                score_col="pool_score",
                quota=quota,
                max_per_image=supervision_max_per_image,
                selected_keys=selected_keys,
                image_counts=supervision_image_counts,
                split_priority=["train", "dev", "heldout"],
                seed=seed + 1000 * offset,
            )
        if chosen.empty:
            continue
        chosen["supervision_pool"] = "supervision_enrichment"
        chosen["supervision_stratum"] = stratum
        chosen["proposed_manual_split"] = "manual_train"
        out_frames.append(chosen)

    if not out_frames:
        return df.iloc[0:0].copy()

    out = pd.concat(out_frames, ignore_index=True)
    keep_cols = [col for col in DEFAULT_KEEP_COLUMNS if col in out.columns]
    extra_cols = [
        col
        for col in (
            "supervision_pool",
            "supervision_stratum",
            "proposed_manual_split",
            "pool_score",
            "benchmark_score",
        )
        if col in out.columns and col not in keep_cols
    ]
    out = out.loc[:, keep_cols + extra_cols].copy()
    out = out.sort_values(
        ["supervision_pool", "supervision_stratum", "image_id", "label"],
        kind="stable",
    )
    return out


def main() -> None:
    args = build_parser().parse_args()
    feature_table = _load_feature_table(args.feature_table)
    feature_table = _exclude_reviewed(feature_table, args.reviewed)
    out = build_next_supervision_pools(
        feature_table,
        benchmark_quota=args.benchmark_quota,
        iia_positive_quota=args.iia_positive_quota,
        iia_iix_ambiguity_quota=args.iia_iix_ambiguity_quota,
        iib_positive_quota=args.iib_positive_quota,
        benchmark_max_per_image=args.benchmark_max_per_image,
        supervision_max_per_image=args.supervision_max_per_image,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"rows: {len(out)}")
    if not out.empty:
        print("pool counts:")
        print(out["supervision_pool"].value_counts().to_string())
        print("stratum counts:")
        print(out["supervision_stratum"].value_counts().to_string())
        print("source fiber_type counts:")
        print(out["fiber_type"].value_counts().to_string())
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
