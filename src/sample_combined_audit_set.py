from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample a combined audit set down to a review-sized CSV."
    )
    parser.add_argument("--input", type=Path, required=True, help="Combined audit set CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output sampled audit CSV.")
    parser.add_argument(
        "--per-bucket",
        type=int,
        default=25,
        help="Maximum number of sampled rows per audit bucket.",
    )
    parser.add_argument(
        "--max-per-image",
        type=int,
        default=12,
        help="Maximum number of sampled rows per image across all buckets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling order.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="dev,heldout,train",
        help="Comma-separated split priority/order for sampling.",
    )
    return parser


def _load_audit(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {"image_id", "label", "audit_bucket_count"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    return df


def _bucket_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("bucket_")]


def _sample_rows(
    df: pd.DataFrame,
    bucket_col: str,
    already_selected: set[tuple[str, int]],
    image_counts: Counter[str],
    *,
    per_bucket: int,
    max_per_image: int,
    split_priority: list[str],
    seed: int,
) -> list[int]:
    subset = df.loc[df[bucket_col].fillna(False).astype(bool)].copy()
    if subset.empty:
        return []

    split_rank = {name: i for i, name in enumerate(split_priority)}
    subset["_split_rank"] = subset["split"].astype(str).map(split_rank).fillna(len(split_priority))
    subset = subset.sample(frac=1.0, random_state=seed).sort_values(
        ["_split_rank", "audit_bucket_count"],
        ascending=[True, False],
        kind="stable",
    )

    chosen: list[int] = []
    for row in subset.itertuples():
        key = (str(row.image_id), int(row.label))
        if key in already_selected:
            continue
        if image_counts[str(row.image_id)] >= max_per_image:
            continue
        chosen.append(int(row.Index))
        already_selected.add(key)
        image_counts[str(row.image_id)] += 1
        if len(chosen) >= per_bucket:
            break
    return chosen


def sample_combined_audit_set(
    df: pd.DataFrame,
    *,
    per_bucket: int = 25,
    max_per_image: int = 12,
    split_priority: list[str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    if split_priority is None:
        split_priority = ["dev", "heldout", "train"]

    selected_keys: set[tuple[str, int]] = set()
    image_counts: Counter[str] = Counter()
    selected_indices: list[int] = []

    bucket_cols = _bucket_columns(df)
    bucket_order = sorted(
        bucket_cols,
        key=lambda col: int(df[col].fillna(False).astype(bool).sum()),
        reverse=True,
    )

    for offset, bucket_col in enumerate(bucket_order):
        selected_indices.extend(
            _sample_rows(
                df,
                bucket_col,
                selected_keys,
                image_counts,
                per_bucket=per_bucket,
                max_per_image=max_per_image,
                split_priority=split_priority,
                seed=seed + offset,
            )
        )

    sampled = df.loc[selected_indices].copy()
    sampled = sampled.sort_values(
        ["split", "image_id", "audit_bucket_count", "label"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    return sampled


def main() -> None:
    args = build_parser().parse_args()
    audit = _load_audit(args.input)
    split_priority = [s.strip() for s in args.splits.split(",") if s.strip()]
    sampled = sample_combined_audit_set(
        audit,
        per_bucket=args.per_bucket,
        max_per_image=args.max_per_image,
        split_priority=split_priority,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(args.output, index=False)
    print(f"rows: {len(sampled)}")
    print(f"images: {sampled['image_id'].nunique() if not sampled.empty else 0}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
