from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split the reviewed audit benchmark into a manual-eval holdback subset and an "
            "optional manual-train supervision subset."
        )
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--eval-frac",
        type=float,
        default=0.25,
        help="Approximate fraction of reviewed rows to reserve for manual_eval_holdout.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic split assignment.",
    )
    return parser


def _load_benchmark(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {"image_id", "label", "audit_final_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    out = df.copy()
    out["label"] = out["label"].astype(int)
    out["audit_final_label"] = (
        out["audit_final_label"].fillna("").astype(str).str.strip().str.lower()
    )
    return out


def split_reviewed_benchmark(
    benchmark: pd.DataFrame,
    *,
    eval_frac: float = 0.25,
    seed: int = 0,
) -> pd.DataFrame:
    if not 0.0 < eval_frac < 1.0:
        raise ValueError("eval_frac must be between 0 and 1.")

    df = benchmark.copy()
    df["manual_supervision_split"] = "manual_train"

    if "manual_round2_pool" in df.columns:
        pool = df["manual_round2_pool"].fillna("").astype(str).str.strip().str.lower()
        df.loc[pool == "manual_eval_candidate", "manual_supervision_split"] = (
            "manual_eval_holdout"
        )
        df.loc[pool == "manual_train_candidate", "manual_supervision_split"] = "manual_train"

    eval_indices: list[int] = []

    for label_name, group in df.groupby("audit_final_label", sort=True):
        if label_name == "":
            continue
        if "manual_round2_pool" in group.columns:
            unlabeled_group = group.loc[group["manual_round2_pool"].fillna("").astype(str).eq("")]
        else:
            unlabeled_group = group
        group = unlabeled_group
        if group.empty:
            continue
        group = group.sort_values(["image_id", "label"], kind="stable").sample(
            frac=1.0, random_state=seed
        )
        n = len(group)
        if n <= 1:
            continue
        n_eval = max(1, int(round(n * eval_frac)))
        n_eval = min(n_eval, n - 1)
        eval_indices.extend(group.index[:n_eval].tolist())

    if eval_indices:
        df.loc[eval_indices, "manual_supervision_split"] = "manual_eval_holdout"

    df["manual_supervision_weight"] = df["manual_supervision_split"].map(
        {
            "manual_train": 1.0,
            "manual_eval_holdout": 0.0,
        }
    )
    return df.sort_values(["manual_supervision_split", "image_id", "label"], kind="stable")


def main() -> None:
    args = build_parser().parse_args()
    benchmark = _load_benchmark(args.benchmark)
    split_df = split_reviewed_benchmark(benchmark, eval_frac=args.eval_frac, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(args.output, index=False)

    print(f"rows: {len(split_df)}")
    print("split counts:")
    print(split_df["manual_supervision_split"].value_counts().to_string())
    print("class counts by split:")
    print(
        split_df.groupby(["manual_supervision_split", "audit_final_label"])
        .size()
        .rename("count")
        .to_string()
    )
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
