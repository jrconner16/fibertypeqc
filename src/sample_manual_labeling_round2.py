from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

DEFAULT_EVAL_QUOTAS = {
    "eval_iia_focus": 60,
    "eval_iib_iix_ambiguity": 60,
    "eval_myosight_disparity": 30,
    "eval_review_flagged_controls": 40,
    "eval_model_disagreement": 20,
    "eval_easy_controls": 40,
}

DEFAULT_TRAIN_QUOTAS = {
    "train_iia_focus": 240,
    "train_iia_ambiguity": 140,
    "train_iib_iix_ambiguity": 220,
    "train_myosight_disparity": 80,
    "train_review_flagged_controls": 80,
    "train_easy_controls": 60,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a second-round manual-labeling sample from the combined audit set, "
            "with separate eval-candidate and train-candidate pools."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Combined audit set CSV.")
    parser.add_argument(
        "--reviewed",
        type=Path,
        default=Path("outputs/validation/reviewed_audit_all.csv"),
        help="Reviewed audit CSV used to exclude already-reviewed fibers.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output round-2 sample CSV.")
    parser.add_argument(
        "--eval-max-per-image",
        type=int,
        default=15,
        help="Maximum eval-candidate rows per image.",
    )
    parser.add_argument(
        "--train-max-per-image",
        type=int,
        default=45,
        help="Maximum train-candidate rows per image.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling order.",
    )
    return parser


def _load_combined(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {
        "image_id",
        "label",
        "split",
        "fiber_type",
        "audit_bucket_count",
        "bucket_model_disagreement",
        "bucket_low_margin",
        "bucket_review_flagged",
        "bucket_high_conf_iia",
        "bucket_matched_myosight_disparity",
        "disparity_bucket",
        "model_confidence",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    out = df.copy()
    out["image_id"] = out["image_id"].astype(str)
    out["label"] = out["label"].astype(int)
    out["fiber_type"] = out["fiber_type"].fillna("").astype(str).str.lower()
    out["split"] = out["split"].fillna("").astype(str).str.lower()
    out["disparity_bucket"] = out["disparity_bucket"].fillna("").astype(str).str.lower()
    return out


def _exclude_reviewed(df: pd.DataFrame, reviewed_path: Path | None) -> pd.DataFrame:
    if reviewed_path is None or not reviewed_path.exists():
        return df
    reviewed = pd.read_csv(reviewed_path, low_memory=False)
    required = {"image_id", "label"}
    missing = sorted(required - set(reviewed.columns))
    if missing:
        raise ValueError(f"{reviewed_path} is missing required columns: {', '.join(missing)}")
    reviewed_keys = set(
        zip(
            reviewed["image_id"].astype(str),
            reviewed["label"].astype(int),
            strict=False,
        )
    )
    mask = ~df.apply(lambda row: (row["image_id"], int(row["label"])) in reviewed_keys, axis=1)
    return df.loc[mask].copy()


def _bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def _build_strata(df: pd.DataFrame) -> dict[str, pd.Series]:
    fiber_type = df["fiber_type"]
    low_margin = _bool(df["bucket_low_margin"])
    review_flagged = _bool(df["bucket_review_flagged"])
    high_conf_iia = _bool(df["bucket_high_conf_iia"])
    model_disagreement = _bool(df["bucket_model_disagreement"])
    myosight_disparity = _bool(df["bucket_matched_myosight_disparity"])
    disparity_bucket = df["disparity_bucket"]
    confidence = pd.to_numeric(df["model_confidence"], errors="coerce").fillna(0.0)

    iia_disparity = disparity_bucket.isin(["iix_vs_iia", "iia_vs_iix"])
    iib_iix_disparity = disparity_bucket.isin(["iib_vs_iix", "iix_vs_iib"])

    return {
        "eval_iia_focus": (fiber_type == "iia") | high_conf_iia | iia_disparity,
        "eval_iib_iix_ambiguity": iib_iix_disparity
        | (fiber_type.isin(["iib", "iix"]) & low_margin),
        "eval_myosight_disparity": myosight_disparity,
        "eval_review_flagged_controls": review_flagged & ~low_margin,
        "eval_model_disagreement": model_disagreement,
        "eval_easy_controls": (~review_flagged) & (~low_margin) & (confidence >= 0.9),
        "train_iia_focus": (fiber_type == "iia") | high_conf_iia | iia_disparity,
        "train_iia_ambiguity": ((fiber_type == "iia") & (low_margin | review_flagged))
        | iia_disparity,
        "train_iib_iix_ambiguity": iib_iix_disparity
        | (fiber_type.isin(["iib", "iix"]) & low_margin),
        "train_myosight_disparity": myosight_disparity,
        "train_review_flagged_controls": review_flagged & ~low_margin,
        "train_easy_controls": (~review_flagged) & (~low_margin) & (confidence >= 0.9),
    }


def _sample_stratum(
    df: pd.DataFrame,
    mask: pd.Series,
    *,
    selected_keys: set[tuple[str, int]],
    image_counts: Counter[str],
    quota: int,
    max_per_image: int,
    split_priority: list[str],
    seed: int,
) -> list[int]:
    subset = df.loc[mask].copy()
    if subset.empty or quota <= 0:
        return []

    split_rank = {name: i for i, name in enumerate(split_priority)}
    subset["_split_rank"] = subset["split"].map(split_rank).fillna(len(split_priority))
    subset["_confidence"] = pd.to_numeric(subset["model_confidence"], errors="coerce").fillna(0.0)
    subset["_margin"] = pd.to_numeric(subset.get("model_margin", 0.0), errors="coerce").fillna(0.0)
    subset = subset.sample(frac=1.0, random_state=seed).sort_values(
        ["_split_rank", "audit_bucket_count", "_margin", "_confidence"],
        ascending=[True, False, True, False],
        kind="stable",
    )

    chosen: list[int] = []
    for row in subset.itertuples():
        key = (str(row.image_id), int(row.label))
        if key in selected_keys:
            continue
        if image_counts[str(row.image_id)] >= max_per_image:
            continue
        chosen.append(int(row.Index))
        selected_keys.add(key)
        image_counts[str(row.image_id)] += 1
        if len(chosen) >= quota:
            break
    return chosen


def sample_manual_labeling_round2(
    df: pd.DataFrame,
    *,
    eval_quotas: dict[str, int] | None = None,
    train_quotas: dict[str, int] | None = None,
    eval_max_per_image: int = 15,
    train_max_per_image: int = 45,
    seed: int = 0,
) -> pd.DataFrame:
    eval_quotas = dict(DEFAULT_EVAL_QUOTAS if eval_quotas is None else eval_quotas)
    train_quotas = dict(DEFAULT_TRAIN_QUOTAS if train_quotas is None else train_quotas)
    strata = _build_strata(df)

    selected_keys: set[tuple[str, int]] = set()
    eval_image_counts: Counter[str] = Counter()
    train_image_counts: Counter[str] = Counter()
    selected_frames: list[pd.DataFrame] = []

    pool_specs = [
        (
            "manual_eval_candidate",
            eval_quotas,
            ["heldout", "dev", "train"],
            eval_max_per_image,
            eval_image_counts,
        ),
        (
            "manual_train_candidate",
            train_quotas,
            ["train", "dev", "heldout"],
            train_max_per_image,
            train_image_counts,
        ),
    ]

    pool_offset = 0
    for pool_name, quotas, split_priority, max_per_image, image_counts in pool_specs:
        for offset, (stratum_name, quota) in enumerate(quotas.items()):
            chosen_idx = _sample_stratum(
                df,
                strata[stratum_name],
                selected_keys=selected_keys,
                image_counts=image_counts,
                quota=quota,
                max_per_image=max_per_image,
                split_priority=split_priority,
                seed=seed + pool_offset + offset,
            )
            if not chosen_idx:
                continue
            chosen = df.loc[chosen_idx].copy()
            chosen["manual_round2_pool"] = pool_name
            chosen["manual_round2_stratum"] = stratum_name
            chosen["manual_round2_split_priority"] = ",".join(split_priority)
            selected_frames.append(chosen)
        pool_offset += 1000

    if not selected_frames:
        return df.iloc[0:0].copy()

    sampled = pd.concat(selected_frames, ignore_index=True)
    sampled = sampled.sort_values(
        ["manual_round2_pool", "image_id", "manual_round2_stratum", "label"],
        kind="stable",
    )
    return sampled


def main() -> None:
    args = build_parser().parse_args()
    combined = _load_combined(args.input)
    remaining = _exclude_reviewed(combined, args.reviewed)
    sampled = sample_manual_labeling_round2(
        remaining,
        eval_max_per_image=args.eval_max_per_image,
        train_max_per_image=args.train_max_per_image,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(args.output, index=False)

    print(f"rows: {len(sampled)}")
    if not sampled.empty:
        print("pool counts:")
        print(sampled["manual_round2_pool"].value_counts().to_string())
        print("fiber_type counts:")
        print(sampled["fiber_type"].value_counts().to_string())
        print("fiber_type by pool:")
        print(
            sampled.groupby(["manual_round2_pool", "fiber_type"])
            .size()
            .rename("count")
            .to_string()
        )
        print("stratum counts:")
        print(sampled["manual_round2_stratum"].value_counts().to_string())
        print("top images:")
        print(sampled["image_id"].value_counts().head(15).to_string())
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
