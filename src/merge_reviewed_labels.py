from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.fiber_type_labels import normalize_review_label, to_biological_label
from src.io_utils import save_dataframe


def _fiber_id_column(df: pd.DataFrame) -> str:
    if "fiber_id" in df.columns:
        return "fiber_id"
    if "label" in df.columns:
        return "label"
    raise ValueError("Fiber table must contain either 'fiber_id' or 'label'.")


def merge_reviewed_labels(fibers: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame:
    id_col = _fiber_id_column(fibers)
    if "fiber_id" not in review.columns:
        raise ValueError("Review table must contain 'fiber_id'.")

    out = fibers.copy()
    out["fiber_id"] = out[id_col].astype(int)
    if "predicted_type" not in out.columns:
        if "fiber_type" in out.columns:
            out["predicted_type"] = out["fiber_type"].astype(str)
        else:
            out["predicted_type"] = "unknown"
    out["predicted_internal_type"] = out["predicted_type"].astype(str)
    out["predicted_biological_type"] = out["predicted_internal_type"].map(to_biological_label)

    review_cols = [
        c
        for c in (
            "fiber_id",
            "corrected_type",
            "is_uncertain",
            "is_hybrid",
            "is_excluded",
            "label_source",
        )
        if c in review.columns
    ]
    review_small = review[review_cols].copy()
    review_small["fiber_id"] = review_small["fiber_id"].astype(int)

    out = out.drop(
        columns=["corrected_type", "is_uncertain", "is_hybrid", "is_excluded", "label_source"],
        errors="ignore",
    ).merge(review_small, on="fiber_id", how="left")

    out["corrected_type"] = out["corrected_type"].fillna("").map(normalize_review_label)
    out["is_uncertain"] = out["is_uncertain"].fillna(False).astype(bool)
    out["is_hybrid"] = out["is_hybrid"].fillna(False).astype(bool)
    out["is_excluded"] = out["is_excluded"].fillna(False).astype(bool)
    out["label_source"] = out["label_source"].fillna("auto_rule")

    has_manual = out["label_source"].eq("manual_gold") & out["corrected_type"].astype(str).ne("")
    out["final_type"] = out["predicted_biological_type"].astype(str)
    out.loc[has_manual, "final_type"] = out.loc[has_manual, "corrected_type"].astype(str)
    out.loc[out["is_uncertain"], "final_type"] = "uncertain"
    out.loc[out["is_hybrid"], "final_type"] = "hybrid"
    out.loc[out["is_excluded"], "final_type"] = "exclude"
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge napari manual review labels into a fiber CSV."
    )
    parser.add_argument("--fibers", required=True, type=Path, help="Original per-fiber CSV.")
    parser.add_argument("--review", required=True, type=Path, help="Manual review CSV from napari.")
    parser.add_argument("--output", type=Path, help="Reviewed output CSV.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output or args.fibers.with_name(f"{args.fibers.stem}_reviewed.csv")
    fibers = pd.read_csv(args.fibers)
    review = pd.read_csv(args.review)
    merged = merge_reviewed_labels(fibers, review)
    save_dataframe(output, merged)

    n_manual = int((merged["label_source"] == "manual_gold").sum())
    n_uncertain = int(merged["is_uncertain"].sum())
    n_hybrid = int(merged["is_hybrid"].sum())
    n_excluded = int(merged["is_excluded"].sum())
    print(f"saved reviewed fibers: {output}")
    print(
        "review counts: "
        f"manual_gold={n_manual}, uncertain={n_uncertain}, hybrid={n_hybrid}, excluded={n_excluded}"
    )
    print("final_type counts:")
    print(merged["final_type"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
