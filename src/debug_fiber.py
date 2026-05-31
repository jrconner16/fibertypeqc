from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _fiber_id_column(df: pd.DataFrame) -> str:
    if "fiber_id" in df.columns:
        return "fiber_id"
    if "label" in df.columns:
        return "label"
    raise ValueError("Fiber table must contain either 'fiber_id' or 'label'.")


def _format_value(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def debug_fiber(fibers: pd.DataFrame, fiber_id: int) -> str:
    id_col = _fiber_id_column(fibers)
    rows = fibers[fibers[id_col].astype(int) == int(fiber_id)]
    if rows.empty:
        raise ValueError(f"fiber_id={fiber_id} not found in table.")
    row = rows.iloc[0]

    sections = [
        (
            "identity",
            ["label", "fiber_id", "fiber_type", "predicted_type", "corrected_type", "final_type"],
        ),
        (
            "review",
            [
                "confidence",
                "needs_review",
                "is_uncertain",
                "is_hybrid",
                "is_excluded",
                "label_source",
            ],
        ),
        (
            "type1",
            [
                "type1_mean",
                "type1_pctl",
                "type1_coverage",
                "type1_threshold",
                "type1_pctl_threshold",
            ],
        ),
        (
            "type2",
            [
                "type2_mean",
                "type2_pctl",
                "type2_coverage",
                "type2_threshold",
                "type2_pctl_threshold",
            ],
        ),
        ("scores", ["score_type1", "score_type2", "type1_cov_threshold", "type2_cov_threshold"]),
        ("geometry", ["area", "area_um2", "typing_interior_area", "typing_erode_px"]),
        ("location", ["centroid_x", "centroid_y"]),
    ]

    lines = [f"fiber_id={fiber_id}"]
    for title, cols in sections:
        present = [c for c in cols if c in row.index]
        if not present:
            continue
        lines.append(f"\n{title}:")
        for col in present:
            lines.append(f"  {col}: {_format_value(row[col])}")

    if {"type1_mean", "type1_threshold", "type2_mean", "type2_threshold"}.issubset(row.index):
        t1_ratio = row["type1_mean"] / max(float(row["type1_threshold"]), 1e-9)
        t2_ratio = row["type2_mean"] / max(float(row["type2_threshold"]), 1e-9)
        lines.append("\nthreshold ratios:")
        lines.append(f"  type1_mean / type1_threshold: {t1_ratio:.3f}")
        lines.append(f"  type2_mean / type2_threshold: {t2_ratio:.3f}")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print classifier/debug features for one fiber.")
    parser.add_argument("--fibers", required=True, type=Path, help="Per-fiber CSV.")
    parser.add_argument(
        "--fiber-id",
        required=True,
        type=int,
        help="Fiber label/fiber_id to inspect.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    fibers = pd.read_csv(args.fibers)
    print(debug_fiber(fibers, args.fiber_id))


if __name__ == "__main__":
    main()
