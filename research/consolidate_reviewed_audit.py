from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate per-image reviewed audit CSVs into one reviewed-audit table and one "
            "adjudicated benchmark table."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing combined_audit_sample_reviewed_*.csv files.",
    )
    parser.add_argument(
        "--reviewed-output",
        type=Path,
        required=True,
        help="Output CSV path for the full consolidated reviewed-audit table.",
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        required=True,
        help="Output CSV path for adjudicated benchmark rows only.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="combined_audit_sample_reviewed_*.csv",
        help="Glob used to find reviewed audit CSVs under --input-dir.",
    )
    return parser


def _reviewed_files(input_dir: Path, pattern: str) -> list[Path]:
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def _normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin(("true", "1", "yes"))


def _final_audit_label(df: pd.DataFrame) -> pd.Series:
    corrected = df.get("audit_corrected_type", pd.Series("", index=df.index)).fillna("").astype(str)
    corrected = corrected.str.strip().str.lower()
    uncertain = _normalize_bool(df.get("audit_is_uncertain", pd.Series(False, index=df.index)))
    excluded = _normalize_bool(df.get("audit_is_excluded", pd.Series(False, index=df.index)))

    final = corrected.copy()
    final = final.where(final != "", "")
    final = final.mask(uncertain, "uncertain")
    final = final.mask(excluded, "exclude")
    return final


def consolidate_reviewed_audit(files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not files:
        raise ValueError("No reviewed audit CSV files were found.")

    frames: list[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path, low_memory=False).copy()
        df["review_file"] = path.name
        frames.append(df)

    reviewed = pd.concat(frames, ignore_index=True)
    reviewed["label"] = reviewed["label"].astype(int)
    reviewed["audit_final_label"] = _final_audit_label(reviewed)
    reviewed["audit_reviewed"] = reviewed["audit_final_label"].astype(str).ne("")
    reviewed["audit_label_source"] = reviewed["audit_reviewed"].map(
        {True: "manual_audit", False: "unreviewed"}
    )
    reviewed = reviewed.sort_values(["image_id", "label", "review_file"], kind="stable")
    reviewed = reviewed.drop_duplicates(subset=["image_id", "label"], keep="last")

    benchmark = reviewed.loc[reviewed["audit_reviewed"]].copy()
    benchmark = benchmark.sort_values(["image_id", "label"], kind="stable")
    reviewed = reviewed.sort_values(["image_id", "label"], kind="stable")
    return reviewed, benchmark


def main() -> None:
    args = build_parser().parse_args()
    files = _reviewed_files(args.input_dir, args.glob)
    reviewed, benchmark = consolidate_reviewed_audit(files)

    args.reviewed_output.parent.mkdir(parents=True, exist_ok=True)
    args.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
    reviewed.to_csv(args.reviewed_output, index=False)
    benchmark.to_csv(args.benchmark_output, index=False)

    print(f"review files: {len(files)}")
    print(f"reviewed rows total: {len(reviewed)}")
    print(f"benchmark rows: {len(benchmark)}")
    print(f"images in benchmark: {benchmark['image_id'].nunique() if not benchmark.empty else 0}")
    print("benchmark label counts:")
    if benchmark.empty:
        print("  none")
    else:
        counts = benchmark["audit_final_label"].value_counts().sort_index()
        for label, count in counts.items():
            print(f"  {label}: {int(count)}")
    print(f"saved reviewed table: {args.reviewed_output}")
    print(f"saved benchmark table: {args.benchmark_output}")


if __name__ == "__main__":
    main()
