from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.consolidate_reviewed_audit import _final_audit_label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply reviewed liberal IIb/IIx label-policy overrides to conservative benchmark "
            "and split tables, producing new liberal outputs without mutating the originals."
        )
    )
    parser.add_argument("--conservative-benchmark", type=Path, required=True)
    parser.add_argument("--conservative-split", type=Path, required=True)
    parser.add_argument("--reviewed-glob", type=str, required=True)
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=Path("outputs/validation"),
        help="Directory containing reviewed liberal queue CSVs.",
    )
    parser.add_argument("--liberal-benchmark-output", type=Path, required=True)
    parser.add_argument("--liberal-split-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument(
        "--label-policy",
        type=str,
        default="liberal_iib_iix_v1",
        help="Policy name recorded on the liberal outputs.",
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


def _load_review_overrides(review_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(path for path in review_dir.glob(pattern) if path.is_file())
    if not files:
        raise ValueError(f"No reviewed policy files found for pattern {pattern!r} in {review_dir}")
    frames = []
    for path in files:
        df = pd.read_csv(path, low_memory=False).copy()
        df["review_file"] = path.name
        frames.append(df)
    reviewed = pd.concat(frames, ignore_index=True)
    reviewed["image_id"] = reviewed["image_id"].astype(str)
    reviewed["label"] = reviewed["label"].astype(int)
    reviewed["liberal_audit_final_label"] = _final_audit_label(reviewed)
    reviewed = reviewed.sort_values(["image_id", "label", "review_file"], kind="stable")
    reviewed = reviewed.drop_duplicates(subset=["image_id", "label"], keep="last")
    reviewed = reviewed.loc[reviewed["liberal_audit_final_label"].astype(str).ne("")].copy()
    return reviewed


def apply_policy_overrides(
    conservative: pd.DataFrame, reviewed: pd.DataFrame, *, policy_name: str
) -> pd.DataFrame:
    override_cols = [
        "image_id",
        "label",
        "liberal_audit_final_label",
        "review_file",
        "audit_corrected_type",
        "audit_is_uncertain",
        "audit_is_excluded",
        "audit_notes",
    ]
    merged = conservative.merge(
        reviewed.loc[:, [col for col in override_cols if col in reviewed.columns]],
        on=["image_id", "label"],
        how="left",
        validate="one_to_one",
    )
    merged["conservative_audit_final_label"] = (
        merged.get("conservative_audit_final_label", merged["audit_final_label"]).fillna("").astype(str)
    )
    liberal = merged["liberal_audit_final_label"].fillna("").astype(str).str.lower().str.strip()
    has_override = liberal.ne("")
    merged["audit_final_label"] = merged["audit_final_label"].where(~has_override, liberal)
    merged["label_policy"] = policy_name
    merged["label_policy_changed"] = has_override & (
        merged["audit_final_label"].astype(str) != merged["conservative_audit_final_label"].astype(str)
    )
    merged["label_policy_review_file"] = merged["review_file_y"] if "review_file_y" in merged.columns else merged.get("review_file")
    if "review_file_y" in merged.columns and "review_file_x" in merged.columns:
        merged = merged.drop(columns=["review_file_x"])
        merged = merged.rename(columns={"review_file_y": "review_file"})
    if "liberal_audit_final_label" in merged.columns:
        merged["liberal_audit_final_label"] = liberal.where(has_override, merged["conservative_audit_final_label"])
    return merged


def build_summary(liberal_split: pd.DataFrame) -> str:
    changed = liberal_split.loc[literal_true(liberal_split["label_policy_changed"])].copy()
    lines = [
        "IIb/IIx label-policy update summary",
        "",
        f"rows in liberal split: {len(liberal_split)}",
        f"rows changed vs conservative: {len(changed)}",
        "",
        "changed rows by split:",
    ]
    if changed.empty:
        lines.append("none")
        return "\n".join(lines)
    lines.append(changed["manual_supervision_split"].value_counts().to_string())
    lines.extend(
        [
            "",
            "changed rows by conservative -> liberal label:",
            changed.groupby(["conservative_audit_final_label", "audit_final_label"])
            .size()
            .rename("count")
            .to_string(),
        ]
    )
    return "\n".join(lines)


def literal_true(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin(("true", "1", "yes"))


def main() -> None:
    args = build_parser().parse_args()
    conservative_benchmark = _load_table(args.conservative_benchmark)
    conservative_split = _load_table(args.conservative_split)
    reviewed = _load_review_overrides(args.review_dir, args.reviewed_glob)

    liberal_benchmark = apply_policy_overrides(
        conservative_benchmark, reviewed, policy_name=args.label_policy
    )
    liberal_split = apply_policy_overrides(
        conservative_split, reviewed, policy_name=args.label_policy
    )

    for path in (args.liberal_benchmark_output, args.liberal_split_output, args.summary_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    liberal_benchmark.to_csv(args.liberal_benchmark_output, index=False)
    liberal_split.to_csv(args.liberal_split_output, index=False)
    args.summary_output.write_text(build_summary(liberal_split), encoding="utf-8")

    print(f"saved liberal benchmark: {args.liberal_benchmark_output}")
    print(f"saved liberal split: {args.liberal_split_output}")
    print(f"saved summary: {args.summary_output}")


if __name__ == "__main__":
    main()
