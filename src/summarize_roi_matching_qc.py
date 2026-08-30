from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import roifile

REQUIRED_MATCH_COLUMNS = {
    "roi_index",
    "label_id",
    "centroid_distance_px",
    "area_ratio",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit ROI matching outputs, exclude ambiguous many-to-one assignments, and "
            "summarize one-to-one match coverage per image."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Matching manifest containing image_id and a MyoSight results path.",
    )
    parser.add_argument(
        "--matched-root",
        type=Path,
        action="append",
        required=True,
        help=("Root searched recursively for <image_id>_matched_fibers.csv. May be repeated."),
    )
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any manifest image lacks a matching output.",
    )
    return parser


def audit_match_rows(matches: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_MATCH_COLUMNS.difference(matches.columns)
    if missing:
        raise ValueError(f"Matched fibers table missing required columns: {sorted(missing)}")

    out = matches.copy()
    out["roi_index"] = pd.to_numeric(out["roi_index"], errors="coerce")
    out["label_id"] = pd.to_numeric(out["label_id"], errors="coerce")

    invalid_identifier = (
        out["roi_index"].isna()
        | out["label_id"].isna()
        | (out["roi_index"] < 0)
        | (out["label_id"] <= 0)
        | (out["roi_index"] % 1 != 0)
        | (out["label_id"] % 1 != 0)
    )
    duplicate_roi = out["roi_index"].duplicated(keep=False) & ~invalid_identifier
    duplicate_label = out["label_id"].duplicated(keep=False) & ~invalid_identifier

    reasons: list[str] = []
    for invalid, roi_dup, label_dup in zip(
        invalid_identifier, duplicate_roi, duplicate_label, strict=True
    ):
        row_reasons = []
        if invalid:
            row_reasons.append("invalid_identifier")
        if roi_dup:
            row_reasons.append("duplicate_roi_index")
        if label_dup:
            row_reasons.append("duplicate_pipeline_label")
        reasons.append(";".join(row_reasons))

    out["duplicate_roi_index"] = duplicate_roi
    out["duplicate_pipeline_label"] = duplicate_label
    out["qc_exclusion_reason"] = reasons
    out["one_to_one_eligible"] = out["qc_exclusion_reason"] == ""
    return out


def summarize_image(
    audited: pd.DataFrame,
    *,
    total_rois: int,
    results_rows: int,
) -> dict[str, int | float | bool]:
    eligible = audited.loc[audited["one_to_one_eligible"]]
    raw_count = len(audited)
    eligible_count = len(eligible)
    duplicate_labels = audited.loc[audited["duplicate_pipeline_label"], "label_id"]

    def median(column: str) -> float:
        values = pd.to_numeric(eligible[column], errors="coerce").dropna()
        return float(values.median()) if not values.empty else float("nan")

    return {
        "total_myosight_rois": int(total_rois),
        "myosight_results_rows": int(results_rows),
        "roi_results_count_mismatch": bool(total_rois != results_rows),
        "raw_matched_rows": int(raw_count),
        "raw_match_rate": float(raw_count / total_rois) if total_rois else float("nan"),
        "one_to_one_eligible_rows": int(eligible_count),
        "one_to_one_match_rate": (
            float(eligible_count / total_rois) if total_rois else float("nan")
        ),
        "excluded_rows": int(raw_count - eligible_count),
        "duplicate_pipeline_label_ids": int(duplicate_labels.nunique()),
        "duplicate_pipeline_label_rows": int(audited["duplicate_pipeline_label"].sum()),
        "duplicate_roi_index_rows": int(audited["duplicate_roi_index"].sum()),
        "median_centroid_distance_px": median("centroid_distance_px"),
        "median_area_ratio": median("area_ratio"),
    }


def _resolve_existing_path(manifest_path: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        raise FileNotFoundError(candidate)

    for root in [manifest_path.parent, *manifest_path.parents]:
        resolved = root / candidate
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Could not resolve manifest path: {value}")


def _myosight_counts(manifest_path: Path, row: pd.Series) -> tuple[int, int]:
    if "myosight_results_dir" in row and pd.notna(row["myosight_results_dir"]):
        results_dir = _resolve_existing_path(manifest_path, str(row["myosight_results_dir"]))
    elif "myosight_results" in row and pd.notna(row["myosight_results"]):
        results_path = _resolve_existing_path(manifest_path, str(row["myosight_results"]))
        results_dir = results_path if results_path.is_dir() else results_path.parent
    else:
        raise ValueError(
            "Manifest must contain myosight_results_dir or myosight_results for ROI counts."
        )

    results = pd.read_csv(results_dir / "Results.txt", sep="\t")
    rois = roifile.roiread(results_dir / "ROISet.zip")
    return len(rois), len(results)


def _discover_matched_files(roots: list[Path]) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for root in roots:
        for path in root.rglob("*_matched_fibers.csv"):
            image_id = path.name.removesuffix("_matched_fibers.csv")
            previous = discovered.get(image_id)
            if previous is not None and previous.resolve() != path.resolve():
                raise ValueError(
                    f"Multiple matched files found for {image_id}: {previous} and {path}"
                )
            discovered[image_id] = path
    return discovered


def main() -> None:
    args = build_parser().parse_args()
    manifest = pd.read_csv(args.manifest)
    if "image_id" not in manifest.columns:
        raise ValueError("Manifest missing required column: image_id")
    if manifest["image_id"].duplicated().any():
        raise ValueError("Manifest image_id values must be unique.")

    matched_by_image = _discover_matched_files(args.matched_root)
    audited_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    missing_images: list[str] = []

    metadata_columns = [
        column
        for column in ["group_id", "mouse_id", "age", "genotype"]
        if column in manifest.columns
    ]
    for _, manifest_row in manifest.iterrows():
        image_id = str(manifest_row["image_id"])
        matched_path = matched_by_image.get(image_id)
        if matched_path is None:
            missing_images.append(image_id)
            continue

        matched = pd.read_csv(matched_path)
        audited = audit_match_rows(matched)
        audited.insert(0, "image_id", image_id)
        audited["matched_fibers_path"] = str(matched_path.resolve())
        audited_tables.append(audited)

        total_rois, results_rows = _myosight_counts(args.manifest, manifest_row)
        summary = summarize_image(
            audited,
            total_rois=total_rois,
            results_rows=results_rows,
        )
        summary_rows.append(
            {
                "image_id": image_id,
                **{column: manifest_row[column] for column in metadata_columns},
                **summary,
                "matched_fibers_path": str(matched_path.resolve()),
            }
        )

    if not audited_tables:
        raise SystemExit("No manifest images had matching output files.")
    if args.require_all and missing_images:
        preview = ", ".join(missing_images[:5])
        raise SystemExit(f"Missing matching outputs for {len(missing_images)} images: {preview}")

    audited_rows = pd.concat(audited_tables, ignore_index=True)
    summary_table = pd.DataFrame(summary_rows)
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    audited_rows.to_csv(args.rows_output, index=False)
    summary_table.to_csv(args.summary_output, index=False)

    print(f"manifest images: {len(manifest)}")
    print(f"summarized images: {len(summary_table)}")
    print(f"images without matching output: {len(missing_images)}")
    print(f"raw matched rows: {len(audited_rows)}")
    print(f"one-to-one eligible rows: {int(audited_rows['one_to_one_eligible'].sum())}")
    print(f"saved audited rows: {args.rows_output}")
    print(f"saved image summary: {args.summary_output}")


if __name__ == "__main__":
    main()
