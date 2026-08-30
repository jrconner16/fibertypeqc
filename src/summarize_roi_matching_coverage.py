from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.summarize_roi_matching_qc import (
    _resolve_existing_path,
    audit_match_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize pipeline-label counts, unmatched reasons, and MyoSight-class coverage "
            "from exhaustive ROI match-candidate outputs."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--all-rois-output", type=Path, required=True)
    parser.add_argument("--image-summary-output", type=Path, required=True)
    parser.add_argument("--class-summary-output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    return parser


def normalize_myosight_label(value: object) -> str:
    text = str(value).strip().lower().replace(" ", "")
    aliases = {
        "2a": "iia",
        "2b": "iib",
        "2x": "iix",
        "typeiia": "iia",
        "typeiib": "iib",
        "typeiix": "iix",
    }
    return aliases.get(text, text)


def apply_one_to_one_outcomes(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {"roi_index", "label_id", "centroid_distance_px", "area_ratio", "candidate_status"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table missing required columns: {sorted(missing)}")
    if candidates["roi_index"].duplicated().any():
        raise ValueError("Candidate table roi_index values must be unique.")

    out = candidates.copy()
    out["one_to_one_eligible"] = False
    out["match_outcome"] = out["candidate_status"].astype(str)

    raw = out.loc[out["candidate_status"] == "matched"].copy()
    if not raw.empty:
        audited = audit_match_rows(raw)
        eligible_by_roi = audited.set_index("roi_index")["one_to_one_eligible"]
        raw_mask = out["candidate_status"] == "matched"
        out.loc[raw_mask, "one_to_one_eligible"] = (
            out.loc[raw_mask, "roi_index"].map(eligible_by_roi).fillna(False).astype(bool)
        )
        ambiguous = raw_mask & ~out["one_to_one_eligible"]
        out.loc[out["one_to_one_eligible"], "match_outcome"] = "matched_one_to_one"
        out.loc[ambiguous, "match_outcome"] = "ambiguous_duplicate_assignment"
    return out


def build_class_summary(all_rois: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    scopes = [("overall", pd.Series("all", index=all_rois.index))]
    for column in ["genotype", "group_id"]:
        if column in all_rois.columns:
            scopes.append((column, all_rois[column].astype(str)))

    outcomes = [
        "matched_one_to_one",
        "no_pipeline_label_at_centroid",
        "centroid_distance_exceeds_threshold",
        "ambiguous_duplicate_assignment",
    ]
    for scope_type, scope_values in scopes:
        scoped = all_rois.assign(_scope_id=scope_values)
        grouped = scoped.groupby(["_scope_id", "myosight_label"], dropna=False)
        summary = grouped.size().rename("total_myosight_rois").reset_index()
        for outcome in outcomes:
            counts = (
                scoped.loc[scoped["match_outcome"] == outcome]
                .groupby(["_scope_id", "myosight_label"])
                .size()
                .rename(outcome)
                .reset_index()
            )
            summary = summary.merge(
                counts, on=["_scope_id", "myosight_label"], how="left", validate="one_to_one"
            )
        summary[outcomes] = summary[outcomes].fillna(0).astype(int)
        summary["one_to_one_coverage"] = (
            summary["matched_one_to_one"] / summary["total_myosight_rois"]
        )
        summary.insert(0, "scope_type", scope_type)
        summary = summary.rename(columns={"_scope_id": "scope_id"})
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def _results_dir(manifest_path: Path, row: pd.Series) -> Path:
    if "myosight_results_dir" in row and pd.notna(row["myosight_results_dir"]):
        path = _resolve_existing_path(manifest_path, str(row["myosight_results_dir"]))
    elif "myosight_results" in row and pd.notna(row["myosight_results"]):
        path = _resolve_existing_path(manifest_path, str(row["myosight_results"]))
    else:
        raise ValueError("Manifest lacks myosight_results_dir or myosight_results.")
    return path if path.is_dir() else path.parent


def _discover_sidecars(root: Path, suffix: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in root.rglob(f"*{suffix}"):
        image_id = path.name.removesuffix(suffix)
        if image_id in paths:
            raise ValueError(f"Multiple {suffix} files found for {image_id}")
        paths[image_id] = path
    return paths


def main() -> None:
    args = build_parser().parse_args()
    manifest = pd.read_csv(args.manifest)
    if "image_id" not in manifest.columns:
        raise ValueError("Manifest missing required column: image_id")
    if manifest["image_id"].duplicated().any():
        raise ValueError("Manifest image_id values must be unique.")

    candidate_paths = _discover_sidecars(args.candidate_root, "_match_candidates.csv")
    metadata_paths = _discover_sidecars(args.candidate_root, "_match_metadata.json")
    metadata_columns = [
        column
        for column in ["group_id", "mouse_id", "age", "genotype"]
        if column in manifest.columns
    ]
    all_roi_tables: list[pd.DataFrame] = []
    image_summaries: list[dict[str, object]] = []
    missing_images: list[str] = []

    for _, manifest_row in manifest.iterrows():
        image_id = str(manifest_row["image_id"])
        candidate_path = candidate_paths.get(image_id)
        metadata_path = metadata_paths.get(image_id)
        if candidate_path is None or metadata_path is None:
            missing_images.append(image_id)
            continue

        candidates = apply_one_to_one_outcomes(pd.read_csv(candidate_path))
        results = pd.read_csv(_results_dir(args.manifest, manifest_row) / "Results.txt", sep="\t")
        results = results.loc[:, [column for column in results.columns if str(column).strip()]]
        if "Label" not in results.columns:
            raise ValueError(f"MyoSight Results.txt lacks Label for {image_id}")
        if len(results) != len(candidates):
            raise ValueError(
                f"ROI/Results candidate count mismatch for {image_id}: "
                f"{len(candidates)} != {len(results)}"
            )

        candidates = candidates.sort_values("roi_index").reset_index(drop=True)
        expected_indices = pd.Series(range(len(candidates)), dtype=candidates["roi_index"].dtype)
        if not candidates["roi_index"].reset_index(drop=True).equals(expected_indices):
            raise ValueError(f"Candidate ROI indices are not contiguous for {image_id}")
        candidates.insert(0, "image_id", image_id)
        for column in metadata_columns:
            candidates[column] = manifest_row[column]
        candidates["myosight_label_raw"] = results["Label"].astype(str).to_numpy()
        candidates["myosight_label"] = candidates["myosight_label_raw"].map(
            normalize_myosight_label
        )
        candidates["candidate_path"] = str(candidate_path.resolve())
        all_roi_tables.append(candidates)

        metadata = json.loads(metadata_path.read_text())
        outcome_counts = candidates["match_outcome"].value_counts()
        total_rois = len(candidates)
        pipeline_count = int(metadata["pipeline_label_count"])
        image_summaries.append(
            {
                "image_id": image_id,
                **{column: manifest_row[column] for column in metadata_columns},
                "total_myosight_rois": total_rois,
                "pipeline_label_count": pipeline_count,
                "pipeline_to_myosight_count_ratio": pipeline_count / total_rois,
                "matched_one_to_one": int(outcome_counts.get("matched_one_to_one", 0)),
                "one_to_one_coverage": float(candidates["one_to_one_eligible"].mean()),
                "no_pipeline_label_at_centroid": int(
                    outcome_counts.get("no_pipeline_label_at_centroid", 0)
                ),
                "centroid_distance_exceeds_threshold": int(
                    outcome_counts.get("centroid_distance_exceeds_threshold", 0)
                ),
                "ambiguous_duplicate_assignment": int(
                    outcome_counts.get("ambiguous_duplicate_assignment", 0)
                ),
            }
        )

    if not all_roi_tables:
        raise SystemExit("No complete candidate/metadata pairs found for manifest images.")
    if args.require_all and missing_images:
        raise SystemExit(
            f"Missing candidate outputs for {len(missing_images)} images: "
            + ", ".join(missing_images[:5])
        )

    all_rois = pd.concat(all_roi_tables, ignore_index=True)
    image_summary = pd.DataFrame(image_summaries)
    class_summary = build_class_summary(all_rois)
    for path in [args.all_rois_output, args.image_summary_output, args.class_summary_output]:
        path.parent.mkdir(parents=True, exist_ok=True)
    all_rois.to_csv(args.all_rois_output, index=False)
    image_summary.to_csv(args.image_summary_output, index=False)
    class_summary.to_csv(args.class_summary_output, index=False)

    print(f"manifest images: {len(manifest)}")
    print(f"summarized images: {len(image_summary)}")
    print(f"images without candidate outputs: {len(missing_images)}")
    print(f"all MyoSight ROIs: {len(all_rois)}")
    print(f"one-to-one matches: {int(all_rois['one_to_one_eligible'].sum())}")
    print(f"saved all-ROI audit: {args.all_rois_output}")
    print(f"saved image summary: {args.image_summary_output}")
    print(f"saved class summary: {args.class_summary_output}")


if __name__ == "__main__":
    main()
