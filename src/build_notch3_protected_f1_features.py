"""Join protected Notch3 one-to-one MyoSight rows to frozen F1 diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.build_existing_ta_audit_f1_features import F1_COLUMNS

TARGET_CLASSES = {"iia", "iib", "iix"}


def _true_mask(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    unexpected = sorted(set(normalized).difference({"true", "false"}))
    if unexpected:
        raise ValueError(f"one_to_one_eligible contains invalid boolean values: {unexpected}")
    return normalized.eq("true")


def build_features(all_rois: pd.DataFrame, diagnostics_root: Path) -> pd.DataFrame:
    required = {
        "image_id",
        "group_id",
        "roi_index",
        "label_id",
        "myosight_label",
        "one_to_one_eligible",
        "match_outcome",
    }
    missing = sorted(required.difference(all_rois.columns))
    if missing:
        raise ValueError(f"All-ROI table missing columns: {missing}")
    selected = all_rois.loc[
        _true_mask(all_rois["one_to_one_eligible"])
        & all_rois["match_outcome"].eq("matched_one_to_one")
        & all_rois["myosight_label"].astype(str).str.lower().isin(TARGET_CLASSES)
    ].copy()
    selected["pipeline_label_id"] = pd.to_numeric(selected["label_id"], errors="raise").astype(int)
    if selected.duplicated(["image_id", "pipeline_label_id"]).any():
        raise ValueError("Eligible protected rows have duplicate image/pipeline-label keys.")

    frames: list[pd.DataFrame] = []
    for image_id, rows in selected.groupby("image_id", sort=True):
        diagnostic = next(
            (diagnostics_root / str(image_id)).glob("*_feature_diagnostics.csv"), None
        )
        if diagnostic is None:
            raise FileNotFoundError(f"Missing frozen diagnostics for {image_id}.")
        diagnostics = pd.read_csv(diagnostic, low_memory=False)
        missing_features = sorted(set(F1_COLUMNS).difference(diagnostics.columns))
        if missing_features:
            raise ValueError(f"Diagnostics for {image_id} missing F1 columns: {missing_features}")
        diagnostics["pipeline_label_id"] = pd.to_numeric(
            diagnostics["label"], errors="raise"
        ).astype(int)
        merged = rows.loc[
            :, ["image_id", "group_id", "roi_index", "pipeline_label_id", "myosight_label"]
        ].merge(
            diagnostics.loc[:, ["pipeline_label_id", *F1_COLUMNS]],
            on="pipeline_label_id",
            how="left",
            validate="one_to_one",
        )
        if merged[F1_COLUMNS].isna().any().any():
            raise ValueError(f"Protected matches lack diagnostics for {image_id}.")
        frames.append(merged)
    output = pd.concat(frames, ignore_index=True).rename(columns={"myosight_label": "target_label"})
    output.insert(0, "cohort_id", "notch3_ta")
    output.insert(5, "label_authority", "reviewed_myosight")
    output.insert(6, "evidence_role", "protected_final_evaluation")
    output.insert(7, "supervision_role", "protected_final_evaluation")
    output.insert(8, "eligible_for_model_fitting", False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-rois", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=22)
    parser.add_argument("--expected-groups", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists() or args.provenance_output.exists():
        raise FileExistsError("Refusing to overwrite an existing output or provenance file.")
    output = build_features(pd.read_csv(args.all_rois), args.diagnostics_root)
    if output["image_id"].nunique() != args.expected_images:
        raise ValueError(
            f"Expected {args.expected_images} images; found {output['image_id'].nunique()}."
        )
    if output["group_id"].nunique() != args.expected_groups:
        raise ValueError(
            f"Expected {args.expected_groups} groups; found {output['group_id'].nunique()}."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    provenance = {
        "images": int(output["image_id"].nunique()),
        "groups": int(output["group_id"].nunique()),
        "rows": len(output),
        "protected_rows_fit": 0,
    }
    args.provenance_output.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"protected rows: {len(output)}")
    print(f"protected groups: {output['group_id'].nunique()}")


if __name__ == "__main__":
    main()
