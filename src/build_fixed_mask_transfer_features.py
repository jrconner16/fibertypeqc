"""Join audited one-to-one ROI labels to current diagnostics on immutable masks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from fibertypeqc.experimental_features import RELATIVE_CHANNEL_FEATURES

TARGET_CLASSES = frozenset({"iia", "iix", "iib"})
REQUIRED_F1_FEATURES = tuple(
    f"type_{marker}.{suffix}"
    for marker in ("iia", "iib")
    for suffix in (
        "mean",
        "p90",
        "coverage_high",
        "snr_mean",
        "snr_p90",
        "center_mean",
        "edge_mean",
    )
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a development-transfer feature table from one-to-one MyoSight matches and "
            "current diagnostics measured on unchanged pipeline masks."
        )
    )
    parser.add_argument("--all-rois", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=None)
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _discover_diagnostics(root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    suffix = "_feature_diagnostics.csv"
    for path in sorted(root.rglob(f"*{suffix}")):
        image_id = path.name.removesuffix(suffix)
        key = image_id.casefold()
        if key in paths:
            raise ValueError(f"Multiple diagnostics files found for image_id {image_id!r}.")
        paths[key] = path
    return paths


def build_transfer_features(
    all_rois: pd.DataFrame,
    diagnostics_paths: dict[str, Path],
    *,
    expected_images: int | None = None,
) -> pd.DataFrame:
    diagnostics_by_casefold = {
        str(key).casefold(): value for key, value in diagnostics_paths.items()
    }
    required = {
        "image_id",
        "roi_index",
        "label_id",
        "myosight_label",
        "one_to_one_eligible",
        "match_outcome",
    }
    missing = sorted(required.difference(all_rois.columns))
    if missing:
        raise ValueError(f"All-ROI audit missing required columns: {missing}")
    image_ids = sorted(all_rois["image_id"].astype(str).unique())
    if expected_images is not None and len(image_ids) != expected_images:
        raise ValueError(f"Expected {expected_images} audited images, found {len(image_ids)}.")
    missing_diagnostics = sorted(
        image_id for image_id in image_ids if image_id.casefold() not in diagnostics_by_casefold
    )
    if missing_diagnostics:
        raise ValueError(f"Missing diagnostics for images: {missing_diagnostics[:5]}")

    eligible = all_rois.loc[
        all_rois["one_to_one_eligible"].astype(bool)
        & all_rois["match_outcome"].eq("matched_one_to_one")
        & all_rois["myosight_label"].astype(str).str.lower().isin(TARGET_CLASSES)
    ].copy()
    eligible["pipeline_label_id"] = pd.to_numeric(eligible["label_id"], errors="raise").astype(int)
    if eligible.duplicated(["image_id", "pipeline_label_id"]).any():
        raise ValueError("Eligible ROI audit still contains duplicate pipeline-label assignments.")

    frames: list[pd.DataFrame] = []
    for image_id in image_ids:
        path = diagnostics_by_casefold[image_id.casefold()]
        diagnostics = pd.read_csv(path)
        missing_features = sorted(set(REQUIRED_F1_FEATURES).difference(diagnostics.columns))
        if missing_features:
            raise ValueError(f"Diagnostics for {image_id} missing F1 features: {missing_features}")
        if "label" not in diagnostics.columns:
            raise ValueError(f"Diagnostics for {image_id} missing label.")
        diagnostics = diagnostics.copy()
        diagnostics["pipeline_label_id"] = pd.to_numeric(
            diagnostics["label"], errors="raise"
        ).astype(int)
        if diagnostics["pipeline_label_id"].duplicated().any():
            raise ValueError(f"Diagnostics for {image_id} contain duplicate labels.")
        diagnostics.insert(0, "image_id", image_id)
        frames.append(diagnostics.drop(columns="label"))
    diagnostics = pd.concat(frames, ignore_index=True)

    metadata_columns = [
        column
        for column in ["image_id", "roi_index", "pipeline_label_id", "myosight_label"]
        if column in eligible.columns
    ]
    joined = eligible.loc[:, metadata_columns].merge(
        diagnostics,
        on=["image_id", "pipeline_label_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing_rows = joined.loc[joined["_merge"] != "both", ["image_id", "pipeline_label_id"]]
        raise ValueError(
            f"Eligible matches lack diagnostics: {missing_rows.head().to_dict('records')}"
        )
    joined = joined.drop(columns="_merge")
    joined.insert(1, "group_id", joined["image_id"])
    joined = joined.rename(columns={"myosight_label": "target_label"})
    joined.insert(4, "label_authority", "myosight_derived")
    joined.insert(5, "evidence_role", "development_transfer")
    return joined


def main() -> None:
    args = build_parser().parse_args()
    for output in [args.output, args.provenance_output]:
        if output.exists():
            raise SystemExit(f"Refusing to overwrite existing output: {output}")
    diagnostics_paths = _discover_diagnostics(args.diagnostics_root)
    table = build_transfer_features(
        pd.read_csv(args.all_rois),
        diagnostics_paths,
        expected_images=args.expected_images,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    provenance = {
        "schema_version": 1,
        "evidence_role": "development_transfer",
        "label_authority": "myosight_derived",
        "segmentation_rerun": False,
        "myosight_labels_modified": False,
        "images": int(table["image_id"].nunique()),
        "rows": int(len(table)),
        "target_counts": {
            str(key): int(value) for key, value in table["target_label"].value_counts().items()
        },
        "all_rois_sha256": _sha256(args.all_rois),
        "diagnostics_sha256": {
            image_id: _sha256(path) for image_id, path in sorted(diagnostics_paths.items())
        },
        "required_f1_features": list(REQUIRED_F1_FEATURES),
        "derived_f2_features": list(RELATIVE_CHANNEL_FEATURES),
    }
    args.provenance_output.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"transfer images: {table['image_id'].nunique()}")
    print(f"one-to-one three-class rows: {len(table)}")
    print(f"target counts: {table['target_label'].value_counts().to_dict()}")
    print(f"saved feature table: {args.output}")
    print(f"saved provenance: {args.provenance_output}")


if __name__ == "__main__":
    main()
