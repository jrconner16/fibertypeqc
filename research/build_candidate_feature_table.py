from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.quantify_classify import FROZEN_ALPHA_BASELINE_FEATURES

METADATA_COLUMNS: tuple[str, ...] = (
    "image_id",
    "feature_diagnostics_path",
    "label",
    "fiber_type",
    "fiber_type_source",
    "classification_method",
    "available_markers",
    "needs_review",
    "model_confidence",
    "model_margin",
    "prob_iib",
    "prob_iia",
    "prob_iix",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble per-image *_feature_diagnostics.csv files into a single candidate-model "
            "feature table. Intended for model-development workflows only."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing *_feature_diagnostics.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path for the assembled candidate feature table.",
    )
    parser.add_argument(
        "--diagnostics-glob",
        type=str,
        default="**/*_feature_diagnostics.csv",
        help="Recursive glob used to find diagnostics files under --input-root.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional image-level metadata CSV. Must include an 'image_id' column and will be "
            "left-joined onto the assembled per-fiber table."
        ),
    )
    return parser


def _image_id_from_path(path: Path) -> str:
    name = path.name
    suffix = "_feature_diagnostics.csv"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def _normalize_image_id(value: str) -> str:
    return str(value).strip().lower()


def _load_manifest(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    manifest = pd.read_csv(path)
    if "image_id" not in manifest.columns:
        raise ValueError("Manifest must include an 'image_id' column.")
    manifest = manifest.copy()
    manifest["_merge_image_id"] = manifest["image_id"].map(_normalize_image_id)
    if manifest["_merge_image_id"].duplicated().any():
        dupes = sorted(
            manifest.loc[manifest["_merge_image_id"].duplicated(), "image_id"]
            .astype(str)
            .unique()
        )
        raise ValueError(f"Manifest contains duplicate image_id values: {', '.join(dupes[:5])}")
    return manifest


def _prepare_manifest(manifest: pd.DataFrame | None) -> pd.DataFrame | None:
    if manifest is None:
        return None
    if "_merge_image_id" in manifest.columns:
        return manifest
    if "image_id" not in manifest.columns:
        raise ValueError("Manifest must include an 'image_id' column.")
    manifest = manifest.copy()
    manifest["_merge_image_id"] = manifest["image_id"].map(_normalize_image_id)
    if manifest["_merge_image_id"].duplicated().any():
        dupes = sorted(
            manifest.loc[manifest["_merge_image_id"].duplicated(), "image_id"]
            .astype(str)
            .unique()
        )
        raise ValueError(f"Manifest contains duplicate image_id values: {', '.join(dupes[:5])}")
    return manifest


def discover_diagnostics_files(input_root: Path, diagnostics_glob: str) -> list[Path]:
    files = sorted(input_root.glob(diagnostics_glob))
    return [path for path in files if path.is_file()]


def assemble_candidate_feature_table(
    diagnostics_files: list[Path],
    manifest: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if not diagnostics_files:
        raise ValueError("No *_feature_diagnostics.csv files were found.")

    frames: list[pd.DataFrame] = []
    for path in diagnostics_files:
        df = pd.read_csv(path)
        image_id = _image_id_from_path(path)
        df.insert(0, "_merge_image_id", _normalize_image_id(image_id))
        df.insert(0, "feature_diagnostics_path", str(path.resolve()))
        df.insert(0, "image_id", image_id)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    manifest = _prepare_manifest(manifest)
    if manifest is not None:
        out = out.merge(manifest, on="_merge_image_id", how="left", validate="many_to_one")
        if "image_id_y" in out.columns:
            out["image_id"] = out["image_id_y"].fillna(out["image_id_x"])
            out = out.drop(columns=["image_id_x", "image_id_y"])
        out = out.drop(columns=["_merge_image_id"])
    else:
        out = out.drop(columns=["_merge_image_id"])

    metadata_cols = [col for col in METADATA_COLUMNS if col in out.columns]
    baseline_cols = [col for col in FROZEN_ALPHA_BASELINE_FEATURES if col in out.columns]
    remaining_cols = [
        col for col in out.columns if col not in set(metadata_cols) | set(baseline_cols)
    ]
    ordered_cols = metadata_cols + baseline_cols + sorted(remaining_cols)
    return out.loc[:, ordered_cols]


def main() -> None:
    args = build_parser().parse_args()
    diagnostics_files = discover_diagnostics_files(args.input_root, args.diagnostics_glob)
    manifest = _load_manifest(args.manifest)
    table = assemble_candidate_feature_table(diagnostics_files, manifest)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    baseline_cols = [col for col in FROZEN_ALPHA_BASELINE_FEATURES if col in table.columns]
    experimental_cols = [
        col
        for col in table.columns
        if col not in set(METADATA_COLUMNS) | set(FROZEN_ALPHA_BASELINE_FEATURES)
    ]

    print(f"diagnostics files: {len(diagnostics_files)}")
    print(f"rows: {len(table)}")
    print(f"images: {table['image_id'].nunique()}")
    print(f"baseline feature columns: {len(baseline_cols)}")
    print(f"experimental/auxiliary columns: {len(experimental_cols)}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
