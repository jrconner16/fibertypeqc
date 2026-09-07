"""Build a private execution manifest for fixed-mask transfer characterization."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve a canonical image manifest to source images, cached pipeline labels, and "
            "MyoSight Results/ROI files without carrying historical split assignments forward."
        )
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=None)
    return parser


def _resolve_existing(path_root: Path, value: object, *, description: str) -> Path:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Blank {description} path.")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = path_root / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _one_file(directory: Path, pattern: str, *, description: str) -> Path:
    matches = sorted(path for path in directory.glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {description} under {directory}; found {len(matches)}."
        )
    return matches[0].resolve()


def build_fixed_mask_manifest(
    source: pd.DataFrame,
    *,
    path_root: Path,
    labels_root: Path,
    expected_images: int | None = None,
) -> pd.DataFrame:
    required = {"image_id", "myosight_summary_path"}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"Source manifest missing required columns: {missing}")
    if not ({"source_image", "input_path"} & set(source.columns)):
        raise ValueError("Source manifest requires source_image or input_path.")
    if source["image_id"].astype(str).str.strip().duplicated().any():
        raise ValueError("Source manifest image_id values must be unique.")
    if expected_images is not None and len(source) != expected_images:
        raise ValueError(f"Expected {expected_images} source images, found {len(source)}.")

    rows: list[dict[str, str]] = []
    for _, row in source.iterrows():
        image_id = str(row["image_id"]).strip()
        if not image_id:
            raise ValueError("Source manifest contains a blank image_id.")
        source_value = row.get("source_image")
        if pd.isna(source_value) or not str(source_value).strip():
            source_value = row.get("input_path")
        input_path = _resolve_existing(path_root, source_value, description="source image")
        results_path = _resolve_existing(
            path_root,
            row["myosight_summary_path"],
            description="MyoSight Results table",
        )
        results_dir = results_path if results_path.is_dir() else results_path.parent
        results_txt = results_dir / "Results.txt"
        roiset_zip = results_dir / "ROISet.zip"
        if not results_txt.is_file() or not roiset_zip.is_file():
            raise FileNotFoundError(
                f"MyoSight package for {image_id} requires Results.txt and ROISet.zip: "
                f"{results_dir}"
            )

        label_dir = labels_root / image_id
        labels_path = _one_file(
            label_dir,
            "*_cellpose_labels.tif",
            description="cached pipeline label TIFF",
        )
        rows.append(
            {
                "image_id": image_id,
                "input_path": str(input_path),
                "labels_path": str(labels_path),
                "myosight_results_dir": str(results_dir.resolve()),
                "evidence_role": "development_transfer",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser().parse_args()
    source = pd.read_csv(args.source_manifest)
    manifest = build_fixed_mask_manifest(
        source,
        path_root=args.path_root.resolve(),
        labels_root=args.labels_root.resolve(),
        expected_images=args.expected_images,
    )
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(f"source images: {len(source)}")
    print(f"fixed-mask rows: {len(manifest)}")
    print(f"evidence roles: {manifest['evidence_role'].value_counts().to_dict()}")
    print(f"saved manifest: {args.output}")


if __name__ == "__main__":
    main()
