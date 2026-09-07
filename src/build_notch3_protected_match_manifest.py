"""Resolve protected Notch3 fixed-mask ROI-matching inputs from private SSD packages."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _one(paths: list[Path], description: str, image_id: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"Expected one {description} for {image_id}; found {len(paths)}.")
    return paths[0]


def build_manifest(
    split: pd.DataFrame,
    *,
    raw_root: Path,
    labels_root: Path,
    results_root: Path,
) -> pd.DataFrame:
    required = {"image_id", "group_id", "role", "raw_czi"}
    missing = sorted(required.difference(split.columns))
    if missing:
        raise ValueError(f"Split manifest missing columns: {missing}")
    protected = split.loc[split["role"].astype(str).eq("protected_holdout")].copy()
    if protected.empty:
        raise ValueError("No protected-holdout rows found.")
    if protected["image_id"].duplicated().any():
        raise ValueError("Protected split has duplicate image IDs.")
    raw_by_name = {path.name: path for path in raw_root.rglob("*.czi")}
    if len(raw_by_name) != len(list(raw_root.rglob("*.czi"))):
        raise ValueError("Raw CZI basenames are not unique under the supplied raw root.")

    rows: list[dict[str, object]] = []
    for _, row in protected.sort_values("image_id").iterrows():
        image_id = str(row["image_id"])
        raw_name = Path(str(row["raw_czi"])).name
        raw_path = raw_by_name.get(raw_name)
        if raw_path is None:
            raise FileNotFoundError(f"Missing raw CZI for {image_id}: {raw_name}")
        label_path = _one(
            list((labels_root / image_id).glob("*_cellpose_labels.tif")),
            "cached label TIFF",
            image_id,
        )
        results_dir = results_root / image_id
        for filename in ["Results.txt", "ROISet.zip"]:
            if not (results_dir / filename).is_file():
                raise FileNotFoundError(f"Missing {filename} for {image_id}: {results_dir}")
        rows.append(
            {
                "image_id": image_id,
                "group_id": str(row["group_id"]),
                "raw_czi": str(raw_path),
                "pipeline_labels": str(label_path),
                "myosight_results_dir": str(results_dir),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-split", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=22)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    output = build_manifest(
        pd.read_csv(args.image_split),
        raw_root=args.raw_root,
        labels_root=args.labels_root,
        results_root=args.results_root,
    )
    if len(output) != args.expected_images:
        raise ValueError(f"Expected {args.expected_images} protected images; found {len(output)}.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"protected images: {len(output)}")


if __name__ == "__main__":
    main()
