"""Generate the small, public-safe deterministic reference TIFF and label mask."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile


def generate_fixture(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((3, 96, 96), 100, dtype=np.uint16)
    labels = np.zeros((96, 96), dtype=np.uint16)

    label_id = 0
    for row, y0 in enumerate((5, 36, 67)):
        for col, x0 in enumerate((5, 36, 67)):
            label_id += 1
            region = np.s_[y0 : y0 + 24, x0 : x0 + 24]
            labels[region] = label_id

            # Direct IIb signal for the first row, direct IIa signal for the
            # second row, and low signal in both channels for residual IIx.
            if row == 0:
                image[0][region] = 3600 + col * 120
                image[1][region] = 180 + col * 10
            elif row == 1:
                image[0][region] = 180 + col * 10
                image[1][region] = 3600 + col * 120
            else:
                image[0][region] = 150 + col * 10
                image[1][region] = 160 + col * 10

            image[2, y0 : y0 + 24, x0] = 5000
            image[2, y0 : y0 + 24, x0 + 23] = 5000
            image[2, y0, x0 : x0 + 24] = 5000
            image[2, y0 + 23, x0 : x0 + 24] = 5000

    image_path = output_dir / "synthetic_reference.tif"
    labels_path = output_dir / "synthetic_reference_labels.tif"
    tifffile.imwrite(
        image_path,
        image,
        imagej=True,
        resolution=(2.0, 2.0),
        metadata={"axes": "CYX", "unit": "um"},
    )
    tifffile.imwrite(labels_path, labels, imagej=True, metadata={"axes": "YX"})
    return image_path, labels_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/reference"),
        help="Directory in which to write the deterministic fixture files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_path, labels_path = generate_fixture(args.output_dir)
    print(f"saved reference image: {image_path}")
    print(f"saved reference labels: {labels_path}")


if __name__ == "__main__":
    main()
