"""Export every true Zeiss scene from a multi-scene CZI as scalar TIFFs."""

from __future__ import annotations

import argparse
from pathlib import Path

from fibertypeqc.czi_scenes import discover_czi_scenes, export_czi_scenes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    scenes = discover_czi_scenes(args.input)
    if not scenes:
        raise SystemExit("Input does not contain multiple Zeiss scenes.")
    for scene in scenes:
        print(
            f"scene {scene.index}: {scene.name or 'unnamed'} "
            f"({scene.x_stop - scene.x_start} x {scene.y_stop - scene.y_start} px)"
        )
    for output in export_czi_scenes(args.input, args.output_dir):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
