"""Export every multi-scene CZI below a directory without running the pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from fibertypeqc.czi_scenes import discover_czi_scenes, export_czi_scenes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    inputs = sorted(
        path for path in args.input_root.rglob("*.czi") if "Split Scenes" not in path.parts
    )
    if not inputs:
        raise SystemExit(f"No raw CZI files found below {args.input_root}")

    exported = 0
    skipped = 0
    for path in inputs:
        scenes = discover_czi_scenes(path)
        if not scenes:
            skipped += 1
            print(f"skip single-scene CZI: {path}")
            continue
        relative_parent = path.parent.relative_to(args.input_root)
        output_dir = args.output_root / relative_parent / path.stem
        existing = sorted(output_dir.glob(f"{path.stem}_section-*.tif"))
        if len(existing) == len(scenes):
            skipped += 1
            print(f"reuse existing export: {output_dir}")
            continue
        if existing:
            raise SystemExit(
                f"Incomplete existing export for {path}: expected {len(scenes)}, "
                f"found {len(existing)} in {output_dir}"
            )
        outputs = export_czi_scenes(path, output_dir)
        exported += len(outputs)
        print(f"exported {len(outputs)} scenes: {path}")

    print(f"done: exported_sections={exported}, skipped_files={skipped}")


if __name__ == "__main__":
    main()
