"""Run one canonical raw-CZI QUAD semantic-model task from a CSV manifest."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from fibertypeqc.czi_scenes import discover_czi_scenes, export_czi_scenes


@dataclass(frozen=True)
class ManifestRow:
    image_id: str
    input_relpath: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--task-index", type=int, required=True, help="Zero-based manifest row index."
    )
    parser.add_argument("--panel-config", type=Path, required=True)
    parser.add_argument("--classifier-path", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--fiber-downsample-factor", type=int, default=2)
    parser.add_argument("--fiber-diameter", type=float, default=30.0)
    return parser.parse_args()


def _load_manifest(path: Path) -> list[ManifestRow]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "input_relpath"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: image_id, input_relpath")
        rows: list[ManifestRow] = []
        seen_ids: set[str] = set()
        for line_number, row in enumerate(reader, start=2):
            image_id = str(row["image_id"] or "").strip()
            relpath = Path(str(row["input_relpath"] or "").strip())
            if not image_id:
                raise ValueError(f"{path}:{line_number} has a blank image_id")
            if image_id in seen_ids:
                raise ValueError(f"{path}:{line_number} has duplicate image_id: {image_id}")
            if not relpath.parts or relpath.is_absolute() or ".." in relpath.parts:
                raise ValueError(f"{path}:{line_number} has unsafe input_relpath: {relpath}")
            if relpath.suffix.lower() != ".czi":
                raise ValueError(f"{path}:{line_number} must reference a raw .czi file: {relpath}")
            rows.append(ManifestRow(image_id=image_id, input_relpath=relpath))
            seen_ids.add(image_id)
    if not rows:
        raise ValueError(f"{path} contains no input rows")
    return rows


def _section_inputs(
    input_path: Path, image_id: str, scene_export_root: Path
) -> list[tuple[str, Path]]:
    """Return direct single-scene input or deterministic fresh TIFF scene exports."""
    scenes = discover_czi_scenes(input_path)
    if not scenes:
        return [(image_id, input_path)]
    scene_dir = scene_export_root / image_id
    expected = len(scenes)
    existing = sorted(scene_dir.glob(f"{input_path.stem}_section-*.tif"))
    if existing and len(existing) != expected:
        raise ValueError(
            f"Incomplete scene export for {input_path}: expected {expected}, found {len(existing)}."
        )
    scene_paths = existing or export_czi_scenes(input_path, scene_dir)
    return [
        (f"{image_id}_section-{number:02d}", scene_path)
        for number, scene_path in enumerate(scene_paths, start=1)
    ]


def build_pipeline_command(
    *,
    input_path: Path,
    output_dir: Path,
    panel_config: Path,
    classifier_path: Path,
    model_manifest: Path,
    fiber_downsample_factor: int,
    fiber_diameter: float,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.run_pipeline",
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--panel-config",
        str(panel_config),
        "--classifier-path",
        str(classifier_path),
        "--model-manifest",
        str(model_manifest),
        "--requested-domain",
        "fiber_identity",
        "--downsample-factor",
        str(fiber_downsample_factor),
        "--diameter",
        str(fiber_diameter),
        "--export-diagnostics",
    ]


def main() -> None:
    args = _parse_args()
    rows = _load_manifest(args.manifest)
    if not 0 <= args.task_index < len(rows):
        raise ValueError(f"--task-index must be in [0, {len(rows) - 1}], got {args.task_index}")
    for path, label in (
        (args.panel_config, "panel config"),
        (args.classifier_path, "classifier"),
        (args.model_manifest, "model manifest"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    row = rows[args.task_index]
    input_path = args.input_root / row.input_relpath
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for section_id, section_path in _section_inputs(
        input_path, row.image_id, args.output_root / "raw_scene_exports"
    ):
        output_dir = args.output_root / section_id
        output_dir.mkdir(parents=True, exist_ok=True)
        command = build_pipeline_command(
            input_path=section_path,
            output_dir=output_dir,
            panel_config=args.panel_config,
            classifier_path=args.classifier_path,
            model_manifest=args.model_manifest,
            fiber_downsample_factor=args.fiber_downsample_factor,
            fiber_diameter=args.fiber_diameter,
        )
        print("running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
