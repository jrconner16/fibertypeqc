"""Run one panel-aware fiber and optional nuclear task from a private CSV manifest."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from fibertypeqc.config import load_channel_config


@dataclass(frozen=True)
class ManifestRow:
    image_id: str
    input_relpath: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one panel-aware fiber/nuclear task selected from a CSV manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--task-index",
        type=int,
        required=True,
        help="Zero-based manifest row index.",
    )
    parser.add_argument("--panel-config", type=Path, required=True)
    parser.add_argument("--fiber-downsample-factor", type=int, default=2)
    parser.add_argument("--fiber-diameter", type=float, default=30.0)
    parser.add_argument("--run-nuclei", action="store_true")
    parser.add_argument("--nuclei-downsample-factor", type=int, default=2)
    parser.add_argument("--nuclei-diameter", type=float, default=15.0)
    parser.add_argument("--nuclei-min-size", type=int, default=30)
    return parser.parse_args()


def _load_manifest(path: Path) -> list[ManifestRow]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"image_id", "input_relpath"}
        if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: image_id, input_relpath")

        rows: list[ManifestRow] = []
        for line_number, row in enumerate(reader, start=2):
            image_id = str(row["image_id"] or "").strip()
            raw_relpath = Path(str(row["input_relpath"] or "").strip())
            if not image_id:
                raise ValueError(f"{path}:{line_number} has a blank image_id")
            if not raw_relpath.parts or raw_relpath.is_absolute() or ".." in raw_relpath.parts:
                raise ValueError(f"{path}:{line_number} has unsafe input_relpath: {raw_relpath}")
            rows.append(ManifestRow(image_id=image_id, input_relpath=raw_relpath))
    if not rows:
        raise ValueError(f"{path} contains no input rows")
    return rows


def _run(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = _parse_args()
    rows = _load_manifest(args.manifest)
    if args.task_index < 0 or args.task_index >= len(rows):
        raise ValueError(f"--task-index must be in [0, {len(rows) - 1}], got {args.task_index}")

    row = rows[args.task_index]
    input_path = args.input_root / row.input_relpath
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    output_dir = args.output_root / row.image_id
    output_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            sys.executable,
            "-m",
            "src.run_pipeline",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--panel-config",
            str(args.panel_config),
            "--downsample-factor",
            str(args.fiber_downsample_factor),
            "--diameter",
            str(args.fiber_diameter),
            "--export-diagnostics",
            "--retain-mode",
            "full",
        ]
    )

    if not args.run_nuclei:
        return

    panel = load_channel_config(args.panel_config)
    if panel.dapi_channel is None:
        raise ValueError("--run-nuclei requires a DAPI channel in --panel-config")
    labels_path = output_dir / f"{input_path.stem.replace(' ', '_')}_cellpose_labels.tif"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Expected fiber labels were not produced: {labels_path}")

    _run(
        [
            sys.executable,
            "-m",
            "src.run_nuclear_stage",
            "--input",
            str(input_path),
            "--fiber-labels",
            str(labels_path),
            "--output-dir",
            str(output_dir / "nuclear"),
            "--dapi-channel",
            str(panel.dapi_channel),
            "--downsample-factor",
            str(args.nuclei_downsample_factor),
            "--diameter",
            str(args.nuclei_diameter),
            "--min-size",
            str(args.nuclei_min_size),
        ]
    )


if __name__ == "__main__":
    main()
