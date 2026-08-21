"""Run and validate the deterministic frozen-baseline public reference workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.validate_reference_outputs import validate_reference_outputs

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_reference(output_dir: Path) -> None:
    resolved_output_dir = output_dir if output_dir.is_absolute() else REPO_ROOT / output_dir

    pipeline_command = [
        sys.executable,
        "-m",
        "scripts.run_pipeline",
        "--input",
        "examples/reference/synthetic_reference.tif",
        "--labels-path",
        "examples/reference/synthetic_reference_labels.tif",
        "--output-dir",
        str(output_dir),
        "--panel-config",
        "examples/reference/panel.yaml",
        "--typing-preprocess",
        "tile_subtract",
        "--typing-tile-size",
        "256",
        "--typing-erode-px",
        "2",
        "--classifier-path",
        "data/models/rebaseline_tile_v2_p75p90_iib_iia_iix.joblib",
        "--model-manifest",
        "data/models/rebaseline_tile_v2_p75p90_iib_iia_iix.yaml",
        "--qc-min-labels",
        "9",
        "--bootstrap-reps",
        "50",
        "--cpu",
    ]
    subprocess.run(pipeline_command, cwd=REPO_ROOT, check=True)

    fibers = resolved_output_dir / "synthetic_reference_fibers.csv"
    reviewed = resolved_output_dir / "synthetic_reference_fibers_final.csv"
    merge_command = [
        sys.executable,
        "-m",
        "scripts.merge_reviewed_labels",
        "--fibers",
        str(fibers),
        "--review",
        "examples/reference/review_corrections.csv",
        "--panel-config",
        "examples/reference/panel.yaml",
        "--output",
        str(reviewed),
    ]
    subprocess.run(merge_command, cwd=REPO_ROOT, check=True)
    validate_reference_outputs(resolved_output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reference"),
        help="Output directory; defaults to outputs/reference.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_reference(args.output_dir)
    print(f"reference workflow passed: {args.output_dir}")


if __name__ == "__main__":
    main()
