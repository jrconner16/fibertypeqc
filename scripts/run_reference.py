"""Run and validate the deterministic frozen-baseline public reference workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

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
    run_four_class_reference(output_dir / "four_class")


FOUR_CLASS_DIR = Path("examples/reference_four_class")


def run_four_class_reference(output_dir: Path) -> None:
    """Run the synthetic four-class model through the pinned semantic path and validate it."""
    resolved_output_dir = output_dir if output_dir.is_absolute() else REPO_ROOT / output_dir
    command = [
        sys.executable,
        "-m",
        "scripts.run_pipeline",
        "--input",
        str(FOUR_CLASS_DIR / "synthetic_four_class.tif"),
        "--labels-path",
        str(FOUR_CLASS_DIR / "synthetic_four_class_labels.tif"),
        "--output-dir",
        str(output_dir),
        "--panel-config",
        str(FOUR_CLASS_DIR / "panel.yaml"),
        "--model",
        "synthetic_four_class_reference_v1",
        "--qc-min-labels",
        "16",
        "--bootstrap-reps",
        "50",
        "--cpu",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    validate_four_class_outputs(resolved_output_dir)


def validate_four_class_outputs(output_dir: Path) -> None:
    expected = json.loads((REPO_ROOT / FOUR_CLASS_DIR / "expected_fiber_types.json").read_text())
    fibers = pd.read_csv(output_dir / "synthetic_four_class_fibers.csv")
    actual = {
        str(label): str(call)
        for label, call in zip(fibers["label"], fibers["fiber_type"], strict=True)
    }
    if actual != expected["fiber_type_by_label"]:
        raise ValueError("Four-class reference fiber calls differ from the synthetic truth.")
    if set(fibers["classification_method"]) != {"semantic_model"}:
        raise ValueError("Four-class reference did not use the semantic model for fiber calls.")
    summary = pd.read_csv(output_dir / "synthetic_four_class_summary.csv").iloc[0]
    for fiber_class in ("i", "iia", "iib", "iix"):
        if abs(float(summary[f"prop_{fiber_class}"]) - 0.25) > 1e-9:
            raise ValueError(f"Four-class reference prop_{fiber_class} is not 0.25.")
    residual_rate = float(summary["residual_rate"])
    if summary["residual_target_class"] != "iix" or abs(residual_rate - 0.25) > 1e-9:
        raise ValueError("Four-class reference residual-class rate is incorrect.")
    run = json.loads((output_dir / "synthetic_four_class_run.json").read_text())
    manifest_path = REPO_ROOT / FOUR_CLASS_DIR / "synthetic_four_class_rf.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest_digest = manifest["artifact_sha256"]
    if run.get("classifier_sha256") != manifest_digest:
        raise ValueError("Four-class reference run did not record the verified model digest.")


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
