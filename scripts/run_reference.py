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
    run_reference_finalization(resolved_output_dir)
    run_four_class_reference(output_dir / "four_class")


def run_reference_finalization(output_dir: Path) -> None:
    """Finalize the reference run through a review project and check it agrees with merge."""
    # Review state must live outside prediction directories (enforced by load_project).
    project_dir = output_dir.parent / f"{output_dir.name}_review_project"
    project_dir.mkdir(parents=True, exist_ok=True)
    project = {
        "schema_version": "review_project.v1",
        "project_id": "synthetic_reference",
        "project_name": "Synthetic reference",
        "panel_manifest": str(REPO_ROOT / "examples/reference/panel.yaml"),
        "model_version": "rebaseline_tile_v2_p75p90_iib_iia_iix",
        "images": [
            {
                "image_id": "synthetic_reference",
                "mouse_id": "synthetic_mouse",
                "section_id": "s1",
                "raw_image_path": str(REPO_ROOT / "examples/reference/synthetic_reference.tif"),
                "prediction_directory": str(output_dir),
                "outputs": {
                    "fiber_table": "synthetic_reference_fibers.csv",
                    "fiber_labels": "synthetic_reference_cellpose_labels.tif",
                },
                "applicable_domains": ["fiber_segmentation", "fiber_typing"],
            }
        ],
    }
    (project_dir / "project.yaml").write_text(yaml.safe_dump(project), encoding="utf-8")
    review_dir = project_dir / "review"
    review_dir.mkdir(exist_ok=True)
    (review_dir / "review_state.json").write_text(
        json.dumps(
            {
                "schema_version": "review_state.v1",
                "project_id": "synthetic_reference",
                "model_version": "rebaseline_tile_v2_p75p90_iib_iia_iix",
                "active_domain": "fiber_typing",
                "active_scope": "image",
                "active_review_mode": "flagged_review",
            }
        ),
        encoding="utf-8",
    )
    final_dir = project_dir / "finalized"
    command = [
        sys.executable,
        "-m",
        "scripts.finalize_review_project",
        "--project",
        str(project_dir / "project.yaml"),
        "--output-dir",
        str(final_dir),
        "--legacy-review",
        f"synthetic_reference={REPO_ROOT / 'examples/reference/review_corrections.csv'}",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    finalized = pd.read_csv(
        final_dir / "synthetic_reference_fibers_finalized.csv", keep_default_na=False
    ).set_index("label")
    merged = pd.read_csv(output_dir / "synthetic_reference_fibers_final.csv").set_index("fiber_id")
    for label, merged_type in merged["final_type"].items():
        row = finalized.loc[label]
        expected = {"uncertain": ("", "unresolved"), "exclude": ("", "excluded")}.get(
            merged_type, (merged_type, row["value_source"])
        )
        if (row["final_type"], row["value_source"]) != expected:
            raise ValueError(f"Finalized reference fiber {label} disagrees with merge output.")
        if row["fiber_type"] != merged.loc[label, "predicted_type"]:
            raise ValueError(f"Finalized reference fiber {label} lost its model prediction.")
    run_reference_results(project_dir, final_dir)


def run_reference_results(project_dir: Path, final_dir: Path) -> None:
    """Summarize the finalized reference and check the composition definitions."""
    results_dir = project_dir / "results"
    command = [
        sys.executable,
        "-m",
        "scripts.summarize_results",
        "--final-dir",
        str(final_dir),
        "--project",
        str(project_dir / "project.yaml"),
        "--output-dir",
        str(results_dir),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    mouse = pd.read_csv(results_dir / "mouse_summary.csv").iloc[0]
    # 9 fibers: 1 excluded by the reviewer, 1 unresolved, 7 resolved (4 IIa, 3 IIb); the model
    # called 6 of the 8 analysis fibers IIa.
    expected = {
        "n_fibers_total": 9,
        "n_excluded": 1,
        "n_unresolved": 1,
        "n_resolved": 7,
        "prop_final_iia": 4 / 7,
        "prop_predicted_iia": 6 / 8,
    }
    for column, value in expected.items():
        if abs(float(mouse[column]) - value) > 1e-9:
            raise ValueError(f"Reference result {column} is {mouse[column]}, expected {value}.")
    if not (results_dir / "cohort_report.html").is_file():
        raise ValueError("Reference cohort report was not written.")


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
