"""Build and evaluate candidate models from a project's review decisions.

Features for the reviewed sections are recomputed with the current model's pinned settings on
the existing masks (no segmentation), then candidates are evaluated and saved under
``<output-dir>``. The active model is never changed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from fibertypeqc.artifacts import file_sha256
from fibertypeqc.model_resolution import default_model_id, resolve_model_argument
from src.model_improver import ImproverError, improve, load_recipe
from src.review.project import Project, load_project
from src.review.storage import load_session

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True, help="Review project YAML")
    parser.add_argument("--panel-config", type=Path, required=True)
    parser.add_argument("--model", default=None, help="Current model ID or file (default model)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder for improver runs")
    parser.add_argument("--recipe", type=Path, default=None, help="Improver recipe YAML")
    return parser


def extract_features(
    project: Project, image_id: str, model: str, panel: Path, cache: Path
) -> pd.DataFrame:
    """Per-fiber model features for one section, computed on its existing mask."""
    image = project.image(image_id)
    labels, fibers = image.outputs["fiber_labels"], image.outputs["fiber_table"]
    folder = cache / image_id
    table = folder / f"{image_id}_feature_diagnostics.csv"
    stamp = folder / "inputs.json"
    inputs = {"labels": file_sha256(labels), "fibers": file_sha256(fibers), "model": model}
    if not (table.is_file() and stamp.is_file() and json.loads(stamp.read_text()) == inputs):
        command = [
            sys.executable, "-m", "scripts.run_pipeline",
            "--input", str(image.raw_image_path),
            "--labels-path", str(labels),
            "--output-dir", str(folder),
            "--image-id", image_id,
            "--panel-config", str(panel),
            "--model", model,
            "--no-crop-auto",
            "--export-diagnostics",
        ]  # fmt: skip
        done = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        if done.returncode != 0 or not table.is_file():
            tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["unknown error"]
            raise ImproverError(f"{image_id}: feature extraction failed: {tail[0]}")
        stamp.write_text(json.dumps(inputs))
    features = pd.read_csv(table, low_memory=False)
    # The recomputed calls must be the calls that were reviewed; otherwise the features are not
    # the ones the current model saw.
    original = pd.read_csv(fibers, low_memory=False)
    recomputed = pd.read_csv(folder / f"{image_id}_fibers.csv", low_memory=False)
    key = "fiber_id" if "fiber_id" in original.columns else "label"
    same = (
        len(original) == len(recomputed) == len(features)
        and (original[key].to_numpy() == recomputed[key].to_numpy()).all()
        and (original[key].to_numpy() == features[key].to_numpy()).all()
        and (
            original["fiber_type"].astype(str).to_numpy()
            == recomputed["fiber_type"].astype(str).to_numpy()
        ).all()
    )
    if not same:
        raise ImproverError(
            f"{image_id}: recomputed model calls differ from the reviewed run; the image, mask, "
            "or model has changed since the run."
        )
    return features


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project = load_project(args.project)
    try:
        if not project.review_state_path.is_file():
            raise ImproverError("No review decisions yet. Review some fibers first.")
        session = load_session(project.review_state_path, expected_project_id=project.project_id)
        model = args.model or default_model_id()
        resolved = resolve_model_argument(model)
        manifest = yaml.safe_load(resolved.manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("features") or not manifest.get("feature_extraction"):
            raise ImproverError(
                f"Model {resolved.model_id} does not declare its features and extraction "
                "settings, so it cannot be adapted."
            )
        reviewed = sorted({decision.image_id for decision in session.object_decisions})
        features = {}
        for index, image_id in enumerate(reviewed, start=1):
            print(f"features {index}/{len(reviewed)}: {image_id}", flush=True)
            features[image_id] = extract_features(
                project, image_id, str(resolved.artifact_path), args.panel_config,
                args.output_dir / "features",
            )  # fmt: skip
        run_dir = args.output_dir / datetime.now(UTC).strftime("improve_%Y%m%dT%H%M%SZ")
        result = improve(
            project, session, features, manifest, run_dir, recipe=load_recipe(args.recipe)
        )
    except (ImproverError, ValueError) as exc:
        print(f"Model improvement stopped: {exc}", file=sys.stderr)
        return 1

    print(f"\n{result['n_labels']} reviewed fibers from {result['n_mice']} mice: ", end="")
    print(", ".join(f"{name} {count}" for name, count in result["labels_by_class"].items()))
    print(f"\n{'model':<20}{'median mouse macro-F1':>24}{'balanced accuracy':>20}")
    for name, values in result["metrics"].items():
        print(
            f"{name:<20}{values['median_mouse_macro_f1']:>24.3f}"
            f"{values['balanced_accuracy']:>20.3f}"
        )
    decision = result["decision"]
    print(f"\nRecommendation: {decision['recommended'] or 'keep the current model'}")
    print(decision["reason"])
    print(f"\n{result['caveat']}")
    print(f"\nrun folder: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
