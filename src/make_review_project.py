"""Build a review project YAML from a run_batch output directory and a sample sheet.

The sample sheet is a CSV with one row per batch input image:

- ``image_id`` (as given to run_batch: manifest ``image_id`` or file stem),
- ``mouse_id``,
- ``raw_image_path`` (path to the image run_batch read),
- optional ``section_id`` (default ``s1``),
- any other columns become per-image ``condition`` fields (e.g. genotype, timepoint).

Images split with ``run_batch --split-czi-scenes`` are expanded to their sections
(``<image_id>_section-NN``), each using its exported scene TIFF as the raw image. Images that
failed in the batch are skipped with a warning. The model version is read from the run records.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REQUIRED_COLUMNS = ("image_id", "mouse_id", "raw_image_path")
RESERVED_COLUMNS = (*REQUIRED_COLUMNS, "section_id")
SECTION_SUFFIX = re.compile(r"^(?P<base>.+)_section-(?P<number>\d{2})$")


def _relative(path: Path, start: Path) -> str:
    return os.path.relpath(path.resolve(), start.resolve())


def _model_version(run_record: Path) -> str:
    record = json.loads(run_record.read_text(encoding="utf-8"))
    manifest = record.get("model_manifest_path")
    if manifest:
        raw = yaml.safe_load((Path(__file__).resolve().parents[1] / manifest).read_text())
        if isinstance(raw, dict) and raw.get("model_id"):
            return str(raw["model_id"])
        return Path(manifest).stem
    classifier = record.get("classifier_path")
    if classifier:
        return Path(classifier).stem
    return "rules"


def build_project(
    batch_dir: Path,
    sample_sheet: Path,
    panel_config: Path,
    project_dir: Path,
    *,
    project_id: str,
    project_name: str,
) -> tuple[dict[str, Any], list[str]]:
    sheet = pd.read_csv(sample_sheet, dtype=str, keep_default_na=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in sheet.columns]
    if missing:
        raise ValueError(f"Sample sheet is missing columns: {', '.join(missing)}")
    if sheet["image_id"].duplicated().any():
        raise ValueError("Sample sheet image_id values must be unique")
    rows = {row["image_id"]: row for row in sheet.to_dict("records")}
    summary = pd.read_csv(batch_dir / "batch_summary.csv", dtype=str, keep_default_na=False)
    warnings: list[str] = []
    images: list[dict[str, Any]] = []
    model_versions: set[str] = set()
    for record in summary.to_dict("records"):
        image_id = record["image_name"]
        if record.get("status") != "success":
            warnings.append(f"skipped {image_id}: batch status {record.get('status')}")
            continue
        match = SECTION_SUFFIX.match(image_id)
        base_id = match.group("base") if match and match.group("base") in rows else image_id
        if base_id not in rows:
            warnings.append(f"skipped {image_id}: not in the sample sheet")
            continue
        sample = rows[base_id]
        prediction_dir = batch_dir / image_id
        labels = prediction_dir / f"{image_id}_cellpose_labels.tif"
        fibers = prediction_dir / f"{image_id}_fibers.csv"
        if not labels.is_file() or not fibers.is_file():
            warnings.append(
                f"skipped {image_id}: fiber labels or table missing (run_batch --retain-mode full)"
            )
            continue
        if base_id != image_id:
            exported = sorted(
                (batch_dir / "raw_scene_exports" / base_id).glob(
                    f"*_section-{match.group('number')}.tif"
                )
            )
            if len(exported) != 1:
                warnings.append(f"skipped {image_id}: exported scene TIFF not found")
                continue
            raw_image = exported[0]
            section_id = f"section-{match.group('number')}"
        else:
            raw_image = Path(sample["raw_image_path"]).expanduser()
            if not raw_image.is_absolute():
                raw_image = (sample_sheet.parent / raw_image).resolve()
            section_id = sample.get("section_id") or "s1"
        model_versions.add(_model_version(prediction_dir / f"{image_id}_run.json"))
        condition = {
            column: value
            for column, value in sample.items()
            if column not in RESERVED_COLUMNS and value != ""
        }
        images.append(
            {
                "image_id": image_id,
                "mouse_id": sample["mouse_id"],
                "section_id": section_id,
                "raw_image_path": _relative(raw_image, project_dir),
                "prediction_directory": _relative(prediction_dir, project_dir),
                "condition": condition,
                "applicable_domains": ["fiber_segmentation", "fiber_typing"],
                "outputs": {"fiber_labels": labels.name, "fiber_table": fibers.name},
            }
        )
    if not images:
        raise ValueError("No successful batch images matched the sample sheet")
    if len(model_versions) != 1:
        raise ValueError(f"Batch images used different models: {sorted(model_versions)}")
    project = {
        "schema_version": "review_project.v1",
        "project_id": project_id,
        "project_name": project_name,
        "panel_manifest": _relative(panel_config, project_dir),
        "model_version": model_versions.pop(),
        "images": images,
    }
    return project, warnings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True, help="run_batch output dir")
    parser.add_argument("--sample-sheet", type=Path, required=True)
    parser.add_argument("--panel-config", type=Path, required=True)
    parser.add_argument(
        "--project-dir",
        type=Path,
        required=True,
        help="Where project.yaml, review state, and QC go; must be outside the batch dir.",
    )
    parser.add_argument("--project-id", default=None, help="Default: project directory name")
    parser.add_argument("--project-name", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_dir = args.project_dir.expanduser().resolve()
    batch_dir = args.batch_dir.expanduser().resolve()
    if project_dir == batch_dir or batch_dir in project_dir.parents:
        print("--project-dir must be outside the batch output directory", file=sys.stderr)
        return 2
    project_dir.mkdir(parents=True, exist_ok=True)
    project_id = args.project_id or project_dir.name
    try:
        project, warnings = build_project(
            batch_dir,
            args.sample_sheet.expanduser().resolve(),
            args.panel_config.expanduser().resolve(),
            project_dir,
            project_id=project_id,
            project_name=args.project_name or project_id,
        )
    except ValueError as exc:
        print(f"Cannot build project: {exc}", file=sys.stderr)
        return 2
    path = project_dir / "project.yaml"
    if path.exists():
        print(f"{path} already exists; remove it or choose another --project-dir", file=sys.stderr)
        return 2
    path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    for warning in warnings:
        print(f"warning: {warning}")
    print(f"wrote {path} with {len(project['images'])} image(s); model {project['model_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
