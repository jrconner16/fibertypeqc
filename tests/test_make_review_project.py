from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.make_review_project import main


def _image_outputs(batch: Path, image_id: str) -> None:
    out = batch / image_id
    out.mkdir(parents=True)
    (out / f"{image_id}_cellpose_labels.tif").write_bytes(b"labels")
    (out / f"{image_id}_fibers.csv").write_text("label,fiber_type\n1,iia\n")
    (out / f"{image_id}_run.json").write_text(
        json.dumps({"model_manifest_path": "manifests/models/quad_four_class_rf_v1.yaml"})
    )


def _batch(tmp_path: Path) -> Path:
    batch = tmp_path / "batch"
    _image_outputs(batch, "plain")
    _image_outputs(batch, "slide_section-01")
    _image_outputs(batch, "slide_section-02")
    scenes = batch / "raw_scene_exports" / "slide"
    scenes.mkdir(parents=True)
    for number in ("01", "02"):
        (scenes / f"slide_section-{number}.tif").write_bytes(b"scene")
    pd.DataFrame(
        {
            "image_name": ["plain", "slide_section-01", "slide_section-02", "broken"],
            "status": ["success", "success", "success", "failed"],
        }
    ).to_csv(batch / "batch_summary.csv", index=False)
    return batch


def _sheet(tmp_path: Path) -> Path:
    (tmp_path / "raw").mkdir()
    sheet = tmp_path / "samples.csv"
    pd.DataFrame(
        {
            "image_id": ["plain", "slide", "broken"],
            "mouse_id": ["m1", "m2", "m3"],
            "raw_image_path": ["raw/plain.tif", "raw/slide.czi", "raw/broken.czi"],
            "section_id": ["left", "", ""],
            "genotype": ["wt", "mdx", "mdx"],
        }
    ).to_csv(sheet, index=False)
    return sheet


def test_builds_project_with_split_sections_and_conditions(tmp_path, capsys):
    batch, sheet = _batch(tmp_path), _sheet(tmp_path)
    panel = tmp_path / "panel.yaml"
    panel.write_text("channels: {}\n")
    project_dir = tmp_path / "project"

    assert (
        main(
            [
                "--batch-dir",
                str(batch),
                "--sample-sheet",
                str(sheet),
                "--panel-config",
                str(panel),
                "--project-dir",
                str(project_dir),
            ]
        )
        == 0
    )

    project = yaml.safe_load((project_dir / "project.yaml").read_text())
    images = {image["image_id"]: image for image in project["images"]}
    assert project["model_version"] == "quad_four_class_rf_v1"
    assert set(images) == {"plain", "slide_section-01", "slide_section-02"}
    assert images["plain"]["section_id"] == "left"
    assert images["plain"]["raw_image_path"] == "../raw/plain.tif"
    assert images["slide_section-02"]["section_id"] == "section-02"
    assert images["slide_section-02"]["raw_image_path"].endswith(
        "raw_scene_exports/slide/slide_section-02.tif"
    )
    assert images["slide_section-01"]["mouse_id"] == "m2"
    assert images["slide_section-01"]["condition"] == {"genotype": "mdx"}
    assert "skipped broken: batch status failed" in capsys.readouterr().out


def test_refuses_to_overwrite_project_or_nest_inside_batch(tmp_path):
    batch, sheet = _batch(tmp_path), _sheet(tmp_path)
    panel = tmp_path / "panel.yaml"
    panel.write_text("channels: {}\n")
    args = ["--batch-dir", str(batch), "--sample-sheet", str(sheet), "--panel-config", str(panel)]

    assert main([*args, "--project-dir", str(tmp_path / "p")]) == 0
    assert main([*args, "--project-dir", str(tmp_path / "p")]) == 2
    assert main([*args, "--project-dir", str(batch / "p")]) == 2


def test_rejects_sample_sheet_without_required_columns(tmp_path, capsys):
    batch = _batch(tmp_path)
    sheet = tmp_path / "bad.csv"
    pd.DataFrame({"image_id": ["plain"]}).to_csv(sheet, index=False)
    panel = tmp_path / "panel.yaml"
    panel.write_text("channels: {}\n")

    assert (
        main(
            [
                "--batch-dir",
                str(batch),
                "--sample-sheet",
                str(sheet),
                "--panel-config",
                str(panel),
                "--project-dir",
                str(tmp_path / "p"),
            ]
        )
        == 2
    )
    assert "missing columns: mouse_id, raw_image_path" in capsys.readouterr().err
