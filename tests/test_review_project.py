from pathlib import Path

import pytest
import yaml

from src.review.project import load_project
from src.review.schemas import Domain


def _write_project(tmp_path: Path, *, images: list[dict] | None = None) -> Path:
    (tmp_path / "panel.yaml").write_text("channels: {}\n", encoding="utf-8")
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "image.tif").write_bytes(b"raw")
    prediction = tmp_path / "predictions" / "image_1"
    prediction.mkdir(parents=True)
    (prediction / "labels.tif").write_bytes(b"labels")
    data = {
        "schema_version": "review_project.v1",
        "project_id": "project_1",
        "project_name": "Synthetic project",
        "panel_manifest": "panel.yaml",
        "model_version": "model.v1",
        "qc_version": "qc.v1",
        "images": images
        or [
            {
                "image_id": "image_1",
                "mouse_id": "mouse_1",
                "section_id": "section_1",
                "condition": {"genotype": "synthetic"},
                "raw_image_path": "raw/image.tif",
                "prediction_directory": "predictions/image_1",
                "outputs": {"fiber_labels": "labels.tif"},
            }
        ],
    }
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_project_manifest_loads_and_resolves_paths(tmp_path: Path) -> None:
    project = load_project(_write_project(tmp_path))

    assert project.project_id == "project_1"
    assert project.image("image_1").mouse_id == "mouse_1"
    assert project.image("image_1").raw_image_path == (tmp_path / "raw/image.tif").resolve()
    assert project.image("image_1").outputs["fiber_labels"] == (
        tmp_path / "predictions/image_1/labels.tif"
    ).resolve()
    reviewed = project.reviewed_mask_path("image_1", Domain.FIBER_SEGMENTATION)
    assert reviewed == (tmp_path / "review/reviewed_fiber_labels/labels.tif").resolve()
    assert project.image("image_1").applicable_domains == {
        Domain.FIBER_SEGMENTATION
    }


def test_project_manifest_infers_and_accepts_explicit_domain_applicability(
    tmp_path: Path,
) -> None:
    path = _write_project(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["images"][0]["outputs"]["fiber_table"] = "fibers.csv"
    (tmp_path / "predictions/image_1/fibers.csv").write_text(
        "fiber_id,fiber_type\n1,iib\n",
        encoding="utf-8",
    )
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    inferred = load_project(path)
    assert inferred.image("image_1").applicable_domains == {
        Domain.FIBER_SEGMENTATION,
        Domain.FIBER_TYPING,
    }

    data["images"][0]["applicable_domains"] = ["nuclei"]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    explicit = load_project(path)
    assert explicit.image("image_1").applicable_domains == {Domain.NUCLEI}


def test_project_manifest_rejects_duplicate_image_ids(tmp_path: Path) -> None:
    image = {
        "image_id": "duplicate",
        "mouse_id": "mouse",
        "section_id": "section",
        "raw_image_path": "raw/image.tif",
        "prediction_directory": "predictions/image_1",
    }
    path = _write_project(tmp_path, images=[image, dict(image)])

    with pytest.raises(ValueError, match="image_id values must be unique"):
        load_project(path)


def test_project_manifest_reports_corrupt_yaml(tmp_path: Path) -> None:
    path = tmp_path / "project.yaml"
    path.write_text("images: [\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid YAML"):
        load_project(path)


def test_project_manifest_reports_missing_artifact_path(tmp_path: Path) -> None:
    path = _write_project(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["images"][0]["outputs"]["fiber_labels"] = "missing.tif"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match=r"images\[0\]\.outputs\.fiber_labels"):
        load_project(path)


def test_project_manifest_can_validate_structure_before_artifacts_exist(tmp_path: Path) -> None:
    path = tmp_path / "project.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "review_project.v1",
                "project_id": "future",
                "project_name": "Future outputs",
                "panel_manifest": "panel.yaml",
                "model_version": "model.v1",
                "images": [
                    {
                        "image_id": "image",
                        "mouse_id": "mouse",
                        "section_id": "section",
                        "raw_image_path": "raw.tif",
                        "prediction_directory": "predictions/image",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    project = load_project(path, validate_paths=False)

    assert project.image("image").section_id == "section"


def test_project_rejects_review_directory_inside_prediction_directory(tmp_path: Path) -> None:
    (tmp_path / "panel.yaml").write_text("channels: {}\n")
    (tmp_path / "raw.tif").write_bytes(b"raw")
    (tmp_path / "review").mkdir()
    path = tmp_path / "project.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "review_project.v1",
                "project_id": "unsafe",
                "project_name": "Unsafe layout",
                "panel_manifest": "panel.yaml",
                "model_version": "model.v1",
                "images": [
                    {
                        "image_id": "image",
                        "mouse_id": "mouse",
                        "section_id": "section",
                        "raw_image_path": "raw.tif",
                        "prediction_directory": ".",
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="review directory must be outside"):
        load_project(path)
