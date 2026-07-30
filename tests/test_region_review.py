from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.review.project import load_project
from src.review.region_review import RegionReviewController
from src.review.schemas import Domain, RegionAction
from src.review.session import ReviewSession


def _project(tmp_path: Path):
    (tmp_path / "panel.yaml").write_text("channels: {}\n", encoding="utf-8")
    directory = tmp_path / "pred"
    directory.mkdir()
    pd.DataFrame([{"fiber_id": 1, "fiber_type": "iib"}]).to_csv(
        directory / "fibers.csv", index=False
    )
    manifest = {
        "schema_version": "review_project.v1",
        "project_id": "regions",
        "project_name": "Regions",
        "panel_manifest": "panel.yaml",
        "model_version": "model.v1",
        "images": [
            {
                "image_id": "one",
                "mouse_id": "mouse_a",
                "section_id": "s1",
                "raw_image_path": "one.tif",
                "prediction_directory": "pred",
                "outputs": {"fiber_table": "fibers.csv"},
                "applicable_domains": ["fiber_typing"],
            }
        ],
    }
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return load_project(path, validate_paths=False)


def _polygon() -> dict:
    return {"type": "Polygon", "coordinates": [[[1, 2], [1, 4], [3, 2], [1, 2]]]}


def test_region_action_persists_geojson_event_and_queue(tmp_path: Path) -> None:
    project = _project(tmp_path)
    session = ReviewSession(
        project_id=project.project_id,
        model_version=project.model_version,
        current_image_id="one",
        reviewer="reviewer",
    )
    controller = RegionReviewController(project, session)

    event = controller.add_region(
        image_id="one",
        geometry=_polygon(),
        domain=Domain.FIBER_TYPING,
        action=RegionAction.QUEUE_OBJECTS,
        reason_code="edge_artifact",
        notes="Review these fibers individually.",
    )
    controller.save(event)

    payload = json.loads(controller.regions_geojson_path.read_text(encoding="utf-8"))
    assert payload["type"] == "FeatureCollection"
    assert payload["features"][0]["geometry"] == _polygon()
    assert payload["features"][0]["properties"]["action"] == "queue_objects"
    assert controller.queue("one")[0].region_id == session.regions[0].region_id
    assert "add_region_annotation" in project.review_events_path.read_text(encoding="utf-8")


def test_region_action_rejects_inapplicable_domain_and_removal_is_audited(tmp_path: Path) -> None:
    project = _project(tmp_path)
    session = ReviewSession(project_id=project.project_id, model_version=project.model_version)
    controller = RegionReviewController(project, session)

    with pytest.raises(ValueError, match="not applicable"):
        controller.add_region(
            image_id="one",
            geometry=_polygon(),
            domain=Domain.NUCLEI,
            action=RegionAction.IGNORE_NUCLEI,
        )

    added = controller.add_region(
        image_id="one",
        geometry=_polygon(),
        domain=Domain.FIBER_TYPING,
        action=RegionAction.UNRESOLVED,
    )
    controller.save(added)
    removed = controller.remove_region(session.regions[0].region_id)
    controller.save(removed)

    assert session.regions == []
    assert json.loads(controller.regions_geojson_path.read_text(encoding="utf-8"))["features"] == []
    assert "remove_region_annotation" in project.review_events_path.read_text(encoding="utf-8")


def test_region_coverage_heatmap_uses_saved_geometry_and_display_scale(tmp_path: Path) -> None:
    project = _project(tmp_path)
    session = ReviewSession(project_id=project.project_id, model_version=project.model_version)
    controller = RegionReviewController(project, session)
    controller.add_region(
        image_id="one",
        geometry={"type": "Polygon", "coordinates": [[[2, 2], [2, 6], [6, 2], [2, 2]]]},
        domain=Domain.FIBER_TYPING,
        action=RegionAction.DETAILED_REVIEW,
    )

    full = controller.coverage_heatmap("one", (10, 10))
    downsampled = controller.coverage_heatmap("one", (5, 5), coordinate_scale=2)

    assert full[3, 3] == 1
    assert full[8, 8] == 0
    assert downsampled[1, 1] == 1
