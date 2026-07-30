from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import yaml

from src.review.invalidation import StaleProduct
from src.review.nuclear_review import NuclearReviewController, NucleusQueueSource
from src.review.project import load_project
from src.review.schemas import NucleusAssociationStatus
from src.review.session import ReviewSession


def _project(tmp_path: Path):
    (tmp_path / "panel.yaml").write_text("channels: {}\n", encoding="utf-8")
    prediction = tmp_path / "pred"
    prediction.mkdir()
    fibers = np.zeros((8, 10), dtype=np.int32)
    fibers[1:7, 1:5] = 1
    fibers[1:7, 6:9] = 2
    nuclei = np.zeros_like(fibers)
    nuclei[2:4, 2:4] = 1
    nuclei[2:4, 7:8] = 2
    tifffile.imwrite(prediction / "fibers.tif", fibers)
    tifffile.imwrite(prediction / "nuclei.tif", nuclei)
    pd.DataFrame(
        [
            {"nucleus_id": 1, "assigned_fiber_id": 1, "assignment_status": "assigned"},
            {
                "nucleus_id": 2,
                "assigned_fiber_id": 0,
                "assignment_status": "unassigned_or_interstitial",
            },
        ]
    ).to_csv(prediction / "nuclei.csv", index=False)
    manifest = {
        "schema_version": "review_project.v1",
        "project_id": "nuclei",
        "project_name": "Nuclei",
        "panel_manifest": "panel.yaml",
        "model_version": "model.v1",
        "images": [
            {
                "image_id": "one",
                "mouse_id": "mouse",
                "section_id": "section",
                "raw_image_path": "raw.tif",
                "prediction_directory": "pred",
                "outputs": {
                    "fiber_labels": "fibers.tif",
                    "nuclei_labels": "nuclei.tif",
                    "nuclei_table": "nuclei.csv",
                },
                "applicable_domains": ["fiber_segmentation", "nuclei"],
            }
        ],
    }
    (tmp_path / "project.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return load_project(tmp_path / "project.yaml", validate_paths=False), prediction / "nuclei.tif"


def test_nuclear_mask_copy_on_write_delete_and_stale_products(tmp_path: Path) -> None:
    project, predicted = _project(tmp_path)
    session = ReviewSession(project_id=project.project_id, model_version=project.model_version)
    controller = NuclearReviewController(project, session)

    event = controller.delete_nucleus("one", 1, reason_code="artifact")
    controller.save(event)

    assert np.any(tifffile.imread(predicted) == 1)
    assert not np.any(controller.load_nuclei_labels("one") == 1)
    assert StaleProduct.NUCLEUS_FEATURES.value in session.stale_products["one"]
    assert StaleProduct.FIBER_TYPE_PREDICTION_FEATURES.value not in session.stale_products["one"]
    assert "delete_nucleus" in project.review_events_path.read_text(encoding="utf-8")


def test_nucleus_association_queue_and_reviewed_output(tmp_path: Path) -> None:
    project, _ = _project(tmp_path)
    session = ReviewSession(project_id=project.project_id, model_version=project.model_version)
    controller = NuclearReviewController(project, session)

    queue = controller.association_queue("one", NucleusQueueSource.UNASSIGNED)
    assert [item.nucleus_id for item in queue] == [2]
    event = controller.set_association(
        "one",
        2,
        fiber_id=2,
        status=NucleusAssociationStatus.ASSIGNED,
        reason_code="manual_overlap_review",
    )
    controller.save(event)

    reviewed = pd.read_csv(controller.reviewed_associations_path)
    assert reviewed["nucleus_id"].tolist() == [1, 2]
    row = reviewed.loc[reviewed["nucleus_id"] == 2].iloc[0]
    assert int(row["reviewed_fiber_id"]) == 2
    assert row["reviewed_association_status"] == "assigned"
    assert StaleProduct.NUCLEUS_ASSOCIATIONS.value in session.stale_products["one"]
    assert StaleProduct.FIBER_NUCLEUS_COUNTS.value in session.stale_products["one"]


def test_association_queue_uses_manual_decisions_and_persists_position(tmp_path: Path) -> None:
    project, _ = _project(tmp_path)
    session = ReviewSession(project_id=project.project_id, model_version=project.model_version)
    controller = NuclearReviewController(project, session)

    controller.set_association_queue("one", NucleusQueueSource.UNASSIGNED)
    assert controller.current_item is not None
    assert controller.current_item.nucleus_id == 2
    controller.set_association("one", 2, fiber_id=2)

    assert controller.association_queue("one", NucleusQueueSource.UNASSIGNED) == ()
    full_queue = controller.association_queue("one", NucleusQueueSource.FULL)
    assert [item.nucleus_id for item in full_queue] == [1, 2]
    assert session.active_queue == "nuclei:one:unassigned_nuclei"
    assert session.queue_position == 0


def test_deleted_nucleus_is_omitted_from_reviewed_associations(tmp_path: Path) -> None:
    project, _ = _project(tmp_path)
    session = ReviewSession(project_id=project.project_id, model_version=project.model_version)
    controller = NuclearReviewController(project, session)
    controller.set_association("one", 1, fiber_id=1)
    event = controller.delete_nucleus("one", 1)
    controller.save(event)

    reviewed = pd.read_csv(controller.reviewed_associations_path)
    assert reviewed["nucleus_id"].tolist() == [2]
    assert session.nucleus_association_decisions == []
