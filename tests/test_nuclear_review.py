from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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


def test_added_nucleus_has_stable_id_and_unresolved_reviewed_output(tmp_path: Path) -> None:
    project, predicted = _project(tmp_path)
    session = ReviewSession(project_id=project.project_id, model_version=project.model_version)
    controller = NuclearReviewController(project, session)
    pixels = np.zeros((8, 10), dtype=bool)
    pixels[5:7, 7:9] = True

    event = controller.add_nucleus("one", pixels, reason_code="missed_by_model")
    controller.save(event)

    assert int(event.target_id) == 3
    assert not np.any(tifffile.imread(predicted) == 3)
    assert np.count_nonzero(controller.load_nuclei_labels("one") == 3) == 4
    reviewed = pd.read_csv(controller.reviewed_associations_path)
    row = reviewed.loc[reviewed["nucleus_id"] == 3].iloc[0]
    assert not row["model_row_available"]
    assert row["reviewed_association_status"] == "unresolved"
    assert StaleProduct.NUCLEUS_FEATURES.value in session.stale_products["one"]

    controller.delete_nucleus("one", 3)
    next_event = controller.add_nucleus("one", pixels)
    assert int(next_event.target_id) == 4


def test_added_nucleus_rejects_overlap_and_mismatched_shape(tmp_path: Path) -> None:
    project, _ = _project(tmp_path)
    controller = NuclearReviewController(
        project, ReviewSession(project_id=project.project_id, model_version=project.model_version)
    )
    overlap = np.zeros((8, 10), dtype=bool)
    overlap[2:4, 2:4] = True

    with pytest.raises(ValueError, match="overlap"):
        controller.add_nucleus("one", overlap)
    with pytest.raises(ValueError, match="shape"):
        controller.add_nucleus("one", np.zeros((2, 2), dtype=bool))


def test_nuclear_review_widget_saves_painted_draft_and_association(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qtwidgets = pytest.importorskip("qtpy.QtWidgets")
    from src.review.nuclear_review_widget import NuclearReviewWidget

    application = qtwidgets.QApplication.instance() or qtwidgets.QApplication([])
    project, _ = _project(tmp_path)
    session = ReviewSession(
        project_id=project.project_id,
        model_version=project.model_version,
        current_image_id="one",
    )
    controller = NuclearReviewController(project, session)
    draft = np.zeros((8, 10), dtype=bool)
    draft[5:7, 7:9] = True
    changes: list[str] = []
    widget = NuclearReviewWidget(
        controller,
        selected_nucleus_id=lambda: 2,
        draft_pixels=lambda: draft,
        reset_draft=lambda: draft.fill(False),
        review_changed=lambda: changes.append("changed"),
    )

    widget._add()
    assert widget.nucleus_id.value() == 3
    assert not draft.any()
    widget.fiber_id.setValue(2)
    widget._associate()

    assert changes == ["changed", "changed"]
    assert session.nucleus_association_decisions[0].nucleus_id == 3
    widget.close()
    widget.deleteLater()
    application.processEvents()
