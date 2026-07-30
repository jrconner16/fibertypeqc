from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.review.fiber_type_review import (
    FiberTypeReviewController,
    load_legacy_fiber_type_decisions,
)
from src.review.project import load_project
from src.review.queues import (
    QueueSource,
    RandomAuditScope,
    build_fiber_type_queue,
    load_fiber_type_rows,
)
from src.review.schemas import Domain, ObjectReviewStatus
from src.review.session import ReviewSession
from src.review.storage import load_session


def _project(tmp_path: Path):
    (tmp_path / "panel.yaml").write_text("channels: {}\n", encoding="utf-8")
    for image_id in ("one", "two"):
        directory = tmp_path / "pred" / image_id
        directory.mkdir(parents=True)
        pd.DataFrame(
            [
                {"fiber_id": 1, "fiber_type": "iib", "model_confidence": 0.9, "model_margin": 0.7},
                {
                    "fiber_id": 2,
                    "fiber_type": "iia",
                    "model_confidence": 0.2,
                    "model_margin": 0.01,
                    "needs_review": True,
                },
                {
                    "fiber_id": 3,
                    "fiber_type": "iix",
                    "prob_iib": 0.45,
                    "prob_iia": 0.45,
                    "prob_iix": 0.1,
                },
            ]
        ).to_csv(directory / "fibers.csv", index=False)
    manifest = {
        "schema_version": "review_project.v1",
        "project_id": "queue",
        "project_name": "Queue",
        "panel_manifest": "panel.yaml",
        "model_version": "model.v1",
        "images": [
            {
                "image_id": "one",
                "mouse_id": "mouse_a",
                "section_id": "s1",
                "raw_image_path": "one.tif",
                "prediction_directory": "pred/one",
                "outputs": {"fiber_table": "fibers.csv"},
                "applicable_domains": ["fiber_typing"],
            },
            {
                "image_id": "two",
                "mouse_id": "mouse_b",
                "section_id": "s1",
                "raw_image_path": "two.tif",
                "prediction_directory": "pred/two",
                "outputs": {"fiber_table": "fibers.csv"},
                "applicable_domains": ["fiber_typing"],
            },
        ],
    }
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return load_project(path, validate_paths=False)


def test_fiber_queues_are_reasoned_and_random_audits_are_reproducible(tmp_path: Path) -> None:
    rows = load_fiber_type_rows(_project(tmp_path))
    flagged = build_fiber_type_queue(rows, QueueSource.FLAGGED)
    assert [(item.image_id, item.fiber_id) for item in flagged] == [("one", 2), ("two", 2)]
    low_margin = build_fiber_type_queue(rows, QueueSource.LOW_PROBABILITY_MARGIN)
    assert low_margin[0].fiber_id == 3
    first = build_fiber_type_queue(
        rows, QueueSource.RANDOM_AUDIT, seed=7, sample_size=1, random_scope=RandomAuditScope.IMAGE
    )
    second = build_fiber_type_queue(
        rows, QueueSource.RANDOM_AUDIT, seed=7, sample_size=1, random_scope=RandomAuditScope.IMAGE
    )
    assert first == second
    assert len(first) == 2
    assert all("seed=7" in item.reason_code for item in first)


def test_canonical_fiber_decision_preserves_model_and_queue_provenance(tmp_path: Path) -> None:
    project = _project(tmp_path)
    queue = build_fiber_type_queue(load_fiber_type_rows(project), QueueSource.FLAGGED)
    session = ReviewSession(project_id=project.project_id, model_version=project.model_version)
    controller = FiberTypeReviewController(session)
    controller.set_queue(queue, QueueSource.FLAGGED.value)
    event = controller.decide("iib", reason_code="reviewed_signal")
    controller.save(project, event)

    restored = load_session(project.review_state_path)
    decision = restored.object_decisions[0]
    assert decision.model_fiber_type == "iia"
    assert decision.reviewed_fiber_type == "iib"
    assert decision.review_status is ObjectReviewStatus.CORRECTED
    assert decision.queue_source == "flagged"
    assert "record_fiber_type_decision" in project.review_events_path.read_text(encoding="utf-8")


def test_undo_restores_prior_decision_and_queue_position(tmp_path: Path) -> None:
    project = _project(tmp_path)
    queue = build_fiber_type_queue(load_fiber_type_rows(project), QueueSource.FLAGGED)
    session = ReviewSession(project_id=project.project_id, model_version=project.model_version)
    controller = FiberTypeReviewController(session)
    controller.set_queue(queue, QueueSource.FLAGGED.value)
    event = controller.decide("iib")
    controller.save(project, event)
    controller.move(1)

    undo_event = controller.undo_last_decision()
    controller.save(project, undo_event)

    assert session.queue_position == 0
    assert session.object_decisions == []
    assert "undo_fiber_type_decision" in project.review_events_path.read_text(encoding="utf-8")


def test_legacy_review_csv_adapts_to_canonical_decisions(tmp_path: Path) -> None:
    path = tmp_path / "legacy.csv"
    pd.DataFrame([{"fiber_id": 7, "predicted_type": "iib", "corrected_type": "iia"}]).to_csv(
        path, index=False
    )

    decisions = load_legacy_fiber_type_decisions(path, image_id="image")

    assert decisions[0].model_fiber_type == "iib"
    assert decisions[0].reviewed_fiber_type == "iia"
    assert decisions[0].queue_source == "legacy_review_csv"


def test_object_inspector_widget_offscreen_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qtwidgets = pytest.importorskip("qtpy.QtWidgets")
    from src.review.object_inspector_widget import ObjectInspectorWidget

    application = qtwidgets.QApplication.instance() or qtwidgets.QApplication([])
    project = _project(tmp_path)
    session = ReviewSession(project_id=project.project_id, model_version=project.model_version)
    widget = ObjectInspectorWidget(project, FiberTypeReviewController(session))

    assert widget.controller.current_item is not None
    assert "Fiber typing" in widget.context.text()
    widget.close()
    widget.deleteLater()
    application.processEvents()


def test_guided_review_widget_auto_advances_and_undoes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qtwidgets = pytest.importorskip("qtpy.QtWidgets")
    from src.review.guided_review_widget import GuidedReviewWidget

    application = qtwidgets.QApplication.instance() or qtwidgets.QApplication([])
    project = _project(tmp_path)
    session = ReviewSession(project_id=project.project_id, model_version=project.model_version)
    widget = GuidedReviewWidget(project, FiberTypeReviewController(session))
    widget.start_flagged_review()
    initial_position = session.queue_position

    assert "K keep model" in widget.shortcut_label.text()
    assert "Autosave is on" in widget.save_status.text()

    widget._keep_model_call()

    assert widget.undo_button.isEnabled()
    assert session.queue_position != initial_position
    widget.undo()
    assert session.queue_position == initial_position
    assert session.object_decisions == []
    widget.close()
    widget.deleteLater()
    application.processEvents()


def test_guided_review_restores_saved_queue_position(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qtwidgets = pytest.importorskip("qtpy.QtWidgets")
    from src.review.guided_review_widget import GuidedReviewWidget

    application = qtwidgets.QApplication.instance() or qtwidgets.QApplication([])
    project = _project(tmp_path)
    session = ReviewSession(project_id=project.project_id, model_version=project.model_version)
    first = GuidedReviewWidget(project, FiberTypeReviewController(session))
    first.start_flagged_review()
    first._move(2)
    expected_position = session.queue_position
    first.close()
    first.deleteLater()

    resumed = GuidedReviewWidget(project, FiberTypeReviewController(session))

    assert resumed._review_started
    assert session.active_queue == QueueSource.FLAGGED.value
    assert session.queue_position == expected_position
    resumed.close()
    resumed.deleteLater()
    application.processEvents()


def test_guided_navigator_and_tutorial_are_recoverable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qtwidgets = pytest.importorskip("qtpy.QtWidgets")
    from src.review.guided_review_widget import GuidedReviewWidget

    application = qtwidgets.QApplication.instance() or qtwidgets.QApplication([])
    project = _project(tmp_path)
    calls: list[object] = []
    widget = GuidedReviewWidget(
        project,
        FiberTypeReviewController(
            ReviewSession(project_id=project.project_id, model_version="model.v1")
        ),
        show_dashboard=lambda: calls.append("cohort"),
        show_section=lambda: calls.append("section"),
        show_domain=lambda domain: calls.append(domain),
        focus_current_object=lambda: calls.append("focus"),
    )
    widget._show_dashboard()
    widget._show_section()
    widget._show_domain(Domain.NUCLEI)
    widget._focus_current_object()
    widget._show_tutorial()
    assert not widget.tutorial_group.isHidden()
    widget._dismiss_tutorial()

    assert calls == ["cohort", "section", Domain.NUCLEI, "focus"]
    assert not widget.tutorial_group.isVisible()
    widget.close()
    widget.deleteLater()
    application.processEvents()
