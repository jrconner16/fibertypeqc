"""Blind reference labelling: separate store, hidden model output, and session timing."""

from __future__ import annotations

import pandas as pd
import pytest

from src.review.finalization import finalize_project
from src.review.storage import load_session
from src.review.timing import SESSIONS_FILENAME, ReviewTimer
from tests.test_review_finalization import _project, _session


def test_reference_project_redirects_review_state_per_reviewer(tmp_path):
    project = _project(tmp_path)
    reference = project.for_reference("Reviewer A")

    assert project.review_state_path == tmp_path / "review" / "review_state.json"
    assert (
        reference.review_state_path == tmp_path / "reference" / "Reviewer_A" / "review_state.json"
    )
    assert reference.review_events_path.parent == reference.review_directory
    assert reference.images == project.images
    with pytest.raises(ValueError, match="reviewer name"):
        project.for_reference("  ")


def test_timer_counts_active_time_and_ignores_long_gaps(tmp_path):
    now = [0.0]
    timer = ReviewTimer(
        tmp_path, reviewer="a", mode="blind_reference", idle_cutoff_seconds=60, clock=lambda: now[0]
    )
    for step, decision in ((5, True), (10, True), (600, False), (4, True)):
        now[0] += step
        timer.activity(decision=decision)

    saved = pd.read_csv(tmp_path / SESSIONS_FILENAME)
    assert len(saved) == 1
    assert saved.loc[0, "n_decisions"] == 3
    assert saved.loc[0, "active_seconds"] == 19.0  # the 600 s break is not counted
    assert saved.loc[0, "mode"] == "blind_reference"

    other = ReviewTimer(tmp_path, reviewer="b", mode="guided_review", clock=lambda: now[0])
    other.activity()
    assert len(pd.read_csv(tmp_path / SESSIONS_FILENAME)) == 2


def test_blind_widget_hides_model_output_and_saves_separately(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qtpy = pytest.importorskip("qtpy.QtWidgets")
    from src.review.fiber_type_review import FiberTypeReviewController
    from src.review.guided_review_widget import GuidedReviewWidget
    from src.review.session import ReviewSession

    application = qtpy.QApplication.instance() or qtpy.QApplication([])
    project = _project(tmp_path)
    reference = project.for_reference("rev1")
    session = ReviewSession(
        project_id=project.project_id, model_version=project.model_version, reviewer="rev1"
    )
    session.current_image_id = "one"
    widget = GuidedReviewWidget(reference, FiberTypeReviewController(session), blind=True)

    widget.start_section_review()
    first = widget.controller.current_item
    shown = widget.details.text()
    widget._keep_model_call()  # must do nothing in blind mode
    widget._record_type("iib")
    application.processEvents()

    assert "Model call" not in shown and "confidence" not in shown and "Why shown" not in shown
    assert first.model_fiber_type.upper() not in shown.replace(f"Fiber {first.fiber_id}", "")
    assert not widget.legend_label.isVisibleTo(widget)
    assert not widget.start_flagged_button.isVisibleTo(widget)
    assert widget.blind_banner.isVisibleTo(widget)
    assert [widget.queue_combo.itemText(i) for i in range(widget.queue_combo.count())] == [
        "random_audit",
        "full_audit",
    ]
    saved = load_session(reference.review_state_path, expected_project_id=project.project_id)
    assert [(d.fiber_id, d.reviewed_fiber_type) for d in saved.object_decisions] == [
        (first.fiber_id, "iib")
    ]
    assert not project.review_state_path.exists()  # normal review state untouched
    assert (reference.review_directory / SESSIONS_FILENAME).is_file()
    # Reference labels are never applied to results.
    manifest = finalize_project(project, _session(project), tmp_path / "final")
    assert manifest["images"][0]["counts"]["reviewed"] == 0
    widget.close()
    widget.deleteLater()


def test_class_buttons_follow_the_projects_classes(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qtpy = pytest.importorskip("qtpy.QtWidgets")
    from src.review.fiber_type_review import FiberTypeReviewController
    from src.review.guided_review_widget import GuidedReviewWidget

    application = qtpy.QApplication.instance() or qtpy.QApplication([])
    project = _project(tmp_path)
    table = tmp_path / "one" / "one_fibers.csv"
    fibers = pd.read_csv(table)
    fibers["fiber_type"] = ["iia", "iib", "iix", "iix"]  # a three-class model: no Type I
    fibers.to_csv(table, index=False)

    widget = GuidedReviewWidget(project, FiberTypeReviewController(_session(project)))

    assert widget.classes == {"iia", "iib", "iix"}
    application.processEvents()
    widget.close()
    widget.deleteLater()
