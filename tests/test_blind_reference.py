"""Blind reference labelling: separate store, hidden model output, and session timing."""

from __future__ import annotations

import numpy as np
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


def _blind_session(project, reviewer, labels, fields=()):
    """Save a reviewer's blind labels ({fiber_id: label}) and optional field polygons."""
    from src.review.finalization import image_input_fingerprints
    from src.review.schemas import FiberTypeDecision, ObjectReviewStatus, RegionAnnotation
    from src.review.session import ReviewSession
    from src.review.storage import save_session
    from tests.test_review_finalization import MODEL_CALLS

    view = project.for_reference(reviewer)
    session = ReviewSession(
        project_id=project.project_id, model_version=project.model_version, reviewer=reviewer
    )
    session.record_input_fingerprints("one", image_input_fingerprints(project.images[0]))
    for fiber_id, label in labels.items():
        session.record_fiber_type_decision(
            FiberTypeDecision(
                image_id="one",
                fiber_id=fiber_id,
                model_fiber_type=MODEL_CALLS[fiber_id],
                reviewed_fiber_type=label,
                review_status=ObjectReviewStatus.CORRECTED,
                queue_source="full_audit",
                reviewer=reviewer,
            )
        )
    for index, (x0, y0, x1, y1) in enumerate(fields, start=1):
        ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        session.add_region(
            RegionAnnotation(
                region_id=f"{reviewer}-{index}",
                image_id="one",
                geometry={"type": "Polygon", "coordinates": [ring]},
                domain="fiber_typing",
                action="analysis_roi",
                reason_code="",
                kind="analysis_roi",
                name=f"field_{index}",
                role="reference_field",
            )
        )
    save_session(view.review_state_path, session)
    return view


def test_reference_export_reports_fields_agreement_and_keeps_model_separate(tmp_path):
    from src.review.reference import export_reference

    project = _project(tmp_path)
    # Fibers 1 and 2 sit in the top half (y < 10); the field covers exactly those two.
    _blind_session(project, "a", {1: "iia", 2: "iix", 3: "i"}, fields=[(0, 0, 20, 10)])
    _blind_session(project, "b", {1: "iia", 2: "iib"}, fields=[(0, 0, 20, 10), (0, 10, 20, 20)])

    manifest = export_reference(project, tmp_path / "reference_export")
    labels = pd.read_csv(tmp_path / "reference_export" / "reference_labels.csv")
    fields = pd.read_csv(tmp_path / "reference_export" / "reference_fields.csv")
    agreement = pd.read_csv(tmp_path / "reference_export" / "reviewer_agreement.csv")

    assert manifest["reviewers"] == {"a": 3, "b": 2}
    assert manifest["n_fibers_labelled"] == 3
    a1 = labels[(labels.reviewer == "a") & (labels.fiber_id == 1)].iloc[0]
    assert (a1.reference_label, a1.model_fiber_type, a1.sampling) == (
        "iia",
        "iia",
        "drawn_field",
    )
    a3 = labels[(labels.reviewer == "a") & (labels.fiber_id == 3)].iloc[0]
    assert a3.sampling == "whole_section" and bool(a3.inputs_unchanged)
    assert labels["blind"].all()
    by_field = fields.set_index(["reviewer", "field_name"])
    assert bool(by_field.loc[("a", "field_1"), "exhaustive"])
    assert bool(by_field.loc[("b", "field_1"), "exhaustive"])
    assert not bool(by_field.loc[("b", "field_2"), "exhaustive"])  # fibers 3 and 4 unlabelled
    assert agreement.loc[0, "n_shared_fibers"] == 2
    assert agreement.loc[0, "agreement"] == 0.5
    assert agreement.loc[0, "cohen_kappa"] == pytest.approx(1 / 3)
    assert not project.review_state_path.exists()


def test_reference_export_needs_blind_labels(tmp_path):
    from src.review.reference import export_reference

    with pytest.raises(ValueError, match="No blind reference labels"):
        export_reference(_project(tmp_path), tmp_path / "out")


def test_blind_field_queue_and_saved_sample_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qtpy = pytest.importorskip("qtpy.QtWidgets")
    from src.review.fiber_type_review import FiberTypeReviewController
    from src.review.guided_review_widget import GuidedReviewWidget
    from src.review.session import ReviewSession

    application = qtpy.QApplication.instance() or qtpy.QApplication([])
    reference = _project(tmp_path).for_reference("rev")

    def new_widget():
        session = ReviewSession(project_id=reference.project_id, model_version="model.v1")
        session.current_image_id = "one"
        return GuidedReviewWidget(
            reference,
            FiberTypeReviewController(session),
            blind=True,
            reference_field_fibers=lambda image_id, capture: {2, 4},
        )

    widget = new_widget()
    widget.start_field_review()
    assert [item.fiber_id for item in widget.controller.queue] == [2, 4]
    assert widget.controller.session.active_queue == "reference_fields"
    widget.seed_spin.setValue(7)
    widget.sample_spin.setValue(3)
    widget.start_random_sample()
    sample = [item.fiber_id for item in widget.controller.queue]
    widget.close()

    reopened = new_widget()
    reopened.start_random_sample()
    application.processEvents()
    assert (reopened.seed_spin.value(), reopened.sample_spin.value()) == (7, 3)
    assert [item.fiber_id for item in reopened.controller.queue] == sample
    reopened.close()


def test_roles_limit_blind_labelling_to_test_mice_and_review_to_pool_mice(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qtpy = pytest.importorskip("qtpy.QtWidgets")
    from src.review.fiber_type_review import FiberTypeReviewController
    from src.review.guided_review_widget import GuidedReviewWidget
    from src.review.roles import allowed_images, load_roles
    from src.review.session import ReviewSession

    application = qtpy.QApplication.instance() or qtpy.QApplication([])
    folder = tmp_path / "project"
    (folder / "review").mkdir(parents=True)
    project = _project(folder / "review", image_ids=("one", "two"))
    assert load_roles(project) == {} and allowed_images(project, "test") is None
    (folder / "evaluation_roles.csv").write_text("mouse_id,role\nmouse_a,pool\n")
    assert allowed_images(project, "pool") == {"one", "two"}
    assert allowed_images(project, "test") == set()

    session = ReviewSession(project_id=project.project_id, model_version="model.v1")
    session.current_image_id = "one"
    blind = GuidedReviewWidget(
        project.for_reference("r"), FiberTypeReviewController(session), blind=True
    )
    blind.start_section_review()
    application.processEvents()
    assert blind.rows.empty and not blind.controller.queue
    assert "limited to test mice" in blind.status.text()
    blind.close()

    (folder / "evaluation_roles.csv").write_text("mouse_id,role\nmouse_a,sealed\n")
    with pytest.raises(ValueError, match="'pool' or 'test'"):
        load_roles(project)


def test_computer_placed_field_is_reproducible_and_holds_the_requested_fibers():
    from src.review.reference import fibers_by_field, random_field_polygon
    from src.review.schemas import RegionAnnotation

    rng = np.random.default_rng(1)
    centroids = {i + 1: tuple(map(float, rng.uniform(0, 1000, 2))) for i in range(600)}

    first = random_field_polygon(centroids, "project:image:0", n_fibers=100)
    again = random_field_polygon(centroids, "project:image:0", n_fibers=100)
    other = random_field_polygon(centroids, "project:image:1", n_fibers=100)

    def members(geometry):
        field = RegionAnnotation(
            region_id="r",
            image_id="image",
            geometry=geometry,
            domain="fiber_typing",
            action="analysis_roi",
            reason_code="",
            kind="analysis_roi",
            name="auto_field_1",
            role="reference_field",
        )
        return set(fibers_by_field([field], centroids))

    assert first == again and first != other
    assert 99 <= len(members(first)) <= 101
    assert len(members(random_field_polygon(centroids, "s", n_fibers=5000))) == 600
