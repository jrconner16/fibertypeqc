"""The staged review panel: section verdicts and areas, clicking a fiber, and blind labelling."""

from __future__ import annotations

import pandas as pd
import pytest

from src.review.fiber_type_review import FiberTypeReviewController
from src.review.finalization import finalize_image
from src.review.image_review import ImageReviewController
from src.review.region_review import RegionReviewController
from src.review.schemas import Domain, DomainStatus, ObjectReviewStatus, RegionAction
from tests.test_review_finalization import _box, _project, _session


def _image_qc(project) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_id": image.image_id,
                "domain": domain.value,
                "applicable": domain is not Domain.NUCLEI,
                "status": "review" if domain is Domain.FIBER_TYPING else "pass",
                "reason_codes": "image.typing.low_confidence_share"
                if domain is Domain.FIBER_TYPING
                else "",
                "review_priority": 1,
            }
            for image in project.images
            for domain in Domain
        ]
    )


@pytest.fixture
def qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("qtpy.QtWidgets")
    return widgets.QApplication.instance() or widgets.QApplication([])


def _stage(project, session, geometry=None):
    from src.review.section_stage_widget import SectionStageWidget

    return SectionStageWidget(
        ImageReviewController(project, _image_qc(project), session),
        RegionReviewController(project, session),
        drawn_geometry=lambda: geometry,
    )


def test_section_verdicts_set_the_statuses_finalization_reads(tmp_path, qt):
    project = _project(tmp_path)
    session = _session(project)
    stage = _stage(project, session)
    assert "suggest a look" in stage.qc.text() and "low confidence share" in stage.qc.text()
    assert "not checked yet" in stage.verdict.text()

    stage.fine_button.click()
    for domain in (Domain.FIBER_SEGMENTATION, Domain.FIBER_TYPING):
        assert session.get_status("one", domain) is DomainStatus.PASS
    assert "looks fine" in stage.verdict.text()

    stage.section_button.click()
    assert session.get_status("one", Domain.FIBER_SEGMENTATION) is DomainStatus.EXCLUDED
    final = finalize_image(project, session, project.images[0]).table
    assert set(final["value_source"]) == {"excluded"}

    stage.clear_button.click()
    assert session.get_status("one", Domain.FIBER_TYPING) is DomainStatus.NOT_REVIEWED
    assert "not checked yet" in stage.verdict.text()
    stage.close()


def test_an_area_left_out_excludes_the_fibers_inside_it(tmp_path, qt):
    project = _project(tmp_path)
    session = _session(project)
    stage = _stage(project, session, geometry=_box(0, 0, 10, 10))  # covers fiber 1 only
    stage.begin_area()
    assert stage.save_area_button.isVisibleTo(stage)
    stage.save_area()
    [region] = session.regions
    assert region.action == RegionAction.EXCLUDE_ALL_ANALYSIS.value
    assert "1 area left out" in stage.areas.text()
    table = finalize_image(project, session, project.images[0]).table.set_index("label")
    assert table.loc[1, "value_source"] == "excluded"
    assert (table.drop(index=1)["value_source"] != "excluded").all()

    stage.remove_last_area()
    assert session.regions == []
    stage.close()


def test_a_clicked_fiber_is_checked_then_the_plan_resumes(tmp_path, qt):
    from src.review.guided_review_widget import CLICKED_QUEUE, GuidedReviewWidget

    project = _project(tmp_path)
    session = _session(project)
    session.current_image_id = "one"
    centered = []  # whether the view would re-center for each fiber shown
    widget = GuidedReviewWidget(
        project,
        FiberTypeReviewController(session),
        object_changed=lambda _image, fiber: centered.append((fiber, widget.center_on_change)),
    )
    widget.start_flagged_review()
    plan = widget.controller.session.active_queue
    position = widget.controller.session.queue_position

    assert widget.select_fiber("one", 2)
    assert centered[-1] == (2, False)  # a clicked fiber is already on screen: no re-centering
    assert widget.center_on_change
    assert widget.controller.session.active_queue == CLICKED_QUEUE
    assert widget.controller.current_item.fiber_id == 2
    widget._record_type("iia")
    [decision] = [d for d in session.object_decisions if d.fiber_id == 2]
    assert decision.reviewed_fiber_type == "iia"
    assert decision.review_status is ObjectReviewStatus.CORRECTED
    assert decision.reason_code == CLICKED_QUEUE
    # Back on the plan, where it was.
    assert widget.controller.session.active_queue == plan
    assert widget.controller.session.queue_position == position

    # Arrows also leave a clicked fiber without deciding it.
    widget.select_fiber("one", 3)
    widget._move(1)
    assert widget.controller.session.active_queue == plan
    widget.close()


def test_clicking_selects_nothing_in_blind_labelling(tmp_path, qt):
    from src.review.guided_review_widget import GuidedReviewWidget

    project = _project(tmp_path)
    reference = project.for_reference("rev")
    session = _session(reference)
    session.current_image_id = "one"
    widget = GuidedReviewWidget(reference, FiberTypeReviewController(session), blind=True)
    assert not widget.select_fiber("one", 2)
    assert widget.stages.count() == 1  # one stage, no Section verdicts or Finish
    widget.close()


def test_stages_and_section_bar(tmp_path, qt):
    from src.review.guided_review_widget import GuidedReviewWidget

    project = _project(tmp_path, image_ids=("one", "two"))
    session = _session(project)
    session.current_image_id = "one"
    shown = []
    stage = _stage(project, session)
    widget = GuidedReviewWidget(
        project,
        FiberTypeReviewController(session),
        finalize_requested=lambda: None,
        show_image=lambda image_id: (setattr(session, "current_image_id", image_id),
                                     shown.append(image_id)),
        section_stage=stage,
    )  # fmt: skip
    assert [widget.stages.tabText(i) for i in range(widget.stages.count())] == [
        "1  Section",
        "2  Fibers",
        "3  Finish",
    ]
    assert widget.stages.currentWidget() is stage  # a fresh review starts with the section
    assert widget.section_label.text().startswith("Section 1 of 2 · mouse_a · one")
    widget.next_section_bar_button.click()
    assert shown == ["two"] and widget.section_label.text().startswith("Section 2 of 2")
    stage.fine_button.click()
    widget.stages.setCurrentWidget(widget.finish_page)
    assert "1 look fine, 0 left out, 1 not checked yet (of 2)" in widget.finish_summary.text()
    widget.start_flagged_review()
    assert widget.stages.currentWidget() is widget.fibers_page
    widget.close()
