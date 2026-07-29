from pathlib import Path

from src.review.invalidation import EditKind, StaleProduct
from src.review.schemas import (
    Domain,
    DomainStatus,
    FiberTypeDecision,
    ObjectReviewStatus,
    RegionAnnotation,
    ReviewMode,
    Scope,
)
from src.review.session import ReviewSession
from src.review.storage import load_session, save_session


def test_save_reload_preserves_status_region_and_resume_position(tmp_path: Path) -> None:
    session = ReviewSession(
        project_id="project",
        model_version="model.v1",
        qc_version="qc.v1",
        reviewer="reviewer",
        current_image_id="image_1",
        active_domain=Domain.NUCLEI,
        active_scope=Scope.OBJECT,
        active_review_mode=ReviewMode.FULL_AUDIT,
    )
    session.set_queue("weak_dapi_support", position=7)
    session.set_status("image_1", Domain.NUCLEI, DomainStatus.REVIEW)
    session.set_status("image_1", Domain.FIBER_TYPING, DomainStatus.PASS)
    session.add_region(
        RegionAnnotation(
            region_id="region_1",
            image_id="image_1",
            geometry={"type": "Polygon", "coordinates": [[[0, 0], [0, 2], [2, 0], [0, 0]]]},
            domain=Domain.NUCLEI,
            action="ignore_nuclei",
            reason_code="fold",
            reviewer="reviewer",
        )
    )
    state_path = tmp_path / "review_state.json"

    save_session(state_path, session)
    restored = load_session(state_path, expected_project_id="project")

    assert restored.current_image_id == "image_1"
    assert restored.active_queue == "weak_dapi_support"
    assert restored.queue_position == 7
    assert restored.get_status("image_1", Domain.NUCLEI) is DomainStatus.REVIEW
    assert restored.get_status("image_1", Domain.FIBER_TYPING) is DomainStatus.PASS
    assert restored.regions[0].action == "ignore_nuclei"
    assert restored.regions[0].domain is Domain.NUCLEI


def test_domain_specific_exclusions_remain_separate() -> None:
    session = ReviewSession(project_id="project", model_version="model.v1")

    session.set_status("image", Domain.NUCLEI, DomainStatus.EXCLUDED)

    assert session.get_status("image", Domain.NUCLEI) is DomainStatus.EXCLUDED
    assert (
        session.get_status("image", Domain.FIBER_SEGMENTATION)
        is DomainStatus.NOT_REVIEWED
    )
    assert session.get_status("image", Domain.FIBER_TYPING) is DomainStatus.NOT_REVIEWED


def test_fiber_mask_edit_marks_all_downstream_products_stale() -> None:
    session = ReviewSession(project_id="project", model_version="model.v1")

    marked = session.mark_stale("image", EditKind.FIBER_MASK)

    assert marked == {
        StaleProduct.FIBER_GEOMETRY_FEATURES,
        StaleProduct.FIBER_TYPE_PREDICTION_FEATURES,
        StaleProduct.NUCLEUS_ASSOCIATIONS,
        StaleProduct.FIBER_NUCLEUS_COUNTS,
    }
    assert session.stale_products["image"] == sorted(product.value for product in marked)


def test_nucleus_edit_does_not_invalidate_fiber_typing() -> None:
    session = ReviewSession(project_id="project", model_version="model.v1")

    marked = session.mark_stale("image", EditKind.NUCLEUS_MASK)

    assert StaleProduct.NUCLEUS_FEATURES in marked
    assert StaleProduct.NUCLEUS_ASSOCIATIONS in marked
    assert StaleProduct.FIBER_NUCLEUS_COUNTS in marked
    assert StaleProduct.FIBER_TYPE_PREDICTION_FEATURES not in marked


def test_fiber_type_correction_does_not_invalidate_segmentation() -> None:
    session = ReviewSession(project_id="project", model_version="model.v1")

    marked = session.mark_stale("image", EditKind.FIBER_TYPE_CORRECTION)

    assert marked == frozenset()
    assert session.stale_products["image"] == []


def test_fiber_type_decision_preserves_model_prediction_on_update(tmp_path: Path) -> None:
    session = ReviewSession(project_id="project", model_version="model.v1")
    session.record_fiber_type_decision(
        FiberTypeDecision(
            image_id="image",
            fiber_id=12,
            model_fiber_type="iib",
            reviewed_fiber_type="iia",
            review_status=ObjectReviewStatus.CORRECTED,
        )
    )
    session.record_fiber_type_decision(
        FiberTypeDecision(
            image_id="image",
            fiber_id=12,
            model_fiber_type="iib",
            reviewed_fiber_type="iix",
            review_status=ObjectReviewStatus.CORRECTED,
        )
    )
    state_path = tmp_path / "review_state.json"

    save_session(state_path, session)
    restored = load_session(state_path)

    assert restored.object_decisions[0].model_fiber_type == "iib"
    assert restored.object_decisions[0].reviewed_fiber_type == "iix"
