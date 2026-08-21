from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.review.image_review import ImageReviewController
from src.review.project import load_project
from src.review.schemas import Domain, DomainStatus, ReviewMode
from src.review.session import ReviewSession
from src.review.storage import load_session


def _fixture(tmp_path: Path) -> tuple[object, pd.DataFrame]:
    (tmp_path / "panel.yaml").write_text("channels: {}\n", encoding="utf-8")
    manifest = {
        "schema_version": "review_project.v1",
        "project_id": "phase3",
        "project_name": "Phase 3",
        "panel_manifest": "panel.yaml",
        "model_version": "model.v1",
        "qc_version": "qc.v1",
        "images": [
            {
                "image_id": "one",
                "mouse_id": "m",
                "section_id": "s1",
                "raw_image_path": "one.tif",
                "prediction_directory": "pred/one",
                "applicable_domains": ["fiber_segmentation", "fiber_typing"],
            },
            {
                "image_id": "two",
                "mouse_id": "m",
                "section_id": "s2",
                "raw_image_path": "two.tif",
                "prediction_directory": "pred/two",
                "applicable_domains": ["fiber_segmentation"],
            },
        ],
    }
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    project = load_project(path, validate_paths=False)
    rows = []
    statuses = {
        "one": {
            "fiber_segmentation": "pass",
            "fiber_typing": "review",
            "nuclei": "not_applicable",
        },
        "two": {
            "fiber_segmentation": "fail",
            "fiber_typing": "not_applicable",
            "nuclei": "not_applicable",
        },
    }
    for image_id, values in statuses.items():
        for domain, status in values.items():
            rows.append(
                {
                    "image_id": image_id,
                    "domain": domain,
                    "applicable": status != "not_applicable",
                    "status": status,
                    "reason_codes": "flag" if status in {"review", "fail"} else "",
                    "review_priority": 10 if status in {"review", "fail"} else 0,
                }
            )
    return project, pd.DataFrame(rows)


def test_phase3_status_action_saves_snapshot_and_event(tmp_path: Path) -> None:
    project, qc = _fixture(tmp_path)
    session = ReviewSession(
        project_id=project.project_id,
        model_version=project.model_version,
        qc_version="qc.v1",
    )
    controller = ImageReviewController(project, qc, session)
    controller.set_domain(Domain.FIBER_TYPING)
    event = controller.set_status(DomainStatus.PASS, reason_code="checked")
    controller.save(event)

    restored = load_session(project.review_state_path, expected_project_id=project.project_id)
    assert restored.current_image_id == "one"
    assert restored.get_status("one", Domain.FIBER_TYPING) is DomainStatus.PASS
    assert "set_domain_status" in project.review_events_path.read_text(encoding="utf-8")

    controller.set_status(DomainStatus.NOT_APPLICABLE)
    assert session.get_status("one", Domain.FIBER_TYPING) is DomainStatus.NOT_APPLICABLE


def test_phase3_review_modes_have_deterministic_navigation(tmp_path: Path) -> None:
    project, qc = _fixture(tmp_path)
    controller = ImageReviewController(
        project,
        qc,
        ReviewSession(project_id="phase3", model_version="model.v1"),
    )
    controller.set_domain(Domain.FIBER_TYPING)
    controller.set_mode(ReviewMode.FLAGGED_REVIEW)
    assert controller.navigation_ids() == ("one",)
    controller.set_mode(ReviewMode.QC_GATED_AUTOMATIC)
    assert controller.navigation_ids() == ("one",)
    controller.set_domain(Domain.FIBER_SEGMENTATION)
    assert controller.navigation_ids() == ("two",)
    controller.set_mode(ReviewMode.DOMAIN_FOCUSED)
    assert controller.navigation_ids() == ("one", "two")
    controller.set_mode(ReviewMode.FULL_AUDIT)
    assert controller.navigation_ids() == ("one", "two")
