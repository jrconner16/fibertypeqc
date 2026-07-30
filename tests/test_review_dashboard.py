from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.review.dashboard import build_dashboard_model, load_dashboard_tables
from src.review.project import load_project
from src.review.qc import IMAGE_QC_SCHEMA_VERSION
from src.review.schemas import (
    Domain,
    DomainStatus,
    FiberTypeDecision,
    ObjectReviewStatus,
)
from src.review.section_selection import select_sections
from src.review.session import ReviewSession


def _write_dashboard_fixture(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "panel.yaml").write_text("channels: {}\n", encoding="utf-8")
    images = [
        ("m1_s1", "mouse_1", "section_1"),
        ("m1_s2", "mouse_1", "section_2"),
        ("m2_s1", "mouse_2", "section_1"),
        ("m3_s1", "mouse_3", "section_1"),
    ]
    manifest = {
        "schema_version": "review_project.v1",
        "project_id": "dashboard_project",
        "project_name": "Dashboard fixture",
        "panel_manifest": "panel.yaml",
        "model_version": "model.v1",
        "images": [
            {
                "image_id": image_id,
                "mouse_id": mouse_id,
                "section_id": section_id,
                "raw_image_path": f"raw/{image_id}.tif",
                "prediction_directory": f"predictions/{image_id}",
                "applicable_domains": ["fiber_segmentation"],
            }
            for image_id, mouse_id, section_id in images
        ],
    }
    project_path = tmp_path / "project.yaml"
    project_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    dispositions = {
        ("m1_s1", "fiber_segmentation"): (True, "pass", False, 1.0, 0, ""),
        ("m1_s1", "fiber_typing"): (
            True,
            "review",
            False,
            0.5,
            10,
            "fiber_typing.probabilities_unavailable",
        ),
        ("m1_s1", "nuclei"): (True, "pass", False, 1.0, 0, ""),
        ("m1_s2", "fiber_segmentation"): (True, "pass", False, 1.0, 0, ""),
        ("m1_s2", "fiber_typing"): (True, "pass", False, 1.0, 0, ""),
        ("m1_s2", "nuclei"): (
            True,
            "fail",
            True,
            0.0,
            100,
            "nuclei.missing_labels",
        ),
        ("m2_s1", "fiber_segmentation"): (
            True,
            "fail",
            True,
            0.0,
            100,
            "fiber_segmentation.missing_labels",
        ),
        ("m2_s1", "fiber_typing"): (
            False,
            "not_applicable",
            False,
            None,
            0,
            "",
        ),
        ("m2_s1", "nuclei"): (False, "not_applicable", False, None, 0, ""),
        ("m3_s1", "fiber_segmentation"): (True, "pass", False, 1.0, 0, ""),
        ("m3_s1", "fiber_typing"): (
            False,
            "not_applicable",
            False,
            None,
            0,
            "",
        ),
        ("m3_s1", "nuclei"): (False, "not_applicable", False, None, 0, ""),
    }
    rows = []
    metadata = {image_id: (mouse_id, section_id) for image_id, mouse_id, section_id in images}
    for image_id, _, _ in images:
        for domain in Domain:
            applicable, status, hard_fail, score, priority, reasons = dispositions[
                (image_id, domain.value)
            ]
            rows.append(
                {
                    "schema_version": IMAGE_QC_SCHEMA_VERSION,
                    "qc_version": "review_qc.v1",
                    "rules_version": "rules.v1",
                    "model_version": "model.v1",
                    "computed_at": "2026-07-29T00:00:00+00:00",
                    "project_id": "dashboard_project",
                    "image_id": image_id,
                    "mouse_id": metadata[image_id][0],
                    "section_id": metadata[image_id][1],
                    "domain": domain.value,
                    "applicable": applicable,
                    "status": status,
                    "hard_fail": hard_fail,
                    "technical_quality_score": score,
                    "review_priority": priority,
                    "reason_codes": reasons,
                    "reason_details_json": "[]",
                }
            )
    qc_directory = tmp_path / "qc"
    qc_directory.mkdir()
    pd.DataFrame(rows).to_csv(qc_directory / "image_qc.csv", index=False)
    return project_path, qc_directory


def test_dashboard_model_summarizes_readiness_progress_and_reasons(
    tmp_path: Path,
) -> None:
    project_path, qc_directory = _write_dashboard_fixture(tmp_path)
    project = load_project(project_path, validate_paths=False)
    tables = load_dashboard_tables(project, qc_directory)
    session = ReviewSession(project_id=project.project_id, model_version=project.model_version)
    session.set_status("m1_s1", Domain.FIBER_SEGMENTATION, DomainStatus.PASS)
    session.set_status("m3_s1", Domain.FIBER_SEGMENTATION, DomainStatus.PASS)
    session.record_fiber_type_decision(
        FiberTypeDecision(
            image_id="m1_s1",
            fiber_id=1,
            model_fiber_type="iib",
            reviewed_fiber_type="iia",
            review_status=ObjectReviewStatus.CORRECTED,
        )
    )

    model = build_dashboard_model(
        project,
        tables,
        strategy="all_passing",
        session=session,
    )

    assert model.summary.mouse_count == 3
    assert model.summary.section_count == 4
    assert model.summary.complete_mouse_count == 1
    assert model.summary.targeted_review_mouse_count == 1
    assert model.summary.no_acceptable_mouse_count == 1
    assert model.summary.applicable_domain_rows == 8
    assert model.summary.reviewed_domain_rows == 2
    assert model.summary.review_progress_fraction == pytest.approx(0.25)
    assert model.summary.object_decision_count == 1
    assert dict(zip(model.mouse_rows["mouse_id"], model.mouse_rows["readiness"], strict=True)) == {
        "mouse_1": "targeted_review",
        "mouse_2": "no_acceptable_section",
        "mouse_3": "complete",
    }
    typing = model.mouse_domain_rows[
        model.mouse_domain_rows["mouse_id"].eq("mouse_1")
        & model.mouse_domain_rows["domain"].eq("fiber_typing")
    ].iloc[0]
    assert typing["readiness"] == "targeted_review"
    assert typing["selected_image_ids"] == "m1_s1|m1_s2"
    reason_row = model.section_rows[
        model.section_rows["image_id"].eq("m1_s1") & model.section_rows["domain"].eq("fiber_typing")
    ].iloc[0]
    assert reason_row["reason_codes"] == "fiber_typing.probabilities_unavailable"


def test_dashboard_best_passing_avoids_review_section(tmp_path: Path) -> None:
    project_path, qc_directory = _write_dashboard_fixture(tmp_path)
    project = load_project(project_path, validate_paths=False)
    tables = load_dashboard_tables(project, qc_directory)

    model = build_dashboard_model(project, tables, strategy="best_passing")

    typing = model.mouse_domain_rows[
        model.mouse_domain_rows["mouse_id"].eq("mouse_1")
        & model.mouse_domain_rows["domain"].eq("fiber_typing")
    ].iloc[0]
    assert typing["selected_image_ids"] == "m1_s2"
    assert typing["readiness"] == "complete"


def test_dashboard_loader_rejects_invalid_boolean_and_missing_rows(tmp_path: Path) -> None:
    project_path, qc_directory = _write_dashboard_fixture(tmp_path)
    project = load_project(project_path, validate_paths=False)
    path = qc_directory / "image_qc.csv"
    table = pd.read_csv(path)
    table["applicable"] = table["applicable"].astype(object)
    table.loc[0, "applicable"] = "not-a-boolean"
    table.to_csv(path, index=False)

    with pytest.raises(ValueError, match="invalid boolean"):
        load_dashboard_tables(project, qc_directory)

    _, qc_directory = _write_dashboard_fixture(tmp_path / "second")
    second_project = load_project(tmp_path / "second/project.yaml", validate_paths=False)
    second_path = qc_directory / "image_qc.csv"
    second_table = pd.read_csv(second_path).iloc[:-1]
    second_table.to_csv(second_path, index=False)
    with pytest.raises(ValueError, match="missing image/domain rows"):
        load_dashboard_tables(second_project, qc_directory)


def test_dashboard_launcher_import_does_not_import_napari() -> None:
    sys.modules.pop("src.review_project_napari", None)
    napari_was_loaded = "napari" in sys.modules

    import src.review_project_napari as launcher

    args = launcher.build_parser().parse_args(
        ["--project", "project.yaml", "--display-downsample", "2"]
    )
    assert args.project
    assert args.display_downsample == 2
    assert ("napari" in sys.modules) is napari_was_loaded


def test_dashboard_loader_rejects_cross_table_provenance_mismatch(
    tmp_path: Path,
) -> None:
    project_path, qc_directory = _write_dashboard_fixture(tmp_path)
    project = load_project(project_path, validate_paths=False)
    tables = load_dashboard_tables(project, qc_directory)
    selection = select_sections(project, tables.image_qc)
    selection["qc_version"] = "different_qc.v1"
    selection.to_csv(qc_directory / "section_selection.csv", index=False)

    with pytest.raises(ValueError, match="does not match image_qc.csv"):
        load_dashboard_tables(project, qc_directory)


def test_dashboard_widget_offscreen_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qtpy = pytest.importorskip("qtpy.QtWidgets")
    from src.review.dashboard_widget import CohortDashboardWidget

    application = qtpy.QApplication.instance() or qtpy.QApplication([])
    project_path, qc_directory = _write_dashboard_fixture(tmp_path)
    project = load_project(project_path, validate_paths=False)
    tables = load_dashboard_tables(project, qc_directory)

    widget = CohortDashboardWidget(project, tables)

    assert widget.section_tree.topLevelItemCount() == 3
    assert "Mice: 3" in widget.summary_label.text()
    widget.strategy_combo.setCurrentText("best_passing")
    application.processEvents()
    assert widget.model.strategy.value == "best_passing"
    widget.close()
    widget.deleteLater()
