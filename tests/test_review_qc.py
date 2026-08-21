from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml

from src.generate_review_qc import main as generate_qc_main
from src.review.project import Project, ProjectImage, load_project
from src.review.qc import generate_project_qc
from src.review.qc_rules import DEFAULT_RULES_PATH, RuleSeverity, load_rule_config
from src.review.schemas import Domain
from src.review.section_selection import select_sections


def _write_complete_project(tmp_path: Path) -> Path:
    (tmp_path / "panel.yaml").write_text("channels: {}\n", encoding="utf-8")
    (tmp_path / "raw.tif").write_bytes(b"synthetic")
    prediction = tmp_path / "predictions" / "image_1"
    prediction.mkdir(parents=True)

    fiber_labels = np.zeros((4, 5), dtype=np.int32)
    fiber_labels[1:3, 1:3] = 1
    fiber_labels[1:3, 3:5] = 2
    tifffile.imwrite(prediction / "fiber_labels.tif", fiber_labels)
    pd.DataFrame(
        {
            "fiber_id": [1, 2],
            "fiber_type": ["iib", "iib"],
            "prob_iib": [0.8, 0.5],
            "prob_iia": [0.2, 0.5],
            "needs_review": [False, True],
        }
    ).to_csv(prediction / "fibers.csv", index=False)

    nuclei_labels = np.zeros((4, 5), dtype=np.int32)
    nuclei_labels[0, 0] = 1
    nuclei_labels[0, 2] = 2
    nuclei_labels[2, 0] = 3
    nuclei_labels[3, 4] = 4
    tifffile.imwrite(prediction / "nuclei_labels.tif", nuclei_labels)
    pd.DataFrame(
        {
            "nucleus_id": [1, 2, 3, 4],
            "assigned_fiber_id": [1, 2, 1, 0],
            "assignment_status": [
                "assigned",
                "assigned",
                "ambiguous",
                "unassigned_or_interstitial",
            ],
            "association_category": [
                "central_interior",
                "boundary_associated",
                "ambiguous",
                "unassigned_or_interstitial",
            ],
            "overlap_fraction": [1.0, 0.75, 0.25, 0.0],
            "distance_to_boundary_px": [2.0, 1.0, np.nan, np.nan],
            "normalized_radial_position": [0.8, 0.2, np.nan, np.nan],
        }
    ).to_csv(prediction / "nuclei.csv", index=False)

    manifest = {
        "schema_version": "review_project.v1",
        "project_id": "synthetic_qc",
        "project_name": "Synthetic QC",
        "panel_manifest": "panel.yaml",
        "model_version": "model.v1",
        "images": [
            {
                "image_id": "image_1",
                "mouse_id": "mouse_1",
                "section_id": "section_1",
                "raw_image_path": "raw.tif",
                "prediction_directory": "predictions/image_1",
                "applicable_domains": [
                    "fiber_segmentation",
                    "fiber_typing",
                    "nuclei",
                ],
                "outputs": {
                    "fiber_labels": "fiber_labels.tif",
                    "fiber_table": "fibers.csv",
                    "nuclei_labels": "nuclei_labels.tif",
                    "nuclei_table": "nuclei.csv",
                },
            }
        ],
    }
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def _domain_row(result, domain: Domain) -> pd.Series:
    return result.image_qc[result.image_qc["domain"].eq(domain.value)].iloc[0]


def _write_custom_rules(tmp_path: Path) -> Path:
    path = tmp_path / "custom_rules.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "review_qc_rules.v1",
                "rules_version": "custom_test_rules.v1",
                "qc_version": "custom_qc.v1",
                "rules": [
                    {
                        "reason_code": "fiber_segmentation.custom_border_review",
                        "domain": "fiber_segmentation",
                        "metric": "border_touching_fiber_fraction",
                        "operator": "gt",
                        "threshold": 0.4,
                        "severity": "review",
                        "enabled": True,
                        "description": "Synthetic test-only threshold.",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_qc_metrics_use_documented_denominators(tmp_path: Path) -> None:
    project = load_project(_write_complete_project(tmp_path))
    rules = load_rule_config()

    result = generate_project_qc(project, rules)

    segmentation = _domain_row(result, Domain.FIBER_SEGMENTATION)
    assert segmentation["status"] == "pass"
    assert segmentation["fiber_count"] == 2
    assert segmentation["segmented_image_fraction"] == pytest.approx(8 / 20)
    assert segmentation["median_fiber_area_px"] == pytest.approx(4.0)
    assert segmentation["border_touching_fiber_fraction"] == pytest.approx(0.5)
    assert segmentation["fiber_id_mismatch_fraction"] == pytest.approx(0.0)

    typing = _domain_row(result, Domain.FIBER_TYPING)
    assert typing["status"] == "pass"
    assert typing["mean_max_probability"] == pytest.approx(0.65)
    assert typing["mean_probability_margin"] == pytest.approx(0.30)
    assert typing["probability_coverage"] == pytest.approx(1.0)
    expected_entropy = (
        -(0.8 * math.log(0.8) + 0.2 * math.log(0.2)) / math.log(2) + 1.0
    ) / 2
    assert typing["mean_normalized_entropy"] == pytest.approx(expected_entropy)
    assert typing["needs_review_fraction"] == pytest.approx(0.5)
    assert not typing["hard_fail"]

    nuclei = _domain_row(result, Domain.NUCLEI)
    assert nuclei["status"] == "pass"
    assert nuclei["nucleus_count"] == 4
    assert nuclei["unassigned_nucleus_fraction"] == pytest.approx(0.25)
    assert nuclei["ambiguous_nucleus_fraction"] == pytest.approx(0.25)
    assert nuclei["assigned_nuclei_per_fiber"] == pytest.approx(1.0)
    assert nuclei["mean_association_overlap"] == pytest.approx(0.5)
    assert len(result.fiber_qc) == 2
    assert len(result.nucleus_qc) == 4


def test_extreme_composition_does_not_trigger_technical_failure(tmp_path: Path) -> None:
    project = load_project(_write_complete_project(tmp_path))

    result = generate_project_qc(project, load_rule_config())
    typing = _domain_row(result, Domain.FIBER_TYPING)
    nuclei = _domain_row(result, Domain.NUCLEI)

    assert typing["type_counts_json"] == '{"iib":2}'
    assert typing["status"] == "pass"
    assert nuclei["status"] == "pass"
    assert not typing["hard_fail"]
    assert not nuclei["hard_fail"]


def test_missing_required_typing_table_fails_only_typing_domain(tmp_path: Path) -> None:
    path = _write_complete_project(tmp_path)
    manifest = yaml.safe_load(path.read_text())
    manifest["images"][0]["outputs"]["fiber_table"] = "missing.csv"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    project = load_project(path, validate_paths=False)

    result = generate_project_qc(project, load_rule_config())

    segmentation = _domain_row(result, Domain.FIBER_SEGMENTATION)
    typing = _domain_row(result, Domain.FIBER_TYPING)
    assert not segmentation["hard_fail"]
    assert typing["hard_fail"]
    assert typing["status"] == "fail"
    assert "fiber_typing.missing_table" in typing["reason_codes"]


def test_non_applicable_missing_artifacts_do_not_fail(tmp_path: Path) -> None:
    path = _write_complete_project(tmp_path)
    manifest = yaml.safe_load(path.read_text())
    manifest["images"][0]["applicable_domains"] = ["fiber_segmentation"]
    manifest["images"][0]["outputs"] = {
        "fiber_labels": "fiber_labels.tif",
    }
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    project = load_project(path)

    result = generate_project_qc(project, load_rule_config())

    assert _domain_row(result, Domain.FIBER_TYPING)["status"] == "not_applicable"
    assert _domain_row(result, Domain.NUCLEI)["status"] == "not_applicable"


def test_empty_nucleus_mask_routes_to_review_not_hard_failure(tmp_path: Path) -> None:
    path = _write_complete_project(tmp_path)
    project = load_project(path)
    image = project.image("image_1")
    tifffile.imwrite(image.outputs["nuclei_labels"], np.zeros((4, 5), dtype=np.int32))
    pd.DataFrame(
        columns=[
            "nucleus_id",
            "assigned_fiber_id",
            "assignment_status",
            "overlap_fraction",
        ]
    ).to_csv(image.outputs["nuclei_table"], index=False)

    result = generate_project_qc(project, load_rule_config())
    nuclei = _domain_row(result, Domain.NUCLEI)

    assert nuclei["status"] == "review"
    assert not nuclei["hard_fail"]
    assert "nuclei.no_objects" in nuclei["reason_codes"]


def test_default_rules_enable_only_structural_hard_failures() -> None:
    config = load_rule_config(DEFAULT_RULES_PATH)

    hard_fail_codes = {
        rule.reason_code
        for rule in config.rules
        if rule.enabled and rule.severity is RuleSeverity.HARD_FAIL
    }
    assert "fiber_segmentation.missing_labels" in hard_fail_codes
    assert "fiber_typing.missing_table" in hard_fail_codes
    assert "nuclei.label_shape_mismatch" in hard_fail_codes
    assert all(
        token not in code
        for code in hard_fail_codes
        for token in ("composition", "central", "area", "unknown", "entropy")
    )
    assert any(
        rule.reason_code == "fiber_typing.high_entropy" and not rule.enabled
        for rule in config.rules
    )


def test_versioned_custom_rule_can_change_review_status(tmp_path: Path) -> None:
    project = load_project(_write_complete_project(tmp_path))
    rules = load_rule_config(_write_custom_rules(tmp_path))

    result = generate_project_qc(project, rules)
    segmentation = _domain_row(result, Domain.FIBER_SEGMENTATION)

    assert segmentation["qc_version"] == "custom_qc.v1"
    assert segmentation["rules_version"] == "custom_test_rules.v1"
    assert segmentation["status"] == "review"
    assert (
        segmentation["reason_codes"]
        == "fiber_segmentation.custom_border_review"
    )


def _selection_project(tmp_path: Path) -> Project:
    images = (
        ProjectImage(
            "pass_section",
            "mouse_1",
            "section_1",
            tmp_path / "raw1",
            tmp_path / "pred1",
        ),
        ProjectImage(
            "review_section",
            "mouse_1",
            "section_2",
            tmp_path / "raw2",
            tmp_path / "pred2",
        ),
        ProjectImage(
            "failed_section",
            "mouse_1",
            "section_3",
            tmp_path / "raw3",
            tmp_path / "pred3",
        ),
        ProjectImage(
            "only_failed",
            "mouse_2",
            "section_1",
            tmp_path / "raw4",
            tmp_path / "pred4",
        ),
    )
    return Project(
        schema_version="review_project.v1",
        project_id="selection",
        project_name="Selection",
        panel_manifest=tmp_path / "panel.yaml",
        model_version="model.v1",
        manifest_path=tmp_path / "project.yaml",
        images=images,
    )


def _selection_image_qc(project: Project) -> pd.DataFrame:
    rows = []
    dispositions = {
        "pass_section": (False, 1.0, 0),
        "review_section": (False, 0.5, 10),
        "failed_section": (True, 0.0, 100),
        "only_failed": (True, 0.0, 100),
    }
    for image in project.images:
        for domain in Domain:
            hard_fail, score, priority = dispositions[image.image_id]
            rows.append(
                {
                    "image_id": image.image_id,
                    "mouse_id": image.mouse_id,
                    "domain": domain.value,
                    "applicable": domain is Domain.FIBER_SEGMENTATION,
                    "hard_fail": hard_fail,
                    "technical_quality_score": score,
                    "review_priority": priority,
                    "qc_version": "review_qc.v1",
                    "rules_version": "rules.v1",
                    "computed_at": "2026-01-01T00:00:00+00:00",
                }
            )
    return pd.DataFrame(rows)


def test_section_selection_strategies_and_no_passing_case(tmp_path: Path) -> None:
    project = _selection_project(tmp_path)
    image_qc = _selection_image_qc(project)

    all_passing = select_sections(project, image_qc, strategy="all_passing")
    mouse_1 = all_passing[
        all_passing["mouse_id"].eq("mouse_1")
        & all_passing["domain"].eq("fiber_segmentation")
    ].iloc[0]
    assert mouse_1["selected_image_ids"] == "pass_section|review_section"

    best = select_sections(project, image_qc, strategy="best_passing")
    mouse_1_best = best[
        best["mouse_id"].eq("mouse_1")
        & best["domain"].eq("fiber_segmentation")
    ].iloc[0]
    assert mouse_1_best["selected_image_ids"] == "pass_section"

    manual = select_sections(
        project,
        image_qc,
        strategy="manual",
        manual_selections={
            "mouse_1": {"fiber_segmentation": ["review_section"]}
        },
    )
    mouse_1_manual = manual[
        manual["mouse_id"].eq("mouse_1")
        & manual["domain"].eq("fiber_segmentation")
    ].iloc[0]
    assert mouse_1_manual["selected_image_ids"] == "review_section"

    mouse_2 = all_passing[
        all_passing["mouse_id"].eq("mouse_2")
        & all_passing["domain"].eq("fiber_segmentation")
    ].iloc[0]
    assert mouse_2["selected_image_ids"] == ""
    assert bool(mouse_2["requires_manual_review"])
    assert mouse_2["reason_code"] == "no_passing_sections"


def test_cli_writes_all_phase_2a_outputs(tmp_path: Path) -> None:
    project_path = _write_complete_project(tmp_path)
    output_dir = tmp_path / "generated_qc"
    custom_rules = _write_custom_rules(tmp_path)

    exit_code = generate_qc_main(
        [
            "--project",
            str(project_path),
            "--rules",
            str(custom_rules),
            "--output-dir",
            str(output_dir),
            "--selection-strategy",
            "best_passing",
        ]
    )

    assert exit_code == 0
    for filename in (
        "image_qc.csv",
        "fiber_qc.csv",
        "nucleus_qc.csv",
        "section_selection.csv",
    ):
        assert (output_dir / filename).is_file()
    image_qc = pd.read_csv(output_dir / "image_qc.csv")
    assert set(image_qc["schema_version"]) == {"review_image_qc.v1"}
    assert set(image_qc["rules_version"]) == {"custom_test_rules.v1"}
    selections = pd.read_csv(output_dir / "section_selection.csv")
    assert set(selections["strategy"]) == {"best_passing"}
