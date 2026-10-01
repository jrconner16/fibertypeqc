from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml

from src.finalize_review_project import main as finalize_main
from src.review.finalization import (
    FinalizationError,
    finalize_image,
    finalize_project,
    image_input_fingerprints,
    import_legacy_reviews,
)
from src.review.project import load_project
from src.review.schemas import FiberTypeDecision, RegionAnnotation
from src.review.session import ReviewSession
from src.review.storage import save_session

# Four square fibers; centroids (x, y): 1=(3.5, 3.5), 2=(13.5, 3.5), 3=(3.5, 13.5), 4=(13.5, 13.5)
SQUARES = {1: (2, 2), 2: (2, 12), 3: (12, 2), 4: (12, 12)}
MODEL_CALLS = {1: "iia", 2: "iib", 3: "i", 4: "iix"}


def _write_image(directory: Path, image_id: str) -> None:
    directory.mkdir(parents=True)
    labels = np.zeros((20, 20), dtype=np.uint16)
    for fiber_id, (y0, x0) in SQUARES.items():
        labels[y0 : y0 + 4, x0 : x0 + 4] = fiber_id
    tifffile.imwrite(directory / f"{image_id}_labels.tif", labels)
    pd.DataFrame(
        {
            "label": list(MODEL_CALLS),
            "fiber_type": list(MODEL_CALLS.values()),
            "prob_iia": [0.9, 0.1, 0.1, 0.2],
            "model_confidence": [0.9, 0.8, 0.6, 0.7],
            "needs_review": [False, False, True, True],
        }
    ).to_csv(directory / f"{image_id}_fibers.csv", index=False)


def _project(tmp_path: Path, image_ids=("one",)):
    (tmp_path / "panel.yaml").write_text("channels: {}\n", encoding="utf-8")
    images = []
    for index, image_id in enumerate(image_ids):
        _write_image(tmp_path / image_id, image_id)
        tifffile.imwrite(
            tmp_path / f"{image_id}.tif", np.zeros((3, 20, 20), np.uint16), photometric="minisblack"
        )
        images.append(
            {
                "image_id": image_id,
                "mouse_id": "mouse_a",
                "section_id": f"s{index + 1}",
                "raw_image_path": f"{image_id}.tif",
                "prediction_directory": image_id,
                "outputs": {
                    "fiber_table": f"{image_id}_fibers.csv",
                    "fiber_labels": f"{image_id}_labels.tif",
                },
                "applicable_domains": ["fiber_segmentation", "fiber_typing"],
            }
        )
    manifest = {
        "schema_version": "review_project.v1",
        "project_id": "final",
        "project_name": "Finalization",
        "panel_manifest": "panel.yaml",
        "model_version": "model.v1",
        "images": images,
    }
    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return load_project(path, validate_paths=False)


def _session(project) -> ReviewSession:
    session = ReviewSession(project_id=project.project_id, model_version=project.model_version)
    for image in project.images:
        session.record_input_fingerprints(image.image_id, image_input_fingerprints(image))
    return session


def _decide(session, fiber_id, status, reviewed="", image_id="one"):
    session.record_fiber_type_decision(
        FiberTypeDecision(
            image_id=image_id,
            fiber_id=fiber_id,
            model_fiber_type=MODEL_CALLS[fiber_id],
            reviewed_fiber_type=reviewed,
            review_status=status,
            reviewer="reviewer_a",
        )
    )


def _box(x0, y0, x1, y1) -> dict:
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


def _region(region_id, geometry, action, domain="fiber_typing", **extra):
    return RegionAnnotation(
        region_id=region_id,
        image_id="one",
        geometry=geometry,
        domain=domain,
        action=action,
        reason_code="test",
        **extra,
    )


def _by_id(result) -> pd.DataFrame:
    return result.table.set_index("label")


def test_unreviewed_fibers_keep_predictions_and_inputs_are_untouched(tmp_path):
    project = _project(tmp_path)
    fibers_path = project.images[0].outputs["fiber_table"]
    before = hashlib.sha256(fibers_path.read_bytes()).hexdigest()

    manifest = finalize_project(project, _session(project), tmp_path / "final")

    table = pd.read_csv(tmp_path / "final" / "one_fibers_finalized.csv")
    assert hashlib.sha256(fibers_path.read_bytes()).hexdigest() == before
    assert table["final_type"].tolist() == list(MODEL_CALLS.values())
    assert set(table["value_source"]) == {"predicted"}
    # Every model column survives next to the final value.
    assert table["prob_iia"].tolist() == [0.9, 0.1, 0.1, 0.2]
    assert table["fiber_type"].tolist() == list(MODEL_CALLS.values())
    assert table["flagged_unreviewed"].tolist() == [False, False, True, True]
    image = manifest["images"][0]
    assert image["verification"] == "verified"
    assert image["counts"]["predicted"] == 4 and image["counts"]["flagged_unreviewed"] == 2
    assert manifest["policies"]["region_membership"] == "centroid_inside"


def test_review_decisions_map_to_final_values(tmp_path):
    project = _project(tmp_path)
    session = _session(project)
    _decide(session, 1, "corrected", reviewed="iib")
    _decide(session, 2, "accepted", reviewed="iib")
    _decide(session, 3, "uncertain")
    _decide(session, 4, "excluded")

    table = _by_id(finalize_image(project, session, project.images[0]))

    assert table.loc[1, ["final_type", "value_source"]].tolist() == ["iib", "reviewed"]
    assert table.loc[1, "fiber_type"] == "iia"  # model call preserved
    assert table.loc[2, ["final_type", "value_source"]].tolist() == ["iib", "reviewed"]
    assert table.loc[3, ["final_type", "value_source"]].tolist() == ["", "unresolved"]
    assert table.loc[4, ["final_type", "value_source", "exclusion_reason"]].tolist() == [
        "",
        "excluded",
        "fiber_review",
    ]
    assert not table.loc[3, "flagged_unreviewed"]
    assert table.loc[1, "reviewer"] == "reviewer_a"


def test_region_exclusion_takes_precedence_and_respects_domain(tmp_path):
    project = _project(tmp_path)
    session = _session(project)
    _decide(session, 1, "corrected", reviewed="iib")
    session.add_region(_region("fold", _box(0, 0, 10, 10), "exclude_all_analysis"))
    session.add_region(_region("dim", _box(10, 0, 20, 10), "ignore_nuclei", domain="nuclei"))
    session.add_region(_region("unsure", _box(0, 10, 10, 20), "unresolved"))

    table = _by_id(finalize_image(project, session, project.images[0]))

    assert table.loc[1, "value_source"] == "excluded"
    assert table.loc[1, "exclusion_reason"] == "region:exclude_all_analysis:fold"
    assert table.loc[2, "value_source"] == "predicted"  # nuclei-only region has no fiber effect
    assert table.loc[3, "value_source"] == "unresolved"
    assert table.loc[4, "value_source"] == "predicted"


def test_analysis_rois_tag_fibers_and_exclude_outside(tmp_path):
    project = _project(tmp_path)
    session = _session(project)
    session.add_region(
        _region(
            "left", _box(0, 0, 9, 20), "analysis_roi", kind="analysis_roi", name="left", role="half"
        )
    )
    session.add_region(
        _region(
            "edge",
            _box(13.5, 0, 20, 9),
            "analysis_roi",
            kind="analysis_roi",
            name="edge",
            role="strip",
        )
    )

    result = finalize_image(project, session, project.images[0])
    table = _by_id(result)

    assert table.loc[1, ["roi_name", "roi_status"]].tolist() == ["left", "assigned"]
    assert table.loc[3, ["roi_name", "roi_status"]].tolist() == ["left", "assigned"]
    assert table.loc[2, "roi_status"] == "boundary"  # centroid x = 13.5 lies on the strip edge
    assert table.loc[4, ["roi_status", "value_source"]].tolist() == ["outside", "excluded"]
    assert any("boundary" in warning for warning in result.warnings)


def test_excluded_image_and_unselected_sections_are_excluded(tmp_path):
    project = _project(tmp_path, image_ids=("one", "two", "three"))
    session = _session(project)
    session.set_status("one", "fiber_typing", "excluded")
    selection = tmp_path / "section_selection.csv"
    pd.DataFrame(
        [{"mouse_id": "mouse_a", "domain": "fiber_typing", "selected_image_ids": "one|two"}]
    ).to_csv(selection, index=False)

    manifest = finalize_project(
        project, session, tmp_path / "final", section_selection_path=selection
    )

    combined = pd.read_csv(tmp_path / "final" / "final_fiber_table.csv", keep_default_na=False)
    reasons = combined.groupby("image_id")["exclusion_reason"].first().to_dict()
    assert reasons == {
        "one": "image:fiber_typing:excluded",
        "two": "",
        "three": "section_not_selected",
    }
    assert manifest["section_selection_applied"] is True


def test_changed_inputs_and_changed_predictions_are_refused(tmp_path):
    project = _project(tmp_path)
    session = _session(project)
    _decide(session, 1, "corrected", reviewed="iib")
    fibers_path = project.images[0].outputs["fiber_table"]
    table = pd.read_csv(fibers_path)
    table.loc[0, "fiber_type"] = "iib"
    table.to_csv(fibers_path, index=False)

    with pytest.raises(FinalizationError, match="fiber_table changed since review"):
        finalize_image(project, session, project.images[0])

    session.input_fingerprints.clear()  # an older session without digests
    with pytest.raises(FinalizationError, match="no longer match the fiber table"):
        finalize_image(project, session, project.images[0])
    with pytest.raises(FinalizationError, match="no input fingerprints"):
        finalize_image(project, session, project.images[0], require_verified=True)


def test_unverified_sessions_finalize_with_warning(tmp_path):
    project = _project(tmp_path)
    session = ReviewSession(project_id=project.project_id, model_version=project.model_version)

    result = finalize_image(project, session, project.images[0])

    assert result.verification == "unverified"
    assert result.warnings == ["review session has no input fingerprints for this image"]


def test_stale_fiber_features_and_model_version_mismatch_are_refused(tmp_path):
    project = _project(tmp_path)
    session = _session(project)
    session.mark_stale("one", "fiber_mask")
    with pytest.raises(FinalizationError, match="stale after review edits"):
        finalize_image(project, session, project.images[0])

    other = ReviewSession(project_id=project.project_id, model_version="model.v0")
    with pytest.raises(FinalizationError, match="used model 'model.v0'"):
        finalize_project(project, other, tmp_path / "final")


def test_record_input_fingerprints_reports_changes():
    session = ReviewSession(project_id="p", model_version="m")

    assert session.record_input_fingerprints("one", {"fiber_table": "a"}) == []
    assert session.record_input_fingerprints("one", {"fiber_table": "a"}) == []
    assert session.record_input_fingerprints("one", {"fiber_table": "b"}) == ["fiber_table"]
    restored = ReviewSession.from_dict(session.to_dict())
    assert restored.input_fingerprints == {"one": {"fiber_table": "a"}}


def test_finalization_is_deterministic_and_cli_refuses_with_exit_code(tmp_path):
    project = _project(tmp_path)
    session = _session(project)
    _decide(session, 3, "uncertain")
    save_session(project.review_state_path, session)

    assert (
        finalize_main(
            ["--project", str(project.manifest_path), "--output-dir", str(tmp_path / "a")]
        )
        == 0
    )
    assert (
        finalize_main(
            ["--project", str(project.manifest_path), "--output-dir", str(tmp_path / "b")]
        )
        == 0
    )
    for name in ("one_fibers_finalized.csv", "final_fiber_table.csv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    manifest = json.loads((tmp_path / "a" / "finalization_manifest.json").read_text())
    assert manifest["images"][0]["counts"]["unresolved"] == 1

    tifffile.imwrite(project.images[0].outputs["fiber_labels"], np.zeros((20, 20), np.uint16))
    assert (
        finalize_main(
            ["--project", str(project.manifest_path), "--output-dir", str(tmp_path / "c")]
        )
        == 2
    )


def test_legacy_review_csv_without_model_column_finalizes(tmp_path):
    project = _project(tmp_path)
    session = _session(project)
    legacy = tmp_path / "one_fibers_manual_review.csv"
    pd.DataFrame(
        {
            "fiber_id": [1, 2, 3],
            "corrected_type": ["iib", "", ""],
            "is_uncertain": [False, True, False],
            "is_excluded": [False, False, True],
        }
    ).to_csv(legacy, index=False)

    assert import_legacy_reviews(project, session, {"one": legacy}) == 3
    table = _by_id(finalize_image(project, session, project.images[0]))

    assert table.loc[1, ["final_type", "value_source"]].tolist() == ["iib", "reviewed"]
    assert table.loc[2, "value_source"] == "unresolved"
    assert table.loc[3, "value_source"] == "excluded"
    assert table.loc[4, "value_source"] == "predicted"


def test_legacy_review_conflicting_with_session_is_refused(tmp_path):
    project = _project(tmp_path)
    session = _session(project)
    _decide(session, 1, "accepted", reviewed="iia")
    legacy = tmp_path / "legacy.csv"
    pd.DataFrame({"fiber_id": [1], "corrected_type": ["iib"]}).to_csv(legacy, index=False)

    with pytest.raises(FinalizationError, match="both the review session and the legacy CSV"):
        import_legacy_reviews(project, session, {"one": legacy})


def test_rule_path_internal_labels_match_normalized_decisions(tmp_path):
    project = _project(tmp_path)
    fibers_path = project.images[0].outputs["fiber_table"]
    table = pd.read_csv(fibers_path)
    table.loc[0, "fiber_type"] = "type2"  # legacy internal name for IIa
    table.to_csv(fibers_path, index=False)
    session = _session(project)
    _decide(session, 1, "corrected", reviewed="iib")  # decision stores the biological "iia"

    finalized = _by_id(finalize_image(project, session, project.images[0]))

    assert finalized.loc[1, ["final_type", "value_source"]].tolist() == ["iib", "reviewed"]


def test_project_without_review_session_finalizes_as_predicted(tmp_path, capsys):
    project = _project(tmp_path)

    assert (
        finalize_main(
            ["--project", str(project.manifest_path), "--output-dir", str(tmp_path / "f")]
        )
        == 0
    )

    table = pd.read_csv(tmp_path / "f" / "final_fiber_table.csv")
    assert set(table["value_source"]) == {"predicted"}
    assert "no review session found" in capsys.readouterr().err
