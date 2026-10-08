"""Promotion: local model manifests, carrying review forward, and refusing silent switches."""

from __future__ import annotations

import json

import pytest
import yaml

from fibertypeqc.artifacts import file_sha256
from fibertypeqc.model_resolution import resolve_model_argument
from src.promote_model import (
    PromotionError,
    carry_review_forward,
    find_candidate,
    improver_recommended,
)
from src.review.schemas import FiberTypeDecision, ObjectReviewStatus
from src.review.session import ReviewSession


def _candidate(root, name="logistic", recommended=None):
    folder = root / "models" / "improve_20260101T000000Z" / "candidates" / name
    folder.mkdir(parents=True)
    artifact = folder / f"{name}.joblib"
    artifact.write_bytes(b"not a real model")
    manifest = {"model_id": f"base__run__{name}", "artifact_sha256": file_sha256(artifact)}
    (folder / f"{name}.yaml").write_text(yaml.safe_dump(manifest))
    (root / "models" / "improve_20260101T000000Z" / "improver_summary.json").write_text(
        json.dumps({"decision": {"recommended": recommended}})
    )
    return artifact


def test_local_model_needs_a_matching_manifest_beside_it(tmp_path):
    artifact = _candidate(tmp_path)

    resolved = resolve_model_argument(str(artifact))
    assert resolved.model_id == "base__run__logistic"
    assert resolved.manifest_path == artifact.with_suffix(".yaml").resolve()

    artifact.write_bytes(b"changed after the manifest was written")
    with pytest.raises(ValueError, match="does not match the manifest beside it"):
        resolve_model_argument(str(artifact))
    artifact.with_suffix(".yaml").unlink()
    with pytest.raises(ValueError, match="is not a registered model"):
        resolve_model_argument(str(artifact))


def test_candidates_are_found_by_name_and_recommendation_is_read(tmp_path):
    artifact = _candidate(tmp_path, recommended="logistic")

    assert find_candidate(tmp_path, "logistic") == artifact.resolve()
    assert find_candidate(tmp_path, str(artifact)) == artifact.resolve()
    assert improver_recommended(artifact) is True
    with pytest.raises(PromotionError, match="No candidate named"):
        find_candidate(tmp_path, "random_forest")
    other = _candidate(tmp_path / "second", recommended=None)
    assert improver_recommended(other) is False


def test_review_is_carried_forward_without_changing_reviewer_labels():
    session = ReviewSession(project_id="p", model_version="old")
    cases = {
        1: ("iib", "iib", ObjectReviewStatus.ACCEPTED),  # kept old call iib
        2: ("iib", "iix", ObjectReviewStatus.CORRECTED),  # corrected to iix
        3: ("iix", "exclude", ObjectReviewStatus.EXCLUDED),
        4: ("iia", "iia", ObjectReviewStatus.ACCEPTED),
    }
    for fiber_id, (model, reviewed, status) in cases.items():
        session.record_fiber_type_decision(
            FiberTypeDecision(
                image_id="a",
                fiber_id=fiber_id,
                model_fiber_type=model,
                reviewed_fiber_type=reviewed,
                review_status=status,
            )
        )
    new_calls = {"a": {1: "iix", 2: "iix", 3: "iib", 4: "iia"}}

    counts = carry_review_forward(session, new_calls, "new")

    by_id = {d.fiber_id: d for d in session.object_decisions}
    assert counts == {"reviewed_fibers": 3, "agree_before": 2, "agree_after": 2}
    # Fiber 1: reviewer said iib; the new model says iix, so it is now a correction to iib.
    assert (by_id[1].reviewed_fiber_type, by_id[1].review_status) == (
        "iib",
        ObjectReviewStatus.CORRECTED,
    )
    # Fiber 2: reviewer said iix; the new model agrees, so it is now a kept call.
    assert (by_id[2].reviewed_fiber_type, by_id[2].review_status) == (
        "iix",
        ObjectReviewStatus.ACCEPTED,
    )
    assert by_id[3].review_status is ObjectReviewStatus.EXCLUDED
    assert by_id[3].model_fiber_type == "iib"
    assert by_id[4].review_status is ObjectReviewStatus.ACCEPTED
    assert session.model_version == "new"
