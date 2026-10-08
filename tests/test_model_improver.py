"""Model improver: labels from review, held-out-mouse evaluation, and the promotion rule."""

from __future__ import annotations

import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from fibertypeqc.model_manifest import load_model_manifest
from src.model_improver import ImproverError, improve, load_recipe, review_labels
from src.review.schemas import FiberTypeDecision, ObjectReviewStatus
from src.review.session import ReviewSession

CLASSES = ["iia", "iib", "iix"]
CENTERS = {"iia": (4.0, 0.0), "iib": (0.0, 4.0), "iix": (0.0, 0.0)}
BASE_MANIFEST = {
    "manifest_version": 1,
    "model_id": "base_model",
    "task": "fiber_identity",
    "feature_schema_version": "multiplanel_features.v1",
    "required_markers": ["laminin", "type_iia", "type_iib"],
    "outputs": CLASSES,
    "features": ["type_iia.mean", "type_iib.mean"],
    "feature_extraction": yaml.safe_load(
        open("manifests/models/ta_three_class_logistic_v1.yaml", encoding="utf-8")
    )["feature_extraction"],
}


def _fixture(mice=4, per_class=12, wrong_every=3):
    """Separable synthetic fibers; the 'current model' miscalls every ``wrong_every``-th fiber."""
    rng = np.random.default_rng(0)
    session = ReviewSession(project_id="p", model_version="base_model", reviewer="r")
    features, mouse_of = {}, {}
    for mouse in range(mice):
        image_id = f"image_{mouse}"
        mouse_of[image_id] = f"mouse_{mouse}"
        rows, fiber_id = [], 0
        for truth in CLASSES:
            for _ in range(per_class):
                fiber_id += 1
                x, y = rng.normal(CENTERS[truth], 0.4)
                rows.append({"label": fiber_id, "type_iia.mean": x, "type_iib.mean": y})
                wrong = fiber_id % wrong_every == 0
                model_call = CLASSES[(CLASSES.index(truth) + 1) % 3] if wrong else truth
                session.record_fiber_type_decision(
                    FiberTypeDecision(
                        image_id=image_id,
                        fiber_id=fiber_id,
                        model_fiber_type=model_call,
                        reviewed_fiber_type=truth,
                        review_status=ObjectReviewStatus.CORRECTED
                        if wrong
                        else ObjectReviewStatus.ACCEPTED,
                        queue_source="flagged",
                    )
                )
        features[image_id] = pd.DataFrame(rows)
    project = SimpleNamespace(
        project_id="p", image=lambda image_id: SimpleNamespace(mouse_id=mouse_of[image_id])
    )
    return project, session, features


def test_improver_recommends_a_candidate_that_beats_the_current_model(tmp_path):
    project, session, features = _fixture()

    result = improve(project, session, features, BASE_MANIFEST, tmp_path / "improve_1")

    assert result["n_labels"] == 144 and result["n_mice"] == 4
    assert result["labels_by_source"] == {"kept_model_call": 96, "corrected": 48}
    assert result["metrics"]["current_model"]["accuracy"] == pytest.approx(2 / 3)
    assert result["metrics"]["logistic"]["median_mouse_macro_f1"] > 0.95
    decision = result["decision"]
    assert decision["recommended"] in {"logistic", "random_forest", "gradient_boosting"}
    assert not decision["keep_current_model"]
    assert all(check["passes"] for check in decision["checks"].values())
    # Every candidate is saved with a loadable manifest that points back to its parent.
    for name, entry in result["candidates"].items():
        manifest_path = tmp_path / "improve_1" / entry["manifest"]
        manifest = load_model_manifest(manifest_path)
        raw = yaml.safe_load(manifest_path.read_text())
        model = joblib.load(tmp_path / "improve_1" / entry["artifact"])
        assert raw["parent_model_id"] == "base_model" and name in manifest.model_id
        assert manifest.artifact_sha256 == entry["artifact_sha256"]
        assert sorted(map(str, model.classes_)) == CLASSES
        assert set(model.predict(features["image_0"][BASE_MANIFEST["features"]])) <= set(CLASSES)
    saved = json.loads((tmp_path / "improve_1" / "improver_summary.json").read_text())
    assert saved["decision"] == decision
    with pytest.raises(FileExistsError):  # a run folder is never overwritten
        improve(project, session, features, BASE_MANIFEST, tmp_path / "improve_1")


def test_current_model_is_kept_when_it_is_already_right(tmp_path):
    project, session, features = _fixture(wrong_every=10_000)

    result = improve(project, session, features, BASE_MANIFEST, tmp_path / "run")

    assert result["decision"]["keep_current_model"]
    assert result["decision"]["recommended"] is None


def test_too_few_mice_never_yields_a_recommendation(tmp_path):
    project, session, features = _fixture(mice=2)

    result = improve(project, session, features, BASE_MANIFEST, tmp_path / "run")

    assert result["decision"]["keep_current_model"]
    assert "Only 2 mice" in result["decision"]["reason"]
    with pytest.raises(ImproverError, match="one mouse"):
        improve(*_fixture(mice=1), BASE_MANIFEST, tmp_path / "single")


def test_each_class_needs_enough_labels(tmp_path):
    project, session, features = _fixture(per_class=2)

    with pytest.raises(ImproverError, match="at least 10 reviewed fibers"):
        improve(project, session, features, BASE_MANIFEST, tmp_path / "run")


def test_review_labels_skip_exclusions_and_use_kept_calls():
    session = ReviewSession(project_id="p", model_version="m")
    for fiber_id, (model, reviewed, status) in enumerate(
        [
            ("iib", "iib", ObjectReviewStatus.ACCEPTED),
            ("iib", "iix", ObjectReviewStatus.CORRECTED),
            ("iib", "exclude", ObjectReviewStatus.EXCLUDED),
            ("iix", "", ObjectReviewStatus.UNRESOLVED),
            ("iix", "i", ObjectReviewStatus.CORRECTED),
        ],
        start=1,
    ):
        session.record_fiber_type_decision(
            FiberTypeDecision(
                image_id="a",
                fiber_id=fiber_id,
                model_fiber_type=model,
                reviewed_fiber_type=reviewed,
                review_status=status,
            )
        )

    labels, skipped = review_labels(session, CLASSES)

    assert labels[["fiber_id", "label", "label_source"]].values.tolist() == [
        [1, "iib", "kept_model_call"],
        [2, "iix", "corrected"],
    ]
    assert skipped == {"excluded": 1, "uncertain_or_unresolved": 1, "outside_model_classes": 1}


def test_default_recipe_declares_candidates_and_rule():
    recipe = load_recipe()

    assert set(recipe["candidates"]) == {"logistic", "random_forest", "gradient_boosting"}
    assert recipe["promotion_rule"]["minimum_mice"] == 3
