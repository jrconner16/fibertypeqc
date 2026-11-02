"""Model builder E4: independent test evaluation, then promotion from no model and rollback."""

from __future__ import annotations

import hashlib
import json
import shutil

import pandas as pd
import pytest
import yaml

from src.model_evaluation import EvaluationError, evaluate, evaluation_status, find_build_candidate
from src.promote_model import PromotionError, _test_check, promote, rollback
from src.review.project import load_project
from tests.test_model_builder import (  # noqa: F401  (pytest fixtures)
    CLASSES,
    TEST_MOUSE,
    _build,
    _label,
    built_project,
    labelled,
)

CONFIG = {
    "schema_version": "fibertypeqc_project.v1",
    "images": "images",
    "panel": "panel.yaml",
    "sample_sheet": "samples.csv",
    "model": "none",
    "classes": CLASSES,
}


@pytest.fixture()
def built(labelled):  # noqa: F811
    """A labelled no-model project with a config file and one build run."""
    root, project = labelled
    (root / "fibertypeqc_project.yaml").write_text(yaml.safe_dump(CONFIG, sort_keys=False))
    result = _build(project, root, name="build_20260101T000000Z")
    return root, project, result["selection"]["decision"]["recommended"]


def _tree(root):
    """Digest of every project file outside models/ (the model log and archives)."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).parts[0] != "models"
    }


def _blind_labels(root):
    return {
        name: digest for name, digest in _tree(root).items() if name.startswith("review/reference/")
    }


def test_evaluate_promote_and_rollback_restore_the_project_exactly(built):
    root, project, chosen = built
    before = _tree(root)
    labels_before = _blind_labels(root)

    result = evaluate(root, project, chosen)
    assert result["test_mice"] == [TEST_MOUSE]
    assert result["labels"]["n_fibers"] == 144  # test-mouse labels only
    assert result["labels"]["left_out"]["training_mice"] > 0  # training labels not read
    assert result["test_set_uses"] == 1
    assert set(result["pooled_recall_by_class"]) == set(CLASSES)
    history = pd.read_csv(root / "models" / "evaluation_history.csv")
    assert len(history) == 1 and history.loc[0, "model_id"].endswith(chosen)
    assert _tree(root) == before  # evaluating changes nothing in the project

    row = promote(root, chosen, root / "panel.yaml")
    assert row["independent_test"]["status"] == "current"
    assert row["independent_test"]["override"] is False
    config = yaml.safe_load((root / "fibertypeqc_project.yaml").read_text())
    assert "classes" not in config and config["model"].endswith(f"{chosen}.joblib")
    promoted = load_project(root / "review" / "project.yaml")
    assert promoted.model_version == row["to_model"]
    assert promoted.review_classes == tuple(CLASSES)  # the configured buttons stay
    fibers = pd.read_csv(root / "batch/syn_m1/syn_m1_fibers.csv")
    assert set(fibers["fiber_type"]) <= set(CLASSES)
    assert _blind_labels(root) == labels_before  # blind labels are never touched

    rollback(root)
    assert _tree(root) == before  # byte-for-byte, including config and run records
    assert load_project(root / "review" / "project.yaml").model_version == "none"


def test_promotion_needs_a_current_evaluation_or_a_recorded_override(built):
    root, project, chosen = built
    before = _tree(root)
    with pytest.raises(PromotionError, match="has not been evaluated"):
        promote(root, chosen, root / "panel.yaml")
    assert _tree(root) == before

    evaluate(root, project, chosen)
    # A new blind label on a test mouse makes the evaluation stale.
    _label(project, "late", {(f"syn_{TEST_MOUSE}", 1): CLASSES[0]})
    assert evaluation_status(root, project, find_build_candidate(root, chosen))["status"] == "stale"
    with pytest.raises(PromotionError, match="changed since"):
        promote(root, chosen, root / "panel.yaml")
    artifact = find_build_candidate(root, chosen)
    assert _test_check(root, project, artifact, override=True) == {
        "status": "stale",
        "evaluation": _test_check(root, project, artifact, override=True)["evaluation"],
        "override": True,
    }


@pytest.mark.integration
def test_an_override_promotion_is_recorded(built):
    root, project, chosen = built
    evaluate(root, project, chosen)
    _label(project, "late", {(f"syn_{TEST_MOUSE}", 1): CLASSES[0]})
    row = promote(root, chosen, root / "panel.yaml", override=True)
    assert row["independent_test"] == {
        **row["independent_test"],
        "status": "stale",
        "override": True,
    }
    checks = pd.read_csv(root / "models" / "independent_test_checks.csv")
    assert checks.loc[0, "status"] == "stale" and bool(checks.loc[0, "override"])


def test_changed_test_measurements_make_the_evaluation_stale(built):
    root, project, chosen = built
    evaluate(root, project, chosen)
    path = root / f"batch/syn_{TEST_MOUSE}/syn_{TEST_MOUSE}_feature_diagnostics.csv"
    path.write_text(path.read_text() + "\n")
    assert evaluation_status(root, project, find_build_candidate(root, chosen))["status"] == "stale"


def test_repeated_evaluations_are_counted(built):
    root, project, chosen = built
    evaluate(root, project, chosen)
    other = next(name for name in ("logistic", "random_forest") if name != chosen)
    result = evaluate(root, project, other)
    assert result["test_set_uses"] == 2
    assert len(pd.read_csv(root / "models" / "evaluation_history.csv")) == 2


def test_without_test_mice_evaluation_is_refused_and_promotion_only_warns(built):
    root, project, chosen = built
    (root / "evaluation_roles.csv").unlink()
    with pytest.raises(EvaluationError, match="no test mice"):
        evaluate(root, project, chosen)
    artifact = find_build_candidate(root, chosen)
    check = _test_check(root, project, artifact, override=False)
    assert check["status"] == "no_test_mice" and check["override"] is False


@pytest.mark.integration
def test_without_test_mice_promotion_is_recorded(built):
    root, project, chosen = built
    (root / "evaluation_roles.csv").unlink()
    row = promote(root, chosen, root / "panel.yaml")
    assert row["independent_test"]["status"] == "no_test_mice"
    checks = pd.read_csv(root / "models" / "independent_test_checks.csv")
    assert checks.loc[0, "status"] == "no_test_mice"


@pytest.mark.integration
def test_measurements_that_do_not_reproduce_stop_promotion_and_change_nothing(
    built, monkeypatch
):
    from src import promote_model

    root, project, chosen = built
    evaluate(root, project, chosen)
    before = _tree(root)
    monkeypatch.setattr(promote_model, "_same_measurements", lambda *args: False)
    with pytest.raises(PromotionError, match="did not reproduce the measurements"):
        promote(root, chosen, root / "panel.yaml")
    assert _tree(root) == before


def test_unknown_candidate_and_typed_project_are_refused(built):
    root, project, _ = built
    with pytest.raises(EvaluationError, match="No built candidate"):
        evaluate(root, project, "nonexistent")
    from dataclasses import replace

    with pytest.raises(EvaluationError, match="model: none"):
        evaluate(root, replace(project, model_version="some_model"), "logistic")


def test_evaluation_folder_keeps_predictions_metrics_and_inputs(built):
    root, project, chosen = built
    result = evaluate(root, project, chosen)
    folder = root / "models" / "evaluations"
    (run,) = list(folder.iterdir())
    for name in ("test_predictions.csv", "metrics_by_mouse.csv", "confusion_matrix.csv"):
        assert (run / name).is_file()
    saved = json.loads((run / "evaluation.json").read_text())
    assert set(saved["inputs"]) == {
        "artifact_sha256",
        "manifest_sha256",
        "test_labels_sha256",
        "test_sections",
    }
    assert set(saved["inputs"]["test_sections"][f"syn_{TEST_MOUSE}"]) == {
        "fiber_labels",
        "measurements",
    }
    assert result["recommended_by_builder"] is True
    shutil.rmtree(folder)
