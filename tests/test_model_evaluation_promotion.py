"""Model builder E4: independent test evaluation, then promotion from no model and rollback."""

from __future__ import annotations

import hashlib
import json
import shutil

import pandas as pd
import pytest
import yaml

from fibertypeqc.cohort_report import build_cohort_report
from src import cli
from src.finalize_review_project import main as finalize_main
from src.model_evaluation import EvaluationError, evaluate, evaluation_status, find_build_candidate
from src.promote_model import PromotionError, _test_check, promote, rollback
from src.reference_snapshot import check as check_snapshot
from src.reference_snapshot import freeze as freeze_snapshot
from src.review.project import load_project
from src.review.schemas import FiberTypeDecision, ObjectReviewStatus
from src.review.session import ReviewSession
from src.review.storage import load_session, save_session
from src.summarize_results import write_results
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


def test_evaluate_promote_and_rollback_restore_the_project_exactly(
    built, monkeypatch, tmp_path_factory
):
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
    # Without 'classes' in the config, the review project keeps the classes and the panel's
    # name for 'negative', and 'prepare' keeps that review project rather than rebuilding it.
    assert promoted.class_labels == {"negative": "I⁻/IIa⁻"}
    steps = []
    monkeypatch.setattr(cli, "_run", lambda step, arguments: steps.append(step) or 0)
    assert cli.command_prepare(cli.load_project_folder(root)) == 0
    assert steps == ["generate_review_qc"]  # the review project is kept, not rebuilt

    # The promoted project runs through to results; outputs go outside the project folder.
    elsewhere = tmp_path_factory.mktemp("after_promotion")
    snapshot = freeze_snapshot(root / "batch", elsewhere / "snapshot.json")
    assert snapshot["sections"] and all(
        check.passed and check.level == "exact"
        for check in check_snapshot(root / "batch", elsewhere / "snapshot.json")
    )
    review = root / "review" / "project.yaml"
    assert finalize_main(["--project", str(review), "--output-dir", str(elsewhere / "final")]) == 0
    manifest = write_results(elsewhere / "final", elsewhere / "results", review)
    assert "negative" in manifest["fiber_classes"]
    assert manifest["class_labels"] == {"negative": "I⁻/IIa⁻"}
    assert "I⁻/IIa⁻ final" in build_cohort_report(elsewhere / "results")

    # A guided-review decision made with the promoted model.
    session = ReviewSession(project_id=promoted.project_id, model_version=row["to_model"])
    session.record_fiber_type_decision(
        FiberTypeDecision(
            image_id="syn_m1", fiber_id=1, model_fiber_type=str(fibers.loc[0, "fiber_type"]),
            reviewed_fiber_type="iia", review_status=ObjectReviewStatus.CORRECTED,
        )
    )  # fmt: skip
    save_session(promoted.review_state_path, session)

    undone = rollback(root)
    assert _tree(root) == before  # byte-for-byte, including config and run records
    assert load_project(root / "review" / "project.yaml").model_version == "none"
    # The outgoing state, with that decision, is archived and not applied to the restored project.
    outgoing = root / undone["outgoing_archive"]
    kept = load_session(outgoing / promoted.review_state_path.relative_to(root))
    assert kept.model_version == row["to_model"]
    assert [(d.image_id, d.fiber_id, d.reviewed_fiber_type) for d in kept.object_decisions] == [
        ("syn_m1", 1, "iia")
    ]
    archived_config = yaml.safe_load((outgoing / "fibertypeqc_project.yaml").read_text())
    assert archived_config["model"].endswith(f"{chosen}.joblib")
    history = pd.read_csv(root / "models" / "model_history.csv", keep_default_na=False)
    assert history["outgoing_archive"].tolist() == ["", undone["outgoing_archive"]]


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


def _skip_retyping(monkeypatch):
    """Promotion bookkeeping without re-typing every section (covered by the end-to-end test)."""
    from src import promote_model

    monkeypatch.setattr(promote_model, "_retype", lambda *args, **kwargs: None)


def test_an_override_promotion_is_recorded_and_reported(built, monkeypatch, capsys):
    from src.promote_model import main

    root, project, chosen = built
    evaluate(root, project, chosen)
    _label(project, "late", {(f"syn_{TEST_MOUSE}", 1): CLASSES[0]})
    _skip_retyping(monkeypatch)
    arguments = ["--project-root", str(root), "--panel-config", str(root / "panel.yaml")]
    assert main([*arguments, "--candidate", chosen]) == 1  # stale: refused
    assert "changed since" in capsys.readouterr().err
    assert main([*arguments, "--candidate", chosen, "--override"]) == 0
    assert "out of date; recorded in models/independent_test_checks.csv" in capsys.readouterr().out
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


def test_without_test_mice_promotion_warns_and_is_recorded(built, monkeypatch, capsys):
    from src.promote_model import main

    root, _, chosen = built
    (root / "evaluation_roles.csv").unlink()
    _skip_retyping(monkeypatch)
    arguments = ["--project-root", str(root), "--panel-config", str(root / "panel.yaml")]
    assert main([*arguments, "--candidate", chosen]) == 0  # no override needed
    assert "WARNING: no test mice are declared" in capsys.readouterr().out
    checks = pd.read_csv(root / "models" / "independent_test_checks.csv")
    assert checks.loc[0, "status"] == "no_test_mice" and not bool(checks.loc[0, "override"])


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


def test_measurement_check_is_exact_on_model_features_only(tmp_path):
    from src.promote_model import _same_measurements

    table = pd.DataFrame({"label": [1, 2], "type_i.mean": [0.25, 0.5], "area": [10.0, 20.0]})
    old, new = tmp_path / "old.csv", tmp_path / "new.csv"
    table.to_csv(old, index=False)
    table.assign(area=[11.0, 21.0]).to_csv(new, index=False)
    assert _same_measurements(old, new, ["type_i.mean"])  # area is not a model feature
    table.assign(**{"type_i.mean": [0.25, 0.5 + 1e-9]}).to_csv(new, index=False)
    assert not _same_measurements(old, new, ["type_i.mean"])  # no tolerance
    table.drop(columns="type_i.mean").to_csv(new, index=False)
    assert not _same_measurements(old, new, ["type_i.mean"])
    table.iloc[:1].to_csv(new, index=False)
    assert not _same_measurements(old, new, ["type_i.mean"])


def test_evaluate_command_reports_panel_names_and_repeated_use(built, capsys):
    from src.evaluate_model import main

    root, _, chosen = built
    assert main(["--project-root", str(root), "--candidate", chosen]) == 0
    printed = capsys.readouterr().out
    assert "recall I⁻/IIa⁻:" in printed and "used for" not in printed
    assert main(["--project-root", str(root), "--candidate", chosen]) == 0
    assert "used for 2 evaluations" in capsys.readouterr().out
    assert main(["--project-root", str(root), "--candidate", "nonexistent"]) == 1


def test_rollback_changes_nothing_if_the_outgoing_state_cannot_be_archived(built, monkeypatch):
    from src import promote_model

    root, _, chosen = built
    (root / "evaluation_roles.csv").unlink()
    _skip_retyping(monkeypatch)
    promote(root, chosen, root / "panel.yaml")
    history_path = root / "models" / "model_history.csv"
    # A history written before 'outgoing_archive' existed.
    pd.read_csv(history_path, dtype=str, keep_default_na=False).drop(
        columns="outgoing_archive"
    ).to_csv(history_path, index=False)
    archives = sorted((root / "models" / "history").iterdir())
    before, history_before = _tree(root), history_path.read_bytes()

    real_archive = promote_model._archive

    def failing_archive(root_, project_, name):
        (root_ / "models" / "history" / name / "partial").mkdir(parents=True)
        raise OSError("disk full")

    monkeypatch.setattr(promote_model, "_archive", failing_archive)
    with pytest.raises(PromotionError, match="nothing was changed"):
        rollback(root)
    assert _tree(root) == before
    assert history_path.read_bytes() == history_before
    assert sorted((root / "models" / "history").iterdir()) == archives  # no partial archive

    monkeypatch.setattr(promote_model, "_archive", real_archive)
    row = rollback(root)
    history = pd.read_csv(history_path, keep_default_na=False)
    assert history["action"].tolist() == ["promote", "rollback"]
    assert history["outgoing_archive"].tolist() == ["", row["outgoing_archive"]]


def test_status_follows_build_and_independent_evaluation(built):
    root, project, chosen = built
    folder = cli.load_project_folder(root)
    status = cli.builder_status(folder)
    assert status["recommended"] == chosen and status["evaluation"] == "missing"
    evaluate(root, project, chosen)
    assert cli.builder_status(folder)["evaluation"] == "current"
    assert cli.builder_status(folder)["test_set_uses"] == 1
    _label(project, "late", {(f"syn_{TEST_MOUSE}", 1): CLASSES[0]})
    assert cli.builder_status(folder)["evaluation"] == "stale"
    (root / "evaluation_roles.csv").unlink()
    assert cli.builder_status(folder)["evaluation"] == "no_test_mice"
