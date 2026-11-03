"""Model builder (E3): blind labels -> evaluated, refitted candidates for a panel with no model."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml

from fibertypeqc.model_manifest import FEATURE_EXTRACTION_FIELDS, load_model_manifest
from fibertypeqc.semantic_model import predict_semantic_candidate
from src.make_review_project import build_project
from src.model_builder import (
    BASELINE,
    BuilderError,
    blind_label_sessions,
    build,
    check_minimums,
    combine_labels,
    decide,
    evaluate,
    load_recipe,
    mouse_metrics,
    training_features,
)
from src.review.finalization import image_input_fingerprints
from src.review.project import load_project
from src.review.schemas import FiberTypeDecision, ObjectReviewStatus
from src.review.session import ReviewSession
from src.review.storage import save_session
from tests.test_no_model_project import FOUR_CLASS, I_IIA_PANEL

TRAINING_MICE = ("m1", "m2", "m3")
TEST_MOUSE = "m4"
CLASSES = ["i", "iia", "negative"]
TRUTH = {
    int(label): {"i": "i", "iia": "iia"}.get(kind, "negative")
    for label, kind in json.loads((FOUR_CLASS / "expected_fiber_types.json").read_text())[
        "fiber_type_by_label"
    ].items()
}


def _tiled(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The four-class fixture tiled 3 x 3 (144 fibers), with a little per-tile brightness noise."""
    image = tifffile.imread(FOUR_CLASS / "synthetic_four_class.tif").astype(np.float32)
    labels = tifffile.imread(FOUR_CLASS / "synthetic_four_class_labels.tif").astype(np.int32)
    rng = np.random.default_rng(seed)
    rows_image, rows_labels = [], []
    for row in range(3):
        tiles_image, tiles_labels = [], []
        for column in range(3):
            offset = 16 * (3 * row + column)
            tiles_image.append(image * rng.uniform(0.85, 1.15))
            tiles_labels.append(np.where(labels > 0, labels + offset, 0))
        rows_image.append(np.concatenate(tiles_image, axis=2))
        rows_labels.append(np.concatenate(tiles_labels, axis=1))
    stacked = np.concatenate(rows_image, axis=1)
    return stacked.clip(0, 65535).astype(np.uint16), np.concatenate(rows_labels, axis=0)


def _truth(fiber_id: int) -> str:
    return TRUTH[(fiber_id - 1) % 16 + 1]


@pytest.fixture(scope="module")
def built_project(tmp_path_factory):
    """A no-model project measured by the real pipeline (masks supplied, no Cellpose)."""
    import sys

    from src import run_pipeline

    root = tmp_path_factory.mktemp("builder")
    (root / "images").mkdir()
    panel = root / "panel.yaml"
    panel.write_text(I_IIA_PANEL)
    batch = root / "batch"
    sheet_rows, summary_rows = [], []
    for seed, mouse in enumerate((*TRAINING_MICE, TEST_MOUSE)):
        image_id = f"syn_{mouse}"
        image, labels = _tiled(seed)
        tifffile.imwrite(root / "images" / f"{image_id}.tif", image, photometric="minisblack")
        tifffile.imwrite(root / "images" / f"{image_id}_labels.tif", labels)
        argv = sys.argv
        sys.argv = [
            "run_pipeline",
            "--input", str(root / "images" / f"{image_id}.tif"),
            "--labels-path", str(root / "images" / f"{image_id}_labels.tif"),
            "--output-dir", str(batch / image_id),
            "--image-id", image_id,
            "--panel-config", str(panel),
            "--no-model", "--cpu", "--bootstrap-reps", "10",
        ]  # fmt: skip
        try:
            run_pipeline.main()
        finally:
            sys.argv = argv
        sheet_rows.append(f"{image_id},{mouse},images/{image_id}.tif")
        summary_rows.append(f"{image_id},success")
    (batch / "batch_summary.csv").write_text("image_name,status\n" + "\n".join(summary_rows))
    sheet = root / "samples.csv"
    sheet.write_text("image_id,mouse_id,raw_image_path\n" + "\n".join(sheet_rows))
    review = root / "review"
    review.mkdir()
    manifest, _ = build_project(
        batch, sheet, panel, review, project_id="builder", project_name="builder",
        classes=CLASSES, negative_label="I⁻/IIa⁻",
    )  # fmt: skip
    (review / "project.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    roles = [f"{mouse},pool" for mouse in TRAINING_MICE] + [f"{TEST_MOUSE},test"]
    (root / "evaluation_roles.csv").write_text("mouse_id,role\n" + "\n".join(roles))
    return root


def _label(project, reviewer: str, decisions: dict[tuple[str, int], str]) -> None:
    """Save a reviewer's blind labels; values may be a class, 'exclude', or 'uncertain'."""
    view = project.for_reference(reviewer)
    session = ReviewSession(project_id=project.project_id, model_version="none", reviewer=reviewer)
    for image in project.images:
        session.record_input_fingerprints(image.image_id, image_input_fingerprints(image))
    for (image_id, fiber_id), value in decisions.items():
        status = {
            "exclude": ObjectReviewStatus.EXCLUDED,
            "uncertain": ObjectReviewStatus.UNCERTAIN,
        }.get(value, ObjectReviewStatus.CORRECTED)
        session.record_fiber_type_decision(
            FiberTypeDecision(
                image_id=image_id,
                fiber_id=fiber_id,
                model_fiber_type="unclassified",
                reviewed_fiber_type=value,
                review_status=status,
                queue_source="full_audit",
                reviewer=reviewer,
            )
        )
    save_session(view.review_state_path, session)


def _all_fibers(mice) -> dict[tuple[str, int], str]:
    return {(f"syn_{mouse}", fiber): _truth(fiber) for mouse in mice for fiber in range(1, 145)}


@pytest.fixture()
def labelled(built_project, tmp_path):
    """Reviewer a labels every fiber (training and test mice); reviewer b overlaps on m1:
    agreeing on 10 fibers, disagreeing on 2, and leaving 1 excluded and 1 uncertain."""
    import shutil

    root = tmp_path / "project"
    shutil.copytree(built_project, root)
    project = load_project(root / "review" / "project.yaml")
    _label(project, "a", _all_fibers((*TRAINING_MICE, TEST_MOUSE)))
    other = {("syn_m1", fiber): _truth(fiber) for fiber in range(1, 11)}
    for fiber in (11, 12):
        other[("syn_m1", fiber)] = "iia" if _truth(fiber) != "iia" else "i"
    other[("syn_m1", 13)] = "exclude"
    other[("syn_m1", 14)] = "uncertain"
    _label(project, "b", other)
    return root, project


def _build(project, root, **kwargs):
    sessions = blind_label_sessions(project, kwargs.pop("reviewers", None))
    return build(
        project, sessions, classes=CLASSES, markers=("i", "iia"),
        output_dir=root / "models" / kwargs.pop("name", "build_test"),
        class_labels={"negative": "I⁻/IIa⁻"}, **kwargs,
    )  # fmt: skip


# --- end to end -----------------------------------------------------------------------------


def test_build_end_to_end(labelled):
    root, project = labelled
    result = _build(project, root)
    out = root / "models" / "build_test"

    labels = result["labels"]
    assert labels["training_mice"] == list(TRAINING_MICE)
    assert labels["left_out"]["test_mice"] == 144  # every test-mouse label isolated
    assert labels["conflicting_fibers"] == 2
    assert labels["agreeing_duplicates_merged"] == 10  # agreement does not add samples
    assert labels["left_out"]["excluded"] == 1 and labels["left_out"]["uncertain"] == 1
    assert labels["n_fibers"] == 3 * 144 - 2
    training = pd.read_csv(out / "training_labels.csv")
    assert TEST_MOUSE not in set(training["mouse_id"])
    assert not training.duplicated(["image_id", "fiber_id"]).any()
    by_mouse = pd.read_csv(out / "selection" / "metrics_by_mouse.csv")
    assert set(by_mouse["mouse_id"]) == set(TRAINING_MICE)

    # Features: marker intensities only, in a fixed order.
    assert result["features"] == training_features(("i", "iia"))
    assert all(name.startswith(("type_i.", "type_iia.")) for name in result["features"])
    record = json.loads((root / "batch/syn_m1/syn_m1_run.json").read_text())
    assert result["feature_extraction"] == record["feature_extraction"]
    assert set(result["feature_extraction"]) == FEATURE_EXTRACTION_FIELDS

    decision = result["selection"]["decision"]
    assert decision["recommended"] in {"logistic", "random_forest", "gradient_boosting"}
    assert result["test_mice"].startswith("not used")

    # Every candidate is refitted and loads through the pipeline's own model path.
    diagnostics = pd.read_csv(root / "batch/syn_m4/syn_m4_feature_diagnostics.csv")
    for name, entry in result["final_models"]["candidates"].items():
        manifest = load_model_manifest(out / entry["manifest"])
        assert list(manifest.outputs) == CLASSES
        assert manifest.required_markers == {"laminin", "type_i", "type_iia"}
        assert manifest.feature_extraction == record["feature_extraction"]
        raw = yaml.safe_load((out / entry["manifest"]).read_text())
        assert raw["recommended"] == (name == decision["recommended"])
        assert raw["class_labels"] == {"negative": "I⁻/IIa⁻"}
        predicted = predict_semantic_candidate(diagnostics, out / entry["artifact"], manifest)
        assert set(predicted["model_prediction"]) <= set(CLASSES)


def test_no_passing_candidate_still_writes_models_and_recommends_none(labelled):
    root, project = labelled
    recipe = load_recipe()
    recipe["recommendation_rule"]["minimum_pooled_class_recall"] = 1.01  # unreachable
    result = _build(project, root, recipe=recipe)

    assert result["selection"]["decision"]["recommended"] is None
    out = root / "models" / "build_test"
    for entry in result["final_models"]["candidates"].values():
        assert (out / entry["artifact"]).is_file()
        assert yaml.safe_load((out / entry["manifest"]).read_text())["recommended"] is False


def test_reviewer_filter_uses_only_that_reviewers_labels(labelled):
    root, project = labelled
    result = _build(project, root, reviewers=["a"])
    assert result["labels"]["reviewers"] == ["a"]
    assert result["labels"]["conflicting_fibers"] == 0
    with pytest.raises(BuilderError, match="nobody"):
        blind_label_sessions(project, ["nobody"])


def test_labels_made_on_different_masks_are_left_out(labelled):
    root, project = labelled
    view = project.for_reference("a")
    state = json.loads(view.review_state_path.read_text())
    state["input_fingerprints"]["syn_m1"]["fiber_labels"] = "0" * 64
    view.review_state_path.write_text(json.dumps(state))
    sessions = blind_label_sessions(project, ["a"])
    table, audit = combine_labels(project, sessions, CLASSES)
    assert audit["left_out"]["inputs_changed_since_labelling"] == 144
    assert "syn_m1" not in set(table["image_id"])
    # With m1's labels gone only two training mice remain, so building is refused.
    with pytest.raises(BuilderError, match="2 training mice"):
        _build(project, root, reviewers=["a"])


def test_sections_measured_differently_are_refused(labelled):
    root, project = labelled
    path = root / "batch/syn_m2/syn_m2_run.json"
    record = json.loads(path.read_text())
    record["feature_extraction"]["typing_erode_px"] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(BuilderError, match="different settings"):
        _build(project, root)


def test_a_project_with_a_model_is_refused(labelled):
    root, project = labelled
    from dataclasses import replace

    with pytest.raises(BuilderError, match="improve"):
        _build(replace(project, model_version="quad_four_class_rf_v1"), root)


# --- rules ----------------------------------------------------------------------------------


def _table(counts: dict[str, dict[str, int]]) -> pd.DataFrame:
    """counts[mouse][class] -> that many labelled fibers."""
    rows = [
        {"mouse_id": mouse, "label": label}
        for mouse, by_class in counts.items()
        for label, n in by_class.items()
        for _ in range(n)
    ]
    return pd.DataFrame(rows)


def test_minimums():
    recipe = load_recipe()
    enough = {m: {"i": 12, "iia": 12, "negative": 12} for m in ("a", "b", "c")}
    check_minimums(_table(enough), CLASSES, recipe)

    with pytest.raises(BuilderError, match="2 training mice; 3 needed"):
        check_minimums(_table({m: {"i": 20, "iia": 20, "negative": 20} for m in "ab"}), CLASSES,
                       recipe)  # fmt: skip
    with pytest.raises(BuilderError, match="i: 27 labelled fibers; 30 needed"):
        check_minimums(_table({**enough, "a": {"i": 3, "iia": 12, "negative": 12}}), CLASSES,
                       recipe)  # fmt: skip
    # 30 labels of a class, but only one mouse contributes at least 5 of them.
    one_mouse = {
        "a": {"i": 26, "iia": 12, "negative": 12},
        "b": {"i": 4, "iia": 12, "negative": 12},
        "c": {"iia": 12, "negative": 12},
    }
    with pytest.raises(BuilderError, match="i: 1 mice with at least 5 labels; 2 needed"):
        check_minimums(_table(one_mouse), CLASSES, recipe)


def test_absent_class_has_na_recall_and_macro_f1_uses_present_classes():
    truth = pd.Series(["i", "i", "iia", "iia", "negative"])
    predicted = pd.Series(["i", "iia", "iia", "iia", "negative"])
    mice = pd.Series(["a", "a", "a", "b", "b"])

    metrics = mouse_metrics(truth, predicted, mice, CLASSES).set_index("mouse_id")
    assert metrics.loc["a", "recall_i"] == 0.5
    assert np.isnan(metrics.loc["a", "recall_negative"])  # absent from mouse a: N/A, not 0
    assert metrics.loc["a", "support_negative"] == 0
    assert np.isnan(metrics.loc["b", "recall_i"])
    assert metrics.loc["b", "macro_f1"] == 1.0  # only iia and negative are present in b


def test_baseline_is_the_majority_of_each_folds_training_mice():
    labels = {"a": {"i": 10, "iia": 1}, "b": {"i": 10, "iia": 1}, "c": {"iia": 30, "i": 1}}
    table = _table(labels)
    table["image_id"], table["fiber_id"] = table["mouse_id"], range(len(table))
    table["f"] = np.where(table["label"] == "i", 1.0, 0.0)
    recipe = {"candidates": {"logistic": load_recipe()["candidates"]["logistic"]}}

    predictions, _, _ = evaluate(table, ["f"], ["i", "iia"], recipe)
    baseline = predictions.groupby("mouse_id")[BASELINE].first()
    # Held out a: training b + c has more IIa; held out c: training a + b has more I.
    assert baseline.to_dict() == {"a": "iia", "b": "iia", "c": "i"}


def test_decide_applies_both_conditions_and_breaks_ties_in_recipe_order():
    recipe = load_recipe()
    rows = []
    for mouse, base in (("a", 0.3), ("b", 0.4), ("c", 0.5)):
        rows += [
            {"model": BASELINE, "mouse_id": mouse, "macro_f1": base},
            {"model": "logistic", "mouse_id": mouse, "macro_f1": base + 0.2},
            {"model": "random_forest", "mouse_id": mouse, "macro_f1": base + 0.2},
            {"model": "gradient_boosting", "mouse_id": mouse, "macro_f1": base + 0.05},
        ]
    by_mouse = pd.DataFrame(rows)

    def summary(recall):
        return {"median_mouse_macro_f1": 0.6, "pooled_recall_by_class": {"i": recall, "iia": 0.9}}

    result = decide(
        by_mouse,
        {"logistic": summary(0.9), "random_forest": summary(0.9),
         "gradient_boosting": summary(0.9), BASELINE: summary(0.0)},
        recipe,
    )  # fmt: skip
    assert result["recommended"] == "logistic"  # tie with random_forest: recipe order
    assert not result["checks"]["gradient_boosting"]["passes"]  # gain 0.05 < 0.10

    result = decide(
        by_mouse,
        {"logistic": summary(0.69), "random_forest": summary(0.69),
         "gradient_boosting": summary(0.9), BASELINE: summary(0.0)},
        recipe,
    )  # fmt: skip
    assert result["recommended"] is None  # recall below 0.70, or gain too small


def test_training_features_exclude_area_dapi_and_emhc():
    names = training_features(("i", "iia"))
    assert names[0] == "type_i.mean" and len(names) == 18
    assert not any(token in name for name in names for token in ("area", "dapi", "emhc"))


# --- command line ---------------------------------------------------------------------------


def test_build_command_runs_and_reports(labelled, capsys):
    from src.build_model import main

    root, _ = labelled
    code = main(
        ["--project", str(root / "review/project.yaml"), "--panel-config", str(root / "panel.yaml"),
         "--output-dir", str(root / "models")]
    )  # fmt: skip
    printed = capsys.readouterr().out
    assert code == 0
    assert "430 labelled fibers from 3 training mice" in printed
    assert "test_mice 144" in printed and "reviewer conflicts 2" in printed
    assert "Recommendation:" in printed
    assert "pooled recall (i, iia, I⁻/IIa⁻)" in printed  # negative is named for the panel


def test_cli_build_requires_a_no_model_project(tmp_path, monkeypatch, capsys):
    from src import cli
    from tests.test_no_model_project import _project_folder

    calls = []
    def fake_run(module, arguments):
        calls.append((module, arguments))
        return 0

    monkeypatch.setattr(cli, "_run", fake_run)
    root = _project_folder(tmp_path)
    assert cli.main(["build", str(root)]) == 2  # not prepared yet
    (root / "review").mkdir()
    (root / "review/project.yaml").write_text("images: []\n")
    assert cli.main(["build", str(root), "--reviewer", "a"]) == 0
    module, arguments = calls[-1]
    assert module == "build_model" and arguments[-2:] == ["--reviewer", "a"]

    typed = _project_folder(tmp_path / "typed", model="some_model_id")
    assert cli.main(["build", str(typed)]) == 2
    assert "improve" in capsys.readouterr().err


def test_blind_labelling_offers_every_mouse_in_a_no_model_project(labelled, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qtpy = pytest.importorskip("qtpy.QtWidgets")
    from src.review.fiber_type_review import FiberTypeReviewController
    from src.review.guided_review_widget import GuidedReviewWidget

    application = qtpy.QApplication.instance() or qtpy.QApplication([])
    _, project = labelled
    view = project.for_reference("c")
    session = ReviewSession(project_id=project.project_id, model_version="none", reviewer="c")
    widget = GuidedReviewWidget(view, FiberTypeReviewController(session), blind=True)
    # Roles are declared, but training (pool) mice must be labelled blind too.
    assert widget.allowed_images is None
    assert set(widget.rows["mouse_id"]) == {*TRAINING_MICE, TEST_MOUSE}
    application.processEvents()
    widget.close()
    widget.deleteLater()
