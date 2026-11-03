"""Projects without a model: fibers are measured but not typed, and are labelled into the
classes named in the project config (model builder milestone E1/E2)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml

from src import cli
from src.fiber_type_labels import (
    default_classes,
    negative_display_label,
    normalize_review_label,
    to_biological_label,
    validate_classes,
)
from src.make_review_project import build_project
from src.quantify_classify import QuantifyConfig, quantify_labels
from src.review.project import load_project
from src.run_batch import BatchChannelOverrides, build_batch_command

REPO_ROOT = Path(__file__).resolve().parents[1]
FOUR_CLASS = REPO_ROOT / "examples" / "reference_four_class"
I_IIA_PANEL = (
    "channels:\n  laminin: 2\n  dapi: null\n  type_i: 0\n  type_iia: 1\n  type_iib: null\n"
    "  type_iix: null\n  emhc: null\nclassification:\n  residual_inference:\n"
    "    enabled: false\n    target_class: null\n    requires_negative_markers: []\n"
)


# --- vocabulary ---------------------------------------------------------------------------


def test_unclassified_negative_and_legacy_unknown_stay_distinct():
    # The legacy rule call 'unknown' keeps its historical meaning (IIx by absence) ...
    assert to_biological_label("unknown") == "iix_candidate"
    assert normalize_review_label("unknown") == "iix"
    # ... while the new tokens are never read as a fiber type they are not.
    assert normalize_review_label("unclassified") == "unclassified"
    assert normalize_review_label("negative") == "negative"
    assert normalize_review_label("uncertain") == "uncertain"


def test_default_classes_and_negative_label_follow_the_panel():
    assert default_classes(("i", "iia")) == ["i", "iia", "negative"]
    assert negative_display_label(("i", "iia")) == "I⁻/IIa⁻"
    assert negative_display_label(("iia", "iib")) == "IIa⁻/IIb⁻"


@pytest.mark.parametrize(
    ("classes", "message"),
    [
        (["i", "iib", "negative"], "no channel"),
        (["i", "unclassified"], "unsupported"),
        (["i", "unknown"], "unsupported"),
        (["i"], "at least two"),
        (["i", "i", "iia"], "more than once"),
        ("i, iia", "list"),
    ],
)
def test_configured_classes_are_validated(classes, message):
    with pytest.raises(ValueError, match=message):
        validate_classes(classes, ("i", "iia"))


def test_hybrid_is_offered_only_when_configured():
    assert validate_classes(["I", "IIa", "hybrid", "negative"], ("i", "iia")) == [
        "i",
        "iia",
        "hybrid",
        "negative",
    ]


# --- measuring without typing -------------------------------------------------------------


def _four_class_inputs():
    image = tifffile.imread(FOUR_CLASS / "synthetic_four_class.tif")
    labels = tifffile.imread(FOUR_CLASS / "synthetic_four_class_labels.tif").astype(np.int32)
    return image, labels


def test_typing_disabled_measures_every_fiber_and_types_none():
    image, labels = _four_class_inputs()
    config = QuantifyConfig(type1_channel=3, type2_channel=1, i_channel=0, typing_disabled=True)

    fibers = quantify_labels(labels, image, config)

    assert len(fibers) == len(np.unique(labels)) - 1
    assert set(fibers["fiber_type"]) == {"unclassified"}
    assert set(fibers["fiber_type_source"]) == {"no_model"}
    assert not fibers["needs_review"].any()
    # The marker measurements a model is later built from are still collected.
    assert {"type1_mean", "type2_mean"}.issubset(fibers.columns)


def test_typing_is_unchanged_when_not_disabled():
    image, labels = _four_class_inputs()
    config = QuantifyConfig(type1_channel=3, type2_channel=1, i_channel=0)

    fibers = quantify_labels(labels, image, config)

    assert "unclassified" not in set(fibers["fiber_type"])
    assert fibers["classification_method"].str.startswith("rules:").all()


def test_batch_command_without_a_model_passes_no_model_and_no_frozen_flags(tmp_path):
    panel = tmp_path / "panel.yaml"
    panel.write_text(I_IIA_PANEL)
    cmd = build_batch_command(
        tmp_path / "image.tif",
        tmp_path / "out",
        channel_overrides=BatchChannelOverrides(channel_config=panel),
        no_model=True,
    )

    assert "--no-model" in cmd
    for flag in ("--model", "--classifier-path", "--model-manifest", "--typing-preprocess"):
        assert flag not in cmd


@pytest.mark.parametrize(
    "extra", [["--model", "x"], ["--classifier-path", "x.joblib"], []]
)
def test_run_batch_rejects_no_model_with_a_model_or_without_a_panel(tmp_path, monkeypatch, extra):
    from src import run_batch

    arguments = ["run_batch", "--input-dir", str(tmp_path), "--no-model", *extra]
    monkeypatch.setattr(run_batch.sys, "argv", arguments)
    with pytest.raises(SystemExit) as error:
        run_batch.main()
    assert error.value.code == 2


# --- project config and commands ----------------------------------------------------------


def _project_folder(tmp_path: Path, **overrides) -> Path:
    root = tmp_path / "proj"
    (root / "images").mkdir(parents=True)
    (root / "panel.yaml").write_text(I_IIA_PANEL)
    (root / "samples.csv").write_text("image_id,mouse_id,raw_image_path\n")
    config = {
        "schema_version": "fibertypeqc_project.v1",
        "images": "images",
        "panel": "panel.yaml",
        "sample_sheet": "samples.csv",
        "model": "none",
    }
    config.update(overrides)
    (root / cli.CONFIG_NAME).write_text(yaml.safe_dump(config, allow_unicode=True))
    return root


def test_model_none_gives_a_no_model_project_with_classes_from_the_panel(tmp_path):
    project = cli.load_project_folder(_project_folder(tmp_path))

    assert project.no_model and project.model is None
    assert project.classes == ("i", "iia", "negative")
    assert project.negative_label == "I⁻/IIa⁻"


def test_configured_classes_are_used_and_checked(tmp_path):
    root = _project_folder(tmp_path, classes=["i", "iia", "hybrid", "negative"])
    assert cli.load_project_folder(root).classes == ("i", "iia", "hybrid", "negative")

    with pytest.raises(cli.ProjectError, match="no channel"):
        cli.load_project_folder(_project_folder(tmp_path / "b", classes=["i", "iib"]))
    with pytest.raises(cli.ProjectError, match="only used with 'model: none'"):
        cli.load_project_folder(
            _project_folder(tmp_path / "c", model="some_model_id", classes=["i", "iia"])
        )


def test_a_missing_model_still_means_the_default_model(tmp_path):
    project = cli.load_project_folder(_project_folder(tmp_path, model=None))
    assert not project.no_model and project.classes == ()


def test_no_model_steps(tmp_path, monkeypatch, capsys):
    root = _project_folder(tmp_path)
    calls: list[tuple[str, list[str]]] = []

    def fake_run(module, arguments):
        calls.append((module, arguments))
        if module == "make_review_project":
            (root / "review").mkdir(exist_ok=True)
            (root / "review/project.yaml").write_text("images: []\n")
        return 0

    monkeypatch.setattr(cli, "_run", fake_run)

    assert cli.main(["run", str(root)]) == 0
    module, arguments = calls[-1]
    assert module == "run_batch" and "--no-model" in arguments and "--model" not in arguments

    (root / "batch").mkdir()
    (root / "batch/batch_summary.csv").write_text("image_name,status\na,success\n")
    (root / "samples.csv").write_text("image_id,mouse_id,raw_image_path\na,m1,images/a.tif\n")
    assert cli.main(["prepare", str(root)]) == 0
    arguments = next(args for name, args in calls if name == "make_review_project")
    assert arguments[arguments.index("--classes") + 1] == "i,iia,negative"
    assert arguments[arguments.index("--negative-label") + 1] == "I⁻/IIa⁻"

    (root / "review/qc").mkdir(parents=True)
    (root / "review/qc/fiber_qc.csv").write_text("fiber_id\n")
    assert cli.main(["review", str(root)]) == 2
    assert "--blind" in capsys.readouterr().err
    assert cli.main(["review", str(root), "--blind", "--reviewer", "a"]) == 0
    for step in ("finalize", "improve"):
        assert cli.main([step, str(root)]) == 2
        assert "no model" in capsys.readouterr().err
    # Promoting a built model is allowed (E4); evaluate scores a chosen candidate on test mice.
    assert cli.main(["promote", str(root), "logistic"]) == 0
    assert calls[-2][0] == "promote_model" and calls[-1][0] == "generate_review_qc"
    assert cli.main(["evaluate", str(root), "logistic"]) == 0
    module, arguments = calls[-1]
    assert module == "evaluate_model" and arguments[-2:] == ["--candidate", "logistic"]


def test_status_counts_blind_labels_and_points_to_blind_labelling(tmp_path, capsys):
    root = _project_folder(tmp_path)
    (root / "batch").mkdir()
    (root / "batch/batch_summary.csv").write_text("image_name,status\na,success\n")
    (root / "samples.csv").write_text("image_id,mouse_id,raw_image_path\na,m1,images/a.tif\n")
    (root / "review/qc").mkdir(parents=True)
    (root / "review/project.yaml").write_text("images: [{}]\n")
    (root / "review/qc/fiber_qc.csv").write_text("fiber_id\n")
    (root / "review/reference/rev1").mkdir(parents=True)
    (root / "review/reference/rev1/review_state.json").write_text(
        json.dumps({"object_decisions": [{}, {}, {}]})
    )

    assert cli.main(["status", str(root)]) == 0
    printed = capsys.readouterr().out
    assert "Model:   none; classes i, iia, negative" in printed
    assert "rev1 3" in printed
    assert "--blind" in printed


# --- review project -----------------------------------------------------------------------


def _batch(tmp_path: Path, typing: str | None) -> tuple[Path, Path, Path]:
    batch = tmp_path / "batch"
    image_dir = batch / "a"
    image_dir.mkdir(parents=True)
    tifffile.imwrite(image_dir / "a_cellpose_labels.tif", np.ones((4, 4), np.uint16))
    pd.DataFrame({"label": [1], "fiber_type": ["unclassified"]}).to_csv(
        image_dir / "a_fibers.csv", index=False
    )
    record = {"classifier_path": None, "model_manifest_path": None}
    if typing:
        record["typing"] = typing
    (image_dir / "a_run.json").write_text(json.dumps(record))
    (batch / "batch_summary.csv").write_text("image_name,status\na,success\n")
    tifffile.imwrite(tmp_path / "a.tif", np.zeros((3, 4, 4), np.uint16))
    sheet = tmp_path / "samples.csv"
    sheet.write_text("image_id,mouse_id,raw_image_path\na,m1,a.tif\n")
    panel = tmp_path / "panel.yaml"
    panel.write_text(I_IIA_PANEL)
    return batch, sheet, panel


def test_review_project_without_a_model_records_its_classes(tmp_path):
    batch, sheet, panel = _batch(tmp_path, typing="none")
    review = tmp_path / "review"
    review.mkdir()

    project, _ = build_project(
        batch, sheet, panel, review, project_id="p", project_name="p",
        classes=["i", "iia", "negative"], negative_label="I⁻/IIa⁻",
    )  # fmt: skip
    assert project["model_version"] == "none"
    (review / "project.yaml").write_text(yaml.safe_dump(project, sort_keys=False))
    loaded = load_project(review / "project.yaml")
    assert loaded.review_classes == ("i", "iia", "negative")
    assert loaded.class_labels == {"negative": "I⁻/IIa⁻"}

    with pytest.raises(ValueError, match="--classes"):
        build_project(batch, sheet, panel, review, project_id="p", project_name="p")


def test_classes_are_refused_for_a_typed_batch(tmp_path):
    batch, sheet, panel = _batch(tmp_path, typing=None)
    review = tmp_path / "review"
    review.mkdir()
    project, _ = build_project(batch, sheet, panel, review, project_id="p", project_name="p")
    assert project["model_version"] == "rules"
    assert "review_classes" not in project
    with pytest.raises(ValueError, match="only used for a batch run without a model"):
        build_project(
            batch, sheet, panel, review, project_id="p", project_name="p", classes=["i", "iia"]
        )


def test_loading_a_no_model_project_requires_classes(tmp_path):
    from tests.test_review_finalization import _project

    project = _project(tmp_path)
    raw = yaml.safe_load(project.manifest_path.read_text())
    raw["model_version"] = "none"
    project.manifest_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="review_classes is required"):
        load_project(project.manifest_path, validate_paths=False)


# --- blind labelling ----------------------------------------------------------------------


def _no_model_review_project(tmp_path):
    from tests.test_review_finalization import _project

    project = _project(tmp_path)
    table = tmp_path / "one" / "one_fibers.csv"
    fibers = pd.read_csv(table)
    fibers["fiber_type"] = "unclassified"
    fibers.to_csv(table, index=False)
    return replace(
        project,
        model_version="none",
        review_classes=("i", "iia", "negative"),
        class_labels={"negative": "I⁻/IIa⁻"},
    )


def test_blind_labelling_offers_the_configured_classes(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qtpy = pytest.importorskip("qtpy.QtWidgets")
    from src.review.fiber_type_review import FiberTypeReviewController
    from src.review.guided_review_widget import GuidedReviewWidget
    from src.review.session import ReviewSession

    application = qtpy.QApplication.instance() or qtpy.QApplication([])
    project = _no_model_review_project(tmp_path).for_reference("rev1")
    session = ReviewSession(
        project_id=project.project_id, model_version=project.model_version, reviewer="rev1"
    )
    widget = GuidedReviewWidget(project, FiberTypeReviewController(session), blind=True)

    assert widget.classes == {"i", "iia", "negative"}
    assert [(key, name) for key, name, _ in widget.class_buttons] == [
        ("1", "I"),
        ("2", "IIa"),
        ("6", "I⁻/IIa⁻"),
    ]
    application.processEvents()
    widget.close()
    widget.deleteLater()


def test_negative_is_recorded_and_an_unclassified_call_cannot_be_kept(tmp_path):
    from src.review.fiber_type_review import FiberTypeReviewController
    from src.review.queues import QueueSource, build_fiber_type_queue, load_fiber_type_rows
    from src.review.schemas import ObjectReviewStatus
    from src.review.session import ReviewSession

    project = _no_model_review_project(tmp_path)
    rows = load_fiber_type_rows(project)
    assert set(rows["model_fiber_type"]) == {"unclassified"}
    queue = build_fiber_type_queue(rows, QueueSource.FULL_AUDIT)
    session = ReviewSession(project_id=project.project_id, model_version="none", reviewer="rev1")
    controller = FiberTypeReviewController(session, queue)

    with pytest.raises(ValueError, match="no model call to keep"):
        controller.decide(None)
    controller.decide("negative")
    decision = session.object_decisions[0]
    assert decision.reviewed_fiber_type == "negative"
    assert decision.review_status is ObjectReviewStatus.CORRECTED


def test_pipeline_without_a_model_writes_measurements_and_no_calls(tmp_path, monkeypatch):
    from src import run_pipeline

    panel = tmp_path / "panel.yaml"
    panel.write_text(I_IIA_PANEL)
    out = tmp_path / "out"
    monkeypatch.setattr(
        run_pipeline.sys,
        "argv",
        [
            "run_pipeline",
            "--input", str(FOUR_CLASS / "synthetic_four_class.tif"),
            "--labels-path", str(FOUR_CLASS / "synthetic_four_class_labels.tif"),
            "--output-dir", str(out),
            "--image-id", "syn",
            "--panel-config", str(panel),
            "--no-model",
            "--cpu",
            "--bootstrap-reps", "10",
        ],  # fmt: skip
    )
    run_pipeline.main()

    fibers = pd.read_csv(out / "syn_fibers.csv")
    assert set(fibers["fiber_type"]) == {"unclassified"}
    diagnostics = pd.read_csv(out / "syn_feature_diagnostics.csv")
    assert {"type_i.mean", "type_iia.mean"}.issubset(diagnostics.columns)
    postrun = json.loads((out / "syn_postrun_qc.json").read_text())
    statuses = {check["code"]: check["status"] for check in postrun["checks"]}
    assert statuses["postrun.classification"] == "not_applicable"
    assert "postrun.uncertainty_rate" not in statuses
    preflight = json.loads((out / "syn_preflight_qc.json").read_text())
    statuses = {check["code"]: check["status"] for check in preflight["checks"]}
    assert statuses["preflight.model_panel_compatible"] == "not_applicable"
    assert statuses["preflight.panel_compatible"] == "pass"
    record = json.loads((out / "syn_run.json").read_text())
    assert record["typing"] == "none" and record["classifier_sha256"] is None
    summary = pd.read_csv(out / "syn_summary.csv")
    assert summary.loc[0, "prop_unclassified"] == 1.0
    assert [column for column in summary.columns if column.startswith("prop_")] == [
        "prop_unclassified"
    ]


def test_pipeline_refuses_no_model_with_a_model(tmp_path, monkeypatch):
    from src import run_pipeline

    monkeypatch.setattr(
        run_pipeline.sys,
        "argv",
        ["run_pipeline", "--input", "x.tif", "--output-dir", str(tmp_path), "--no-model",
         "--model", "ta_three_class_logistic_v1"],
    )  # fmt: skip
    with pytest.raises(SystemExit) as error:
        run_pipeline.main()
    assert error.value.code == 2


# --- QC without a model -------------------------------------------------------------------


def test_classification_qc_is_not_applicable_and_never_changes_the_overall_status():
    from fibertypeqc.qc_contract import build_qc_report, postrun_checks

    stats = {
        "flag_low_labels": False,
        "flag_high_uncertainty_rate": False,
        "flag_median_area_outlier": False,
        "flag_high_type_corr": False,
        "n_labels": 500,
    }
    limits = {
        "min_labels": 300,
        "max_uncertainty_rate": 0.35,
        "median_area_min": 200.0,
        "median_area_max": 15000.0,
        "max_type_corr": 0.92,
    }
    typed = postrun_checks(stats, **limits)
    untyped = postrun_checks(stats, **limits, typing_applicable=False)

    assert [check["code"] for check in typed] == [
        "postrun.fiber_count",
        "postrun.uncertainty_rate",
        "postrun.median_area",
        "postrun.marker_correlation",
    ]
    assert [(check["code"], check["status"]) for check in untyped] == [
        ("postrun.fiber_count", "pass"),
        ("postrun.classification", "not_applicable"),
        ("postrun.median_area", "pass"),
        ("postrun.marker_correlation", "pass"),
    ]
    assert build_qc_report(stage="postrun", checks=untyped)["overall_status"] == "pass"
    # Segmentation checks still warn as usual.
    low = postrun_checks({**stats, "flag_low_labels": True}, **limits, typing_applicable=False)
    assert build_qc_report(stage="postrun", checks=low)["overall_status"] == "warn"


def test_review_qc_marks_typing_not_applicable_without_a_model(tmp_path):
    from src.review.qc import generate_project_qc
    from src.review.qc_rules import load_rule_config

    project = _no_model_review_project(tmp_path)
    image_qc = generate_project_qc(project, load_rule_config()).image_qc
    by_domain = image_qc.set_index("domain")

    assert by_domain.loc["fiber_typing", "status"] == "not_applicable"
    assert not by_domain.loc["fiber_typing", "applicable"]
    assert by_domain.loc["fiber_segmentation", "status"] in {"pass", "review", "fail"}
    # The typing measurements are still recorded.
    assert by_domain.loc["fiber_typing", "typing_row_count"] == 4
