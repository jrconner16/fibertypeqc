from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fibertypeqc.panel_presets import list_presets
from src import cli
from src import workflow_steps as ws


@pytest.fixture(autouse=True)
def _model_found(request, monkeypatch):
    """The fixtures name a model file that is not there; only the model tests check for it."""
    if "check_model" not in request.keywords:
        monkeypatch.setattr(ws, "model_problem", lambda _project: "")


def _project(tmp_path: Path, model: str = "models/my_model.joblib", mice=("m1", "m2")) -> Path:
    root = tmp_path / "proj"
    (root / "images").mkdir(parents=True)
    preset = list_presets()["three_marker_iib_iia_laminin"][0]
    (root / "panel.yaml").write_text(preset.read_text())
    pd.DataFrame(
        {
            "image_id": [f"s{n}" for n in range(len(mice))],
            "mouse_id": list(mice),
            "raw_image_path": [f"images/s{n}.tif" for n in range(len(mice))],
        }
    ).to_csv(root / "samples.csv", index=False)
    config = {
        "schema_version": cli.CONFIG_SCHEMA,
        "images": "images",
        "panel": "panel.yaml",
        "sample_sheet": "samples.csv",
        "model": model,
    }
    (root / cli.CONFIG_NAME).write_text(yaml.safe_dump(config))
    return root


def _batch(root: Path) -> None:
    (root / "batch").mkdir(exist_ok=True)
    pd.DataFrame({"image_name": ["s0"], "status": ["success"]}).to_csv(
        root / "batch/batch_summary.csv", index=False
    )


def _prepared(root: Path) -> None:
    _batch(root)
    (root / "review/qc").mkdir(parents=True, exist_ok=True)
    (root / "review/project.yaml").write_text(yaml.safe_dump({"images": [{"image_id": "s0"}]}))
    (root / "review/qc/fiber_qc.csv").write_text("fiber_id\n")


def _flow(root: Path) -> ws.Workflow:
    return ws.workflow(cli.load_project_folder(root))


def _states(flow: ws.Workflow) -> dict[str, str]:
    return {step.key: step.state for step in flow.steps}


# --- workflow steps (no GUI) ----------------------------------------------------------------


def test_model_project_steps_follow_status(tmp_path):
    root = _project(tmp_path)
    flow = _flow(root)
    assert [step.key for step in flow.steps] == list(ws.MODEL_PATH)
    assert flow.current.key == "run" and flow.current.command == "run PROJECT"
    assert _states(flow) == {
        "setup": "done", "samples": "done", "run": "current",
        "prepare": "upcoming", "review": "upcoming", "finalize": "upcoming",
    }  # fmt: skip
    assert flow.step("samples").summary == "2 images · 2 mice"
    assert flow.step("setup").summary.startswith("stains laminin + IIa, IIb · model my_model")

    _batch(root)
    assert _flow(root).current.key == "prepare"
    _prepared(root)
    flow = _flow(root)
    assert flow.current.key == "review" and _states(flow)["prepare"] == "done"

    (root / "review/review").mkdir()
    state = root / "review/review/review_state.json"
    state.write_text(json.dumps({"object_decisions": [{}], "regions": []}))
    (root / "final").mkdir()
    manifest = root / "final/finalization_manifest.json"
    manifest.write_text("{}")
    now = time.time()
    os.utime(state, (now - 100, now - 100))
    assert _flow(root).current.key == "finalize"  # finalized, report not built yet
    (root / "results").mkdir()
    (root / "results/cohort_report.html").write_text("<html></html>")
    flow = _flow(root)
    assert flow.finished and flow.current is None
    assert set(_states(flow).values()) == {"done"}
    os.utime(state, (now + 100, now + 100))  # reviewed again after finalizing
    assert _flow(root).current.key == "finalize"


def test_launcher_and_status_agree_on_the_next_step(tmp_path, capsys):
    root = _project(tmp_path)
    for prepare in (lambda: None, lambda: _batch(root), lambda: _prepared(root)):
        prepare()
        flow = _flow(root)
        cli.main(["status", str(root)])
        printed = capsys.readouterr().out
        command = ws.command_arguments(flow.current.command, cli.load_project_folder(root))
        assert f"python -m fibertypeqc {' '.join(command)}" in printed


def test_missing_mouse_ids_make_samples_the_current_step_after_run(tmp_path):
    root = _project(tmp_path, mice=("m1", ""))
    flow = _flow(root)
    assert flow.current.key == "run"  # segmentation does not need mouse IDs
    assert flow.step("samples").state == "attention"

    _batch(root)
    flow = _flow(root)
    samples = flow.current
    assert samples.key == "samples" and samples.command == ""
    assert samples.summary.startswith("1 image still needs a mouse ID (s1)")
    assert _states(flow)["run"] == "done" and _states(flow)["prepare"] == "upcoming"
    assert flow.notes == ()  # the Samples step explains it; no duplicate note


def test_builder_project_shows_only_builder_steps(tmp_path, monkeypatch):
    root = _project(tmp_path, model="none")
    flow = _flow(root)
    assert [step.key for step in flow.steps] == list(ws.BUILDER_PATH)
    assert flow.current.key == "run" and flow.current.title == "Segment and measure"

    _prepared(root)
    flow = _flow(root)
    assert flow.current.key == "label"
    assert flow.current.command == "review PROJECT --blind --reviewer NAME"
    project = cli.load_project_folder(root)
    assert ws.command_arguments(flow.current.command, project, "ann") == [
        "review", str(root), "--blind", "--reviewer", "ann",
    ]  # fmt: skip

    (root / "review/reference/ann").mkdir(parents=True)
    (root / "review/reference/ann/review_state.json").write_text(
        json.dumps({"object_decisions": [{}, {}]})
    )
    flow = _flow(root)
    assert flow.current.key == "build" and flow.step("label").summary == "ann: 2"

    def built(evaluation):
        return lambda _project: {
            "build": "build_1", "recommended": "logistic", "evaluation": evaluation,
        }  # fmt: skip

    monkeypatch.setattr(cli, "builder_status", built("missing"))
    flow = _flow(root)
    assert flow.current.key == "evaluate" and flow.current.command == "evaluate PROJECT logistic"
    assert flow.step("promote").command == "promote PROJECT logistic"
    monkeypatch.setattr(cli, "builder_status", built("no_test_mice"))
    flow = _flow(root)
    assert flow.current.key == "promote" and _states(flow)["evaluate"] == "skipped"
    assert "no independent check" in flow.step("evaluate").summary


# --- window ---------------------------------------------------------------------------------


@pytest.fixture
def qapp(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("qtpy.QtWidgets")
    return widgets.QApplication.instance() or widgets.QApplication([])


def _window(qapp, root):
    from src.launcher import LauncherWindow

    window = LauncherWindow(root)
    window.show()
    qapp.processEvents()
    return window


def _row(window, key):
    return next(row for row in window.rows if row.step.key == key)


def _wait(qapp, condition, seconds=20.0):
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert condition()


def test_only_the_current_step_has_a_main_button(qapp, tmp_path):
    root = _project(tmp_path)
    window = _window(qapp, root)

    run = _row(window, "run")
    assert run.action_button.text() == "Run" and run.action_button.isEnabled()
    for key in ("prepare", "review", "finalize"):
        assert _row(window, key).action_button is None
    assert _row(window, "samples").again_button.text() == "Edit"
    assert window.project_label.text() == "proj"

    run.copy_button.click()
    assert qapp.clipboard().text() == f"uv run --frozen python -m fibertypeqc run {root}"
    window.close()


def test_builder_evaluate_and_promote_are_copy_only(qapp, tmp_path, monkeypatch):
    root = _project(tmp_path, model="none")
    _prepared(root)
    monkeypatch.setattr(
        cli,
        "builder_status",
        lambda _p: {"build": "build_1", "recommended": "logistic", "evaluation": "missing"},
    )
    (root / "review/reference/ann").mkdir(parents=True)
    (root / "review/reference/ann/review_state.json").write_text(
        json.dumps({"object_decisions": [{}]})
    )
    window = _window(qapp, root)

    evaluate = _row(window, "evaluate")
    assert evaluate.step.state == "current"
    assert evaluate.action_button is None and evaluate.copy_button is not None
    assert not any(row.step.key in ("review", "finalize") for row in window.rows)
    assert _row(window, "build").again_button.text() == "Build again"
    window.close()


def test_review_offers_finalize_once_there_are_decisions(qapp, tmp_path):
    root = _project(tmp_path)
    _prepared(root)
    window = _window(qapp, root)
    review = _row(window, "review")
    assert review.action_button.text() == "Open review" and review.finalize_button is None
    labels = review.findChildren(type(window.project_label))
    assert any("review window" in label.text() for label in labels)
    window.close()

    (root / "review/review").mkdir()
    (root / "review/review/review_state.json").write_text(json.dumps({"object_decisions": [{}]}))
    window = _window(qapp, root)
    started = []
    window.start_arguments = started.append
    _row(window, "review").finalize_button.click()
    assert started == [["finalize", str(root)]]
    window.close()


def test_label_step_needs_a_reviewer_name(qapp, tmp_path):
    root = _project(tmp_path, model="none")
    _prepared(root)
    window = _window(qapp, root)

    label = _row(window, "label")
    assert not label.action_button.isEnabled()
    label.reviewer_edit.setText("ann")
    assert label.action_button.isEnabled()
    window.close()


def test_running_a_step_streams_the_log_and_keeps_the_window_responsive(qapp, tmp_path):
    from qtpy.QtCore import QTimer

    root = _project(tmp_path)
    window = _window(qapp, root)
    started: list[list[str]] = []
    script = (
        "import sys, time\n"
        "print('Cellpose device: cuda', flush=True)\n"
        "for n in range(5):\n"
        "    print(f'image {n}', flush=True); time.sleep(0.15)\n"
        "sys.exit(0)\n"
    )

    def fake(arguments):
        started.append(arguments)
        return sys.executable, ["-c", script]

    window.process_command = fake
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(20)

    _row(window, "run").action_button.click()
    assert started == [["run", str(root)]]
    assert window.running and window.stop_button.isVisible()
    assert not _row(window, "run").action_button.isEnabled()
    assert not window.open_button.isEnabled()
    _wait(qapp, lambda: "image 2" in window.log.toPlainText())
    assert window.running and len(ticks) > 5  # the event loop kept running during the step
    _wait(qapp, lambda: not window.running)
    timer.stop()

    log = window.log.toPlainText()
    assert f"$ uv run --frozen python -m fibertypeqc run {root}" in log
    assert "image 4" in log and "[finished with exit code 0]" in log
    assert window.activity_label.text() == "Finished. Device: cuda."
    assert window.open_button.isEnabled() and _row(window, "run").action_button.isEnabled()
    window.close()


def test_closing_while_a_step_runs_asks_first(qapp, tmp_path, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    root = _project(tmp_path)
    window = _window(qapp, root)
    window.process_command = lambda _a: (sys.executable, ["-c", "import time; time.sleep(30)"])
    _row(window, "run").action_button.click()
    _wait(qapp, lambda: window.process.state() == window.process.Running)

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    window.close()
    assert window.isVisible() and window.running
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    window.close()
    assert not window.isVisible() and not window.running


def _tif_folder(tmp_path: Path) -> Path:
    import tifffile

    folder = tmp_path / "raw"
    folder.mkdir()
    for name in ("m1_s1", "m2_s1"):
        tifffile.imwrite(
            folder / f"{name}.tif",
            np.zeros((3, 8, 8), dtype=np.uint16),
            imagej=True,
            metadata={"axes": "CYX"},
        )
    return folder


def test_new_project_form_writes_what_init_writes(qapp, tmp_path):
    from src.launcher import NewProjectDialog

    images = _tif_folder(tmp_path)
    assert (
        cli.main(
            [
                "init",
                str(tmp_path / "by_init"),
                "--images",
                str(images),
                "--panel",
                "three_marker_iib_iia_laminin",
                "--model",
                "ta_three_class_logistic_v1",
                "--mouse-id-pattern",
                r"^(m\d+)_",
                "--yes",
            ]
        )  # fmt: skip
        == 0
    )

    dialog = NewProjectDialog()
    dialog.parent_edit.setText(str(tmp_path))
    dialog.name_edit.setText("by_form")
    dialog.images_edit.setText(str(images))
    dialog._images_changed()
    assert dialog.images_info.text().startswith("2 image(s); m1_s1.tif has 3 channel(s)")
    dialog.panel_combo.setCurrentIndex(dialog.panel_combo.findData("three_marker_iib_iia_laminin"))
    assert dialog.model_combo.currentData() == "ta_three_class_logistic_v1"  # follows the panel
    dialog.pattern_edit.setText(r"^(m\d+)_")
    assert "m1_s1 → m1" in dialog.mice_preview.text()
    dialog.create()

    assert dialog.created == tmp_path / "by_form"
    for name in ("panel.yaml", "samples.csv", cli.CONFIG_NAME):
        assert (tmp_path / "by_form" / name).read_text() == (
            tmp_path / "by_init" / name
        ).read_text()


def test_new_project_form_routes_unmatched_stains_to_building_a_model(qapp, tmp_path):
    from src.launcher import NO_MODEL, NewProjectDialog

    dialog = NewProjectDialog()
    dialog.parent_edit.setText(str(tmp_path))
    dialog.name_edit.setText("custom")
    dialog.images_edit.setText(str(_tif_folder(tmp_path)))
    dialog._images_changed()
    dialog.panel_combo.setCurrentIndex(dialog.panel_combo.findData("custom"))
    for marker, channel in {"laminin": 0, "type_i": 1, "type_iia": 2}.items():
        combo = dialog.channel_combos[marker]
        combo.setCurrentIndex(combo.findData(channel))
    assert dialog.model_combo.currentData() == NO_MODEL
    assert "build one" in dialog.model_info.text()
    dialog.create()

    project = cli.load_project_folder(tmp_path / "custom")
    assert project.no_model and project.classes


def test_new_project_form_reports_problems_in_place(qapp, tmp_path):
    from src.launcher import NewProjectDialog

    dialog = NewProjectDialog()
    dialog.parent_edit.setText(str(tmp_path))
    dialog.create()
    assert dialog.error_label.text() == "Give the project a name."
    dialog.name_edit.setText("p")
    dialog.images_edit.setText(str(tmp_path / "missing"))
    dialog.create()
    assert "Images folder not found" in dialog.error_label.text() and dialog.created is None


def test_samples_dialog_saves_mouse_ids_and_keeps_image_columns_fixed(qapp, tmp_path):
    from qtpy.QtCore import Qt

    from src.launcher import SamplesDialog

    root = _project(tmp_path, mice=("m1", ""))
    project = cli.load_project_folder(root)
    dialog = SamplesDialog(project)
    assert dialog.message.text() == "1 image still needs a mouse ID."
    assert not dialog.table.item(0, 0).flags() & Qt.ItemIsEditable  # image_id
    dialog.table.item(1, 1).setText("m2")
    assert dialog.message.text() == "2 images from 2 mice."
    dialog.save()

    cli.check_sample_sheet(project)
    sheet = pd.read_csv(root / "samples.csv", dtype=str, keep_default_na=False)
    assert sheet["mouse_id"].tolist() == ["m1", "m2"]
    assert _flow(root).step("samples").state == "done"


def test_app_command_rejects_a_folder_that_is_not_a_project(qapp, tmp_path, capsys):
    assert cli.main(["app", str(tmp_path)]) == 2
    assert "not a FiberTypeQC project folder" in capsys.readouterr().err


@pytest.mark.check_model
def test_a_missing_model_is_explained_before_run(tmp_path):
    root = _project(tmp_path)
    (root / cli.CONFIG_NAME).write_text(
        (root / cli.CONFIG_NAME).read_text() + "# keep this comment\n"
    )
    flow = _flow(root)
    setup = flow.current
    assert setup.key == "setup" and setup.command == ""
    assert setup.explanation.startswith("The model file is not where the project expects it:")
    assert _states(flow)["run"] == "upcoming"

    _batch(root)  # once run has happened, the model no longer blocks the next step
    assert _flow(root).current.key == "prepare"

    ws.set_model(cli.load_project_folder(root), "/models/other model.joblib")
    text = (root / cli.CONFIG_NAME).read_text()
    assert "# keep this comment" in text and text.count("model:") == 1
    assert cli.load_project_folder(root).model == "/models/other model.joblib"


@pytest.mark.check_model
def test_choose_model_file_updates_the_project(qapp, tmp_path, monkeypatch):
    from qtpy.QtWidgets import QFileDialog

    root = _project(tmp_path)
    window = _window(qapp, root)
    setup = _row(window, "setup")
    assert setup.step.state == "current" and setup.action_button.text() == "Choose model file…"
    assert _row(window, "run").action_button is None

    chosen = tmp_path / "elsewhere" / "model.joblib"
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(chosen), ""))
    setup.action_button.click()
    assert window.project.model == str(chosen)
    window.close()


def test_setup_can_be_edited_until_segmentation_runs(qapp, tmp_path):
    from src.launcher import NO_MODEL, NewProjectDialog

    images = _tif_folder(tmp_path)
    root = tmp_path / "study"
    cli.main(
        ["init", str(root), "--images", str(images), "--panel", "three_marker_iib_iia_laminin",
         "--model", "ta_three_class_logistic_v1", "--mouse-id-pattern", r"^(m\d+)_", "--yes"]
    )  # fmt: skip
    (root / cli.CONFIG_NAME).write_text((root / cli.CONFIG_NAME).read_text() + "# mine\n")
    window = _window(qapp, root)
    assert _row(window, "setup").again_button.isEnabled()
    window.close()

    dialog = NewProjectDialog(project=cli.load_project_folder(root))
    assert dialog.panel_combo.currentData() == "three_marker_iib_iia_laminin"
    assert (
        dialog.model_combo.currentData() == "ta_three_class_logistic_v1"
    )  # kept, not re-suggested
    dialog.panel_combo.setCurrentIndex(dialog.panel_combo.findData("custom"))
    for marker, channel in {"laminin": 0, "type_i": 1, "type_iia": 2}.items():
        combo = dialog.channel_combos[marker]
        combo.setCurrentIndex(combo.findData(channel))
    assert dialog.model_combo.currentData() == NO_MODEL
    dialog.save()

    project = cli.load_project_folder(root)
    assert project.no_model and "# mine" in (root / cli.CONFIG_NAME).read_text()
    sheet = pd.read_csv(root / "samples.csv", keep_default_na=False)
    assert sheet["mouse_id"].tolist() == ["m1", "m2"]  # mouse IDs survive the edit

    _batch(root)
    window = _window(qapp, root)
    edit = _row(window, "setup").again_button
    assert not edit.isEnabled() and "Create a new project" in edit.toolTip()
    with pytest.raises(cli.ProjectError):
        ws.update_setup(project, images, {}, "x", "none")
    window.close()


def test_a_private_model_is_explained_in_plain_words():
    message = ws.friendly_model_problem(
        "Model 'quad_four_class_rf_v1' is distributed privately. Set FIBERTYPEQC_MODEL_ROOT to "
        "the directory containing quad_four_class_rf_v1.joblib."
    )
    assert "FIBERTYPEQC_MODEL_ROOT" not in message
    assert "Download quad_four_class_rf_v1.joblib from the FiberTypeQC releases page" in message


def test_new_project_form_warns_when_the_model_is_not_on_this_computer(qapp, monkeypatch):
    from src.launcher import NewProjectDialog

    monkeypatch.delenv("FIBERTYPEQC_MODEL_ROOT", raising=False)
    dialog = NewProjectDialog()
    dialog.panel_combo.setCurrentIndex(dialog.panel_combo.findData("four_marker_i_iia_laminin_iib"))
    assert dialog.model_combo.currentData() == "quad_four_class_rf_v1"
    assert "downloaded separately" in dialog.model_warning.text()
    assert "A model file…" in dialog.model_warning.text()


def test_a_panel_that_does_not_fit_the_images_is_caught_before_run(qapp, tmp_path):
    images = _tif_folder(tmp_path)  # three channels
    root = tmp_path / "study"
    cli.main(
        ["init", str(root), "--images", str(images), "--panel", "four_marker_i_iia_laminin_iib",
         "--model", "quad_four_class_rf_v1", "--yes"]
    )  # fmt: skip
    flow = _flow(root)
    setup = flow.current
    assert setup.key == "setup" and setup.command == ""
    assert "expects Type IIb in channel 3" in setup.explanation
    assert "m1_s1.tif has 3 channels (numbered 0 to 2)" in setup.explanation

    # A run in which every image failed leaves the set-up editable, so it can be fixed.
    (root / "batch").mkdir()
    pd.DataFrame({"image_name": ["m1_s1"], "status": ["failed"]}).to_csv(
        root / "batch/batch_summary.csv", index=False
    )
    project = cli.load_project_folder(root)
    assert ws.can_edit_setup(project) and _flow(root).current.fix == "panel"
    window = _window(qapp, root)
    assert _row(window, "setup").action_button.text() == "Edit set-up…"
    window.close()
    pd.DataFrame({"image_name": ["m1_s1"], "status": ["success"]}).to_csv(
        root / "batch/batch_summary.csv", index=False
    )
    assert not ws.can_edit_setup(project)


def test_new_project_form_picks_a_panel_that_fits_and_refuses_one_that_does_not(qapp, tmp_path):
    from src.launcher import NewProjectDialog

    dialog = NewProjectDialog()
    assert dialog.panel_combo.currentData() == "four_marker_i_iia_laminin_iib"  # list order
    dialog.parent_edit.setText(str(tmp_path))
    dialog.name_edit.setText("p")
    dialog.images_edit.setText(str(_tif_folder(tmp_path)))
    dialog._images_changed()
    assert dialog.panel_combo.currentData() == "three_marker_iib_iia_laminin"

    dialog.panel_combo.setCurrentIndex(dialog.panel_combo.findData("four_marker_i_iia_laminin_iib"))
    dialog.create()
    assert "expects Type IIb in channel 3" in dialog.error_label.text()
    assert dialog.created is None and not (tmp_path / "p").exists()


def test_new_project_form_clears_old_errors_and_explains_the_model_choice(qapp, tmp_path):
    from src.launcher import NO_MODEL, NewProjectDialog

    dialog = NewProjectDialog()
    dialog.create()
    assert dialog.error_label.text() == "Give the project a name."
    dialog.name_edit.setText("p")  # any change makes the old error out of date
    assert dialog.error_label.text() == ""

    dialog.images_edit.setText(str(_tif_folder(tmp_path)))
    dialog._images_changed()
    assert dialog.model_info.text() == "ta_three_class_logistic_v1 is made for these stains."
    dialog.model_combo.setCurrentIndex(dialog.model_combo.findData(NO_MODEL))
    assert dialog.model_info.text().startswith(
        "ta_three_class_logistic_v1 is made for these stains, but you chose to build your own"
    )
