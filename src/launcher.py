"""Project launcher: the workflow as a guided list of steps (experimental).

``python -m fibertypeqc app [PROJECT]`` opens one window per project. It shows only the steps on
this project's path, worked out from the files on disk (``src.workflow_steps``), with the next
step expanded and the later ones greyed out. Every action runs the same ``python -m fibertypeqc``
command as the terminal, in a separate process, so results are identical and a crash in a step
never takes the launcher down. The launcher never submits cluster jobs: on a cluster, start a
desktop session on a suitable node and run steps here, or copy a step's command into a job.
"""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

import pandas as pd
import yaml
from qtpy.QtCore import QProcess, QProcessEnvironment, Qt, QUrl
from qtpy.QtGui import QDesktopServices, QFontDatabase, QPalette, QTextCursor
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibertypeqc.model_resolution import released_models, resolve_model_argument
from fibertypeqc.panel_presets import MARKER_LABELS, MARKERS, list_presets, panel_from_channels
from src.cli import (
    CONFIG_NAME,
    REPO_ROOT,
    ProjectError,
    ProjectFolder,
    create_project,
    image_channel_count,
    image_files,
    load_project_folder,
    sample_rows,
)
from src.workflow_steps import (
    ATTENTION,
    CURRENT,
    DONE,
    SKIPPED,
    UPCOMING,
    Step,
    Workflow,
    can_edit_setup,
    channels_beyond,
    command_arguments,
    friendly_model_problem,
    plural,
    set_model,
    update_setup,
    workflow,
)

COPY_PREFIX = "uv run --frozen python -m fibertypeqc"
MARKS = {DONE: "✓", CURRENT: "●", UPCOMING: "○", ATTENTION: "!", SKIPPED: "–"}
MARK_COLORS = {DONE: "done", CURRENT: "current", ATTENTION: "warning"}
# (light theme, dark theme); Qt follows the system appearance, so pick by the window color.
COLORS = {
    "done": ("#2e7d32", "#81c784"),
    "current": ("#1565c0", "#64b5f6"),
    "warning": ("#e65100", "#ffb74d"),
    "error": ("#c62828", "#ef9a9a"),
}


def _dark() -> bool:
    return QApplication.palette().color(QPalette.Window).lightness() < 128


def color(name: str) -> str:
    return COLORS[name][_dark()]


def _mute(widget: QWidget) -> QWidget:
    """Show secondary text faded. Fading keeps the theme's own text color, which matters on
    macOS: in dark mode Qt can report the light palette while drawing dark, so a grey computed
    from the palette may be unreadable."""
    effect = QGraphicsOpacityEffect(widget)
    effect.setOpacity(0.6)
    widget.setGraphicsEffect(effect)
    return widget


ACTIONS = {
    "setup": "Choose model file…",
    "samples": "Edit samples",
    "run": "Run",
    "prepare": "Prepare",
    "review": "Open review",
    "finalize": "Finalize and build report",
    "label": "Open blind labelling",
    "build": "Build",
}
AGAIN = {
    "prepare": "Prepare again",
    "review": "Review more",
    "finalize": "Finalize again",
    "label": "Label more",
    "build": "Build again",
}
# Evaluate and promote get their own screens with confirmations (issue #84); until then the
# launcher shows the command to run in a terminal.
COPY_ONLY = {"evaluate", "promote"}
LAUNCHER_TEXT = {
    ("run", "new"): (
        "Finds every fiber and measures it (and calls its type when the project has a model). "
        "Runs on this computer and uses its GPU when there is one; it can take a while. On a "
        "cluster, open this window in a desktop session on a GPU node."
    ),
    ("review", "ready_for_review"): (
        "Check the flagged fibers in the review window. Decisions save as you go, so you can "
        "close it and come back. When you are done, finalize to build the results and report."
    ),
}
DEVICE_LINE = re.compile(r"Cellpose device:\s*(\w+)")


def copy_text(arguments: list[str]) -> str:
    return f"{COPY_PREFIX} {shlex.join(arguments)}"


# --- step list ------------------------------------------------------------------------------


class StepRow(QFrame):
    """One step: a mark, the title and summary, and, for the current step, its actions."""

    def __init__(self, window: LauncherWindow, step: Step) -> None:
        super().__init__()
        self.step = step
        self.action_button: QPushButton | None = None
        self.again_button: QPushButton | None = None
        self.copy_button: QPushButton | None = None
        self.finalize_button: QPushButton | None = None
        self.again_allowed = True
        self.reviewer_edit: QLineEdit | None = None
        self.setObjectName("currentStep" if step.state == CURRENT else "step")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)

        header = QHBoxLayout()
        mark = QLabel(MARKS[step.state])
        mark.setFixedWidth(18)
        if step.state in MARK_COLORS:
            mark.setStyleSheet(f"color: {color(MARK_COLORS[step.state])}; font-weight: bold;")
        title = QLabel(step.title)
        title.setStyleSheet("font-weight: bold;" if step.state == CURRENT else "")
        header.addWidget(mark)
        header.addWidget(title)
        if step.summary and step.state != CURRENT:
            summary = QLabel(step.summary)
            _mute(summary)
            summary.setWordWrap(True)
            header.addWidget(summary, 1)
        else:
            header.addStretch(1)
        if step.key == "samples" and step.state != CURRENT:
            self.again_button = QPushButton("Edit")
            self.again_button.clicked.connect(window.edit_samples)
        elif step.key == "setup" and step.fix != "panel":
            self.again_button = QPushButton("Edit")
            self.again_button.clicked.connect(window.edit_setup)
            self.again_allowed = window.project is not None and can_edit_setup(window.project)
            if not self.again_allowed:
                self.again_button.setToolTip(
                    "Segmentation has run with these settings. Create a new project to use "
                    "another panel or images."
                )
        elif step.state == DONE and step.key in AGAIN and window.can_repeat(step.key):
            self.again_button = QPushButton(AGAIN[step.key])
            self.again_button.clicked.connect(lambda: window.start_step(step))
        if self.again_button is not None:
            header.addWidget(self.again_button)
        layout.addLayout(header)

        if step.state == CURRENT:
            self._add_current(window, layout)
        if step.state == UPCOMING:
            for label in (title, mark):
                _mute(label)

    def _add_current(self, window: LauncherWindow, layout: QVBoxLayout) -> None:
        step = self.step
        body = QVBoxLayout()
        body.setContentsMargins(28, 0, 0, 4)
        if step.summary:
            body.addWidget(_wrapped(step.summary))
        stage = window.workflow.status["stage"] if window.workflow else ""
        # The launcher's own wording where the status text speaks of terminal commands.
        explanation = LAUNCHER_TEXT.get((step.key, stage), step.explanation)
        if explanation:
            body.addWidget(_wrapped(explanation))
        if step.key in COPY_ONLY:
            body.addWidget(
                _wrapped(
                    "The launcher does not run this step yet. Copy the command and run it in a "
                    "terminal from the FiberTypeQC folder, then press Refresh."
                )
            )
        actions = QHBoxLayout()
        if step.key == "label":
            self.reviewer_edit = QLineEdit(window.project.reviewer if window.project else "")
            self.reviewer_edit.setPlaceholderText("Your name")
            self.reviewer_edit.setMaximumWidth(180)
            actions.addWidget(QLabel("Reviewer:"))
            actions.addWidget(self.reviewer_edit)
        if step.key in ACTIONS and step.key not in COPY_ONLY:
            text = "Edit set-up…" if step.fix == "panel" else ACTIONS[step.key]
            self.action_button = QPushButton(text)
            self.action_button.setDefault(True)
            self.action_button.clicked.connect(lambda: window.start_step(step, self.reviewer()))
            actions.addWidget(self.action_button)
            if self.reviewer_edit is not None:
                self.reviewer_edit.textChanged.connect(lambda _text: self.set_busy(window.running))
        finalize = window.workflow.step("finalize") if window.workflow else None
        if step.key == "review" and finalize and "fiber_decisions" in window.workflow.status:
            # Status keeps pointing at review until finalize has run; offer it once there are
            # decisions to finalize.
            self.finalize_button = QPushButton(ACTIONS["finalize"])
            self.finalize_button.clicked.connect(lambda: window.start_step(finalize))
            actions.addWidget(self.finalize_button)
        if step.command:
            self.copy_button = QPushButton("Copy command")
            self.copy_button.setFlat(True)
            self.copy_button.clicked.connect(lambda: window.copy_command(step, self.reviewer()))
            actions.addWidget(self.copy_button)
        actions.addStretch(1)
        body.addLayout(actions)
        layout.addLayout(body)

    def reviewer(self) -> str:
        return self.reviewer_edit.text().strip() if self.reviewer_edit is not None else ""

    def set_busy(self, busy: bool) -> None:
        if self.again_button is not None:
            self.again_button.setEnabled(not busy and self.again_allowed)
        if self.finalize_button is not None:
            self.finalize_button.setEnabled(not busy)
        if self.action_button is not None:
            needs_name = self.reviewer_edit is not None and not self.reviewer()
            self.action_button.setEnabled(not busy and not needs_name)


def _wrapped(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


# --- main window ----------------------------------------------------------------------------


class LauncherWindow(QMainWindow):
    def __init__(self, project_root: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("FiberTypeQC")
        self.resize(720, 720)
        self.project: ProjectFolder | None = None
        self.workflow: Workflow | None = None
        self.rows: list[StepRow] = []
        self.device = ""
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.setWorkingDirectory(str(REPO_ROOT))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(environment)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)

        central = QWidget()
        outer = QVBoxLayout(central)
        top = QHBoxLayout()
        self.project_label = QLabel("No project open")
        self.project_label.setStyleSheet("font-size: 15px; font-weight: bold;")
        self.project_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.open_button = QPushButton("Open project…")
        self.open_button.clicked.connect(self.choose_project)
        self.new_button = QPushButton("New project…")
        self.new_button.clicked.connect(self.new_project)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        self.more_button = QToolButton()
        self.more_button.setText("More")
        self.more_button.setPopupMode(QToolButton.InstantPopup)
        self.more_menu = QMenu(self.more_button)
        self.more_button.setMenu(self.more_menu)
        top.addWidget(self.project_label, 1)
        for widget in (self.open_button, self.new_button, self.refresh_button, self.more_button):
            top.addWidget(widget)
        outer.addLayout(top)
        self.path_label = QLabel("")
        _mute(self.path_label)
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.path_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        outer.addWidget(self.path_label)

        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        self.banner.setVisible(False)
        self.banner.setStyleSheet(
            "padding: 6px; border-radius: 4px; background: palette(alternate-base);"
        )
        outer.addWidget(self.banner)
        self.report_button = QPushButton("Open report")
        self.report_button.clicked.connect(self.open_report)
        self.report_button.setVisible(False)
        outer.addWidget(self.report_button, 0, Qt.AlignLeft)

        self.steps_widget = QWidget()
        self.steps_layout = QVBoxLayout(self.steps_widget)
        self.steps_layout.setSpacing(2)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self.steps_widget)
        outer.addWidget(scroll, 3)

        log_header = QHBoxLayout()
        self.log_toggle = QToolButton()
        self.log_toggle.setText("Log")
        self.log_toggle.setCheckable(True)
        self.log_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.log_toggle.setArrowType(Qt.RightArrow)
        self.log_toggle.toggled.connect(self._show_log)
        self.activity_label = QLabel("")
        log_header.addWidget(self.log_toggle)
        log_header.addWidget(self.activity_label, 1)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self.stop)
        log_header.addWidget(self.stop_button)
        outer.addLayout(log_header)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.log.setMaximumBlockCount(5000)
        self.log.setVisible(False)
        self.log.setMaximumHeight(220)
        outer.addWidget(self.log, 2)
        self.setCentralWidget(central)
        self.setStyleSheet(
            "QFrame#currentStep { border: 1px solid palette(highlight); border-radius: 6px; }"
        )
        if project_root is not None:
            self.open_project(project_root)
        else:
            self.refresh()

    # --- state ------------------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.NotRunning

    def open_project(self, root: Path) -> bool:
        try:
            self.project = load_project_folder(root)
        except ProjectError as exc:
            QMessageBox.warning(self, "Not a project", str(exc))
            return False
        self.refresh()
        return True

    def refresh(self) -> None:
        self.rows = []
        while self.steps_layout.count():
            item = self.steps_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.banner.setVisible(False)
        self.report_button.setVisible(False)
        if self.project is None:
            self.workflow = None
            self.project_label.setText("No project open")
            self.path_label.setText("")
            self.steps_layout.addWidget(
                _wrapped(
                    "Open a project folder, or create a new one from a folder of images. "
                    "The steps for your project appear here, one at a time."
                )
            )
            self.steps_layout.addStretch(1)
            self._set_busy(self.running)
            return
        try:
            # Reload the config: promotion switches a project from model: none to a model.
            self.project = load_project_folder(self.project.root)
            self.workflow = workflow(self.project)
        except (ProjectError, OSError, ValueError) as exc:
            self.workflow = None
            self.project_label.setText(self.project.root.name)
            self.steps_layout.addWidget(_wrapped(f"This project cannot be read: {exc}"))
            self.steps_layout.addStretch(1)
            self._set_busy(self.running)
            return
        self.project_label.setText(self.project.root.name)
        self.path_label.setText(str(self.project.root))
        for note in self.workflow.notes:
            self._show_banner(f"Note: {note}")
        if self.workflow.finished:
            self._show_banner(
                "All steps are done. Open the report, or review more and finalize again."
            )
            self.report_button.setVisible(True)
        for step in self.workflow.steps:
            row = StepRow(self, step)
            self.rows.append(row)
            self.steps_layout.addWidget(row)
        self.steps_layout.addStretch(1)
        self._set_busy(self.running)

    def _show_banner(self, text: str) -> None:
        existing = self.banner.text() if self.banner.isVisible() else ""
        self.banner.setText(f"{existing}\n{text}".strip())
        self.banner.setVisible(True)

    def can_repeat(self, key: str) -> bool:
        if self.project is None:
            return False
        return (
            key != "finalize" or (self.project.final_dir / "finalization_manifest.json").is_file()
        )

    def _update_menu(self) -> None:
        self.more_menu.clear()
        project = self.project
        enabled = project is not None and not self.running
        # Rebuilt on every refresh and when a step starts or stops.
        folder = self.more_menu.addAction("Open project folder", self.open_folder)
        folder.setEnabled(project is not None)
        if project is not None and (project.root / "reference_snapshot.json").is_file():
            action = self.more_menu.addAction(
                "Check against the reference snapshot",
                lambda: self.start_arguments(["snapshot", str(project.root), "--check"]),
            )
            action.setEnabled(enabled and (project.batch_dir / "batch_summary.csv").is_file())
        if project is not None and not project.no_model and self.workflow is not None:
            action = self.more_menu.addAction(
                "Improve the model from review decisions",
                lambda: self.start_arguments(["improve", str(project.root)]),
            )
            action.setEnabled(enabled and project.review_project.is_file())
        if project is not None:
            action = self.more_menu.addAction("Segment and type again…", self.run_again)
            action.setEnabled(enabled and (project.batch_dir / "batch_summary.csv").is_file())
            self.more_menu.addAction(
                "Copy the status command",
                lambda: self._copy(copy_text(["status", str(project.root)])),
            )

    # --- actions ----------------------------------------------------------------------------

    def choose_project(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open a FiberTypeQC project folder")
        if folder:
            self.open_project(Path(folder))

    def new_project(self) -> None:
        dialog = NewProjectDialog(self)
        if dialog.exec() == QDialog.Accepted and dialog.created is not None:
            self.open_project(dialog.created)

    def edit_setup(self) -> None:
        if self.project is None:
            return
        dialog = NewProjectDialog(self, project=self.project)
        dialog.exec()
        self.refresh()

    def edit_samples(self) -> None:
        if self.project is None:
            return
        dialog = SamplesDialog(self.project, self)
        dialog.exec()
        self.refresh()

    def choose_model_file(self) -> None:
        if self.project is None:
            return
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Model file",
            str(self.project.root.parent),
            "Model files (*.joblib *.pkl *.pickle)",
        )
        if path:
            set_model(self.project, path)
            self.refresh()

    def open_folder(self) -> None:
        if self.project is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.project.root)))

    def open_report(self) -> None:
        if self.project is not None:
            report = self.project.results_dir / "cohort_report.html"
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(report)))

    def run_again(self) -> None:
        if self.project is None:
            return
        answer = QMessageBox.question(
            self,
            "Segment and type again?",
            "This runs segmentation and typing again on every image and replaces the batch "
            "outputs. Review decisions were made on the current outputs, so this is normally "
            "only needed after a failure or a change of images. Continue?",
        )
        if answer == QMessageBox.Yes:
            self.start_arguments(["run", str(self.project.root)])

    def start_step(self, step: Step, reviewer: str = "") -> None:
        if self.project is None:
            return
        if step.key == "samples":
            self.edit_samples()
            return
        if step.key == "setup":
            if step.fix == "panel":
                self.edit_setup()
            else:
                self.choose_model_file()
            return
        command = step.command or {"label": "review PROJECT --blind --reviewer NAME"}.get(
            step.key, f"{step.key} PROJECT"
        )
        if step.key == "label" and not reviewer:
            reviewer, ok = QInputDialog.getText(self, "Reviewer", "Your name:")
            if not ok or not reviewer.strip():
                return
        self.start_arguments(command_arguments(command, self.project, reviewer.strip()))

    def copy_command(self, step: Step, reviewer: str = "") -> None:
        if self.project is not None and step.command:
            arguments = command_arguments(step.command, self.project, reviewer or "NAME")
            self._copy(copy_text(arguments))

    def _copy(self, text: str) -> None:
        QApplication.clipboard().setText(text)
        self.activity_label.setText(f"Copied: {text}")

    def process_command(self, arguments: list[str]) -> tuple[str, list[str]]:
        """The program and arguments for a step (tests replace this with a stand-in)."""
        return sys.executable, ["-m", "fibertypeqc", *arguments]

    def start_arguments(self, arguments: list[str]) -> None:
        if self.running:
            return
        program, program_arguments = self.process_command(arguments)
        self.log.appendPlainText(f"\n$ {copy_text(arguments)}")
        self.log.appendPlainText("")  # the step's output starts on its own line
        self.log_toggle.setChecked(True)
        self.device = ""
        self.activity_label.setText(f"Running: {arguments[0]} …")
        self.process.start(program, program_arguments)
        self._set_busy(True)

    def stop(self) -> None:
        if self.running:
            self.process.kill()

    def _read_output(self) -> None:
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)
        match = DEVICE_LINE.search(text)
        if match:
            self.device = match.group(1)
            self.activity_label.setText(f"Running on {self.device} …")

    def _finished(self, code: int, _status=None) -> None:
        self._read_output()
        self.log.appendPlainText(f"[finished with exit code {code}]")
        message = (
            "Finished." if code == 0 else f"Stopped with an error (exit code {code}); see the log."
        )
        if self.device:
            message += f" Device: {self.device}."
        self.activity_label.setText(message)
        self._set_busy(False)
        self.refresh()

    def _process_error(self, error) -> None:
        if error == QProcess.FailedToStart:
            self.activity_label.setText("The step could not be started; see the log.")
            self.log.appendPlainText(f"[could not start: {self.process.errorString()}]")
            self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        self.stop_button.setVisible(busy)
        for widget in (self.open_button, self.new_button, self.refresh_button):
            widget.setEnabled(not busy)
        for row in self.rows:
            row.set_busy(busy)
        self._update_menu()

    def _show_log(self, shown: bool) -> None:
        self.log.setVisible(shown)
        self.log_toggle.setArrowType(Qt.DownArrow if shown else Qt.RightArrow)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if self.running:
            answer = QMessageBox.question(
                self,
                "A step is still running",
                "Stop it and close? Steps that are stopped part-way can be run again.",
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.process.kill()
            self.process.waitForFinished(3000)
        event.accept()


# --- new project ----------------------------------------------------------------------------

CUSTOM_PANEL = "custom"
NO_MODEL = "none"
MODEL_FILE = "file"


class NewProjectDialog(QDialog):
    """The questions ``init`` asks, as a form; it writes the project with the same code.

    With ``project``, it edits that project's images, panel, and model instead (only before
    segmentation has run; see ``update_setup``).
    """

    def __init__(self, parent: QWidget | None = None, project: ProjectFolder | None = None) -> None:
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Project set-up" if project else "New project")
        self.created: Path | None = None
        self.model_file = ""
        self.channel_count: int | None = None
        self._loading = False
        form = QFormLayout()
        # The macOS default keeps fields at their size hint, which clips wrapped help text.
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.parent_edit = QLineEdit(str(Path.home()))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. ta_cohort_2026")
        self.images_edit = QLineEdit()
        self.images_edit.editingFinished.connect(self._images_changed)
        if project is None:
            form.addRow("Save the project in", _with_browse(self.parent_edit, self._browse_parent))
            form.addRow("Project name", self.name_edit)
        form.addRow("Images folder", _with_browse(self.images_edit, self._browse_images))
        self.images_info = QLabel("")
        _mute(self.images_info)
        form.addRow("", self.images_info)

        self.panel_combo = QComboBox()
        for name, (_path, description) in list_presets().items():
            self.panel_combo.addItem(name, name)
            self.panel_combo.setItemData(self.panel_combo.count() - 1, description, Qt.ToolTipRole)
        self.panel_combo.addItem("Custom: I will choose the channel for each stain", CUSTOM_PANEL)
        self.panel_combo.currentIndexChanged.connect(self._panel_changed)
        form.addRow("Stain panel", self.panel_combo)
        self.panel_info = _wrapped("")
        _mute(self.panel_info)
        form.addRow("", self.panel_info)
        self.channel_box = QWidget()
        channel_form = QFormLayout(self.channel_box)
        channel_form.setContentsMargins(0, 0, 0, 0)
        self.channel_combos: dict[str, QComboBox] = {}
        for marker in MARKERS:
            combo = QComboBox()
            combo.currentIndexChanged.connect(lambda _index: self._suggest_model())
            self.channel_combos[marker] = combo
            channel_form.addRow(MARKER_LABELS[marker], combo)
        self._fill_channels(None)
        form.addRow("", self.channel_box)

        self.model_combo = QComboBox()
        self.models = dict(released_models())
        for model_id in self.models:
            self.model_combo.addItem(model_id, model_id)
        self.model_combo.addItem("None: build a model from my own labels", NO_MODEL)
        self.model_combo.addItem("A model file…", MODEL_FILE)
        self.model_combo.activated.connect(self._model_chosen)
        self.model_combo.currentIndexChanged.connect(lambda _index: self._check_model())
        form.addRow("Model", self.model_combo)
        self.model_info = _wrapped("")
        _mute(self.model_info)
        form.addRow("", self.model_info)
        self.model_warning = _wrapped("")
        self.model_warning.setStyleSheet(f"color: {color('warning')};")
        form.addRow("", self.model_warning)

        self.pattern_edit = QLineEdit()
        self.pattern_edit.setPlaceholderText(r"optional, e.g. ^(m\d+)_  (first group = mouse ID)")
        self.pattern_edit.textChanged.connect(self._preview_mice)
        self.mice_preview = QLabel("")
        _mute(self.mice_preview)
        if project is None:
            form.addRow("Mouse ID from file name", self.pattern_edit)
            form.addRow("", self.mice_preview)

        self.error_label = _wrapped("")
        self.error_label.setStyleSheet(f"color: {color('error')};")
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Save" if project else "Create project")
        buttons.accepted.connect(self.save if project else self.create)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.error_label)
        layout.addWidget(buttons)
        self._panel_changed()
        if project is not None:
            self._load(project)
        self.resize(620, 0)

    def _load(self, project: ProjectFolder) -> None:
        """Show the project's current images, panel, and model."""
        self._loading = True
        self.images_edit.setText(str(project.images))
        self._images_changed()
        channels = (yaml.safe_load(project.panel.read_text(encoding="utf-8")) or {}).get(
            "channels", {}
        )
        for name, (path, _description) in list_presets().items():
            preset = yaml.safe_load(path.read_text(encoding="utf-8")).get("channels", {})
            if preset == channels:
                self.panel_combo.setCurrentIndex(self.panel_combo.findData(name))
                break
        else:
            self.panel_combo.setCurrentIndex(self.panel_combo.findData(CUSTOM_PANEL))
            for marker, combo in self.channel_combos.items():
                index = combo.findData(channels.get(marker))
                combo.setCurrentIndex(max(index, 0))
        if project.no_model:
            self.model_combo.setCurrentIndex(self.model_combo.findData(NO_MODEL))
        elif project.model and project.model in self.models:
            self.model_combo.setCurrentIndex(self.model_combo.findData(project.model))
        elif project.model:
            self.model_file = project.model
            index = self.model_combo.findData(MODEL_FILE)
            self.model_combo.setItemText(index, f"File: {Path(project.model).name}")
            self.model_combo.setCurrentIndex(index)
        self._loading = False
        self._check_model()

    def _check_model(self) -> None:
        """Warn now, not at Run, when the chosen model is not on this computer."""
        if not hasattr(self, "model_warning"):
            return
        choice = self.model_combo.currentData()
        value = self.model_file if choice == MODEL_FILE else choice
        problem = ""
        if value and value != NO_MODEL:
            try:
                resolve_model_argument(value)
            except ValueError as exc:
                problem = friendly_model_problem(str(exc)).replace(
                    "choose it here", 'pick "A model file…" above and choose it'
                )
        self.model_warning.setText(problem)
        self.model_warning.setVisible(bool(problem))
        self._fit_text()

    def _browse_parent(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Save the project in", self.parent_edit.text()
        )
        if folder:
            self.parent_edit.setText(folder)

    def _browse_images(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Folder containing your images")
        if folder:
            self.images_edit.setText(folder)
            self._images_changed()

    def _images(self) -> list[Path]:
        text = self.images_edit.text().strip()
        return image_files(Path(text).expanduser().resolve()) if text else []

    def _images_changed(self) -> None:
        try:
            images = self._images()
        except ProjectError as exc:
            self.images_info.setText(str(exc))
            self._fill_channels(None)
            return
        channels = image_channel_count(images[0]) if images else None
        self.images_info.setText(
            f"{len(images)} image(s)"
            + (
                f"; {images[0].name} has {channels} channel(s), numbered 0 to {channels - 1}"
                if channels
                else ""
            )
        )
        self._fill_channels(channels)
        self.channel_count = channels
        if not self._loading and channels and self.panel_combo.currentData() != CUSTOM_PANEL:
            # Prefer a preset that fits the images over the first one in the list.
            name = self.panel_combo.currentData()
            preset = yaml.safe_load(list_presets()[name][0].read_text(encoding="utf-8"))
            if channels_beyond(preset, channels):
                for other, (path, _description) in list_presets().items():
                    mapping = yaml.safe_load(path.read_text(encoding="utf-8"))
                    if not channels_beyond(mapping, channels):
                        self.panel_combo.setCurrentIndex(self.panel_combo.findData(other))
                        break
        self._preview_mice()

    def _fill_channels(self, channels: int | None) -> None:
        for combo in self.channel_combos.values():
            chosen = combo.currentData()
            combo.clear()
            combo.addItem("not stained", None)
            for number in range(channels or 8):
                combo.addItem(f"channel {number}", number)
            if chosen is not None and combo.findData(chosen) >= 0:
                combo.setCurrentIndex(combo.findData(chosen))

    def _panel_changed(self) -> None:
        custom = self.panel_combo.currentData() == CUSTOM_PANEL
        self.channel_box.setVisible(custom)
        presets = list_presets()
        name = self.panel_combo.currentData()
        self.panel_info.setText(
            "Choose the channel for each stain in your images. A wrong choice is not reported "
            "as an error but gives wrong fiber types, so check each one."
            if custom
            else presets[name][1]
        )
        self._suggest_model()
        self._fit_text()

    def _fit_text(self) -> None:
        """Give wrapped help text the height it needs (a form layout does not ask for it)."""
        for label in (self.panel_info, self.model_info, self.model_warning, self.error_label):
            width = label.width() if label.isVisible() else self.width() - 200
            label.setMinimumHeight(label.heightForWidth(max(width, 300)) if label.text() else 0)
        self.adjustSize()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        for label in (self.panel_info, self.model_info, self.model_warning, self.error_label):
            if label.isVisible() and label.text():
                label.setMinimumHeight(label.heightForWidth(label.width()))

    def _suggest_model(self) -> None:
        """Select the released model made for exactly these stains, or none when there is none."""
        if not hasattr(self, "model_combo") or self._loading:
            return
        try:
            panel, _source = self.panel()
        except ProjectError:
            panel = {"channels": {}}
        model_id = matching_model(panel, self.models)
        self.model_combo.setCurrentIndex(self.model_combo.findData(model_id or NO_MODEL))
        self._check_model()  # also when the suggestion was already selected (no change signal)
        self.model_info.setText(
            f"{model_id} is made for these stains."
            if model_id
            else "No released model is made for these stains, so FiberTypeQC will help you "
            "build one: you label a sample of fibers, and it trains and checks a model for you."
        )

    def _model_chosen(self) -> None:
        if self.model_combo.currentData() != MODEL_FILE:
            return
        path, _filter = QFileDialog.getOpenFileName(
            self, "Model file", "", "Model files (*.joblib *.pkl *.pickle)"
        )
        if path:
            self.model_file = path
            self.model_combo.setItemText(
                self.model_combo.currentIndex(), f"File: {Path(path).name}"
            )
        else:
            self.model_combo.setCurrentIndex(0)

    def _preview_mice(self) -> None:
        try:
            images = self._images()
            rows = sample_rows(images[:3], self.pattern_edit.text().strip() or None)
        except ProjectError as exc:
            self.mice_preview.setText(str(exc))
            return
        if not rows or not self.pattern_edit.text().strip():
            self.mice_preview.setText(
                "Optional. Mouse IDs can also be typed in later, in the samples table."
            )
            return
        self.mice_preview.setText(
            "\n".join(f"{row['image_id']} → {row['mouse_id'] or '(no match)'}" for row in rows)
        )

    def checked_panel(self) -> tuple[dict, str]:
        """The panel, refused when it uses channels the images do not have."""
        panel, source = self.panel()
        beyond = channels_beyond(panel, self.channel_count)
        if beyond:
            raise ProjectError(
                f"This panel expects {', '.join(beyond)}, but the images have "
                f"{self.channel_count} channels (0 to {self.channel_count - 1}). Choose another "
                "panel, or Custom to set each channel."
            )
        return panel, source

    def panel(self) -> tuple[dict, str]:
        choice = self.panel_combo.currentData()
        if choice != CUSTOM_PANEL:
            path = list_presets()[choice][0]
            return yaml.safe_load(path.read_text(encoding="utf-8")), f"preset {choice}"
        channels = {marker: combo.currentData() for marker, combo in self.channel_combos.items()}
        try:
            return panel_from_channels(channels), "entered in the launcher"
        except ValueError as exc:
            raise ProjectError(str(exc)) from exc

    def model(self) -> str:
        choice = self.model_combo.currentData()
        if choice == MODEL_FILE:
            if not self.model_file:
                raise ProjectError("Choose a model file, or another model.")
            return self.model_file
        return choice

    def create(self) -> None:
        try:
            name = self.name_edit.text().strip()
            if not name:
                raise ProjectError("Give the project a name.")
            if not self.images_edit.text().strip():
                raise ProjectError("Choose the folder containing your images.")
            root = Path(self.parent_edit.text().strip()).expanduser().resolve() / name
            images_dir = Path(self.images_edit.text().strip()).expanduser().resolve()
            panel, source = self.checked_panel()
            create_project(
                root,
                images_dir,
                panel,
                source,
                self.model(),
                self.pattern_edit.text().strip() or None,
            )
        except ProjectError as exc:
            self.error_label.setText(str(exc))
            self._fit_text()
            return
        self.created = root
        self.accept()

    def save(self) -> None:
        try:
            if not self.images_edit.text().strip():
                raise ProjectError("Choose the folder containing your images.")
            images_dir = Path(self.images_edit.text().strip()).expanduser().resolve()
            panel, source = self.checked_panel()
            update_setup(self.project, images_dir, panel, source, self.model())
        except ProjectError as exc:
            self.error_label.setText(str(exc))
            return
        self.accept()


TYPING_STAINS = ("type_i", "type_iia", "type_iib", "type_iix")


def matching_model(panel: dict, models: dict[str, tuple[str, ...]]) -> str | None:
    """The first released model whose fiber-type stains are exactly the panel's, if any.

    A model is never suggested for a panel with more or fewer type stains: a three-class model
    on a panel with Type I would call Type I fibers IIx.
    """
    channels = panel.get("channels", {}) or {}
    if channels.get("laminin") is None:
        return None
    stains = {name for name in TYPING_STAINS if channels.get(name) is not None}
    for model_id, required in models.items():
        if set(required) & set(TYPING_STAINS) == stains and "laminin" in required:
            return model_id
    return None


def _with_browse(edit: QLineEdit, browse) -> QWidget:
    box = QWidget()
    layout = QHBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(edit, 1)
    button = QPushButton("Browse…")
    button.clicked.connect(browse)
    layout.addWidget(button)
    return box


# --- samples --------------------------------------------------------------------------------

LOCKED_COLUMNS = ("image_id", "raw_image_path")


class SamplesDialog(QDialog):
    """Edit samples.csv: mouse IDs and group columns (image columns stay as written by init)."""

    def __init__(self, project: ProjectFolder, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.project = project
        self.setWindowTitle("Samples")
        self.sheet = pd.read_csv(project.sample_sheet, dtype=str, keep_default_na=False)
        layout = QVBoxLayout(self)
        layout.addWidget(
            _wrapped(
                "Give each image its mouse ID. Add columns such as genotype or age to define the "
                "groups compared in the report."
            )
        )
        if project.review_project.is_file():
            warning = _wrapped(
                "Prepare has already used this sheet. Changes here are saved but do not reach "
                "the existing review project."
            )
            warning.setStyleSheet(f"color: {color('warning')};")
            layout.addWidget(warning)
        self.table = QTableWidget()
        self._fill()
        self.table.itemChanged.connect(lambda _item: self._validate())
        layout.addWidget(self.table, 1)
        self.message = QLabel("")
        layout.addWidget(self.message)
        row = QHBoxLayout()
        add_column = QPushButton("Add column…")
        add_column.clicked.connect(self.add_column)
        row.addWidget(add_column)
        row.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        row.addWidget(buttons)
        layout.addLayout(row)
        self.resize(760, 480)
        self._validate()

    def _fill(self) -> None:
        self.table.blockSignals(True)
        columns = list(self.sheet.columns)
        self.table.setColumnCount(len(columns))
        self.table.setRowCount(len(self.sheet))
        self.table.setHorizontalHeaderLabels(columns)
        for r, values in enumerate(self.sheet.itertuples(index=False)):
            for c, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if columns[c] in LOCKED_COLUMNS:
                    # Disabled items are drawn in the style's own disabled color.
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable & ~Qt.ItemIsEnabled)
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)

    def _read(self) -> pd.DataFrame:
        columns = [
            self.table.horizontalHeaderItem(c).text() for c in range(self.table.columnCount())
        ]
        rows = [
            [
                (self.table.item(r, c).text() if self.table.item(r, c) else "").strip()
                for c in range(self.table.columnCount())
            ]
            for r in range(self.table.rowCount())
        ]
        return pd.DataFrame(rows, columns=columns)

    def _validate(self) -> bool:
        sheet = self._read()
        blank = int(sheet["mouse_id"].eq("").sum()) if "mouse_id" in sheet else len(sheet)
        if blank:
            self.message.setText(
                f"{plural(blank, 'image still needs', 'images still need')} a mouse ID."
            )
            self.message.setStyleSheet(f"color: {color('warning')};")
        else:
            mice = sheet["mouse_id"].nunique()
            self.message.setText(
                f"{plural(len(sheet), 'image', 'images')} from {plural(mice, 'mouse', 'mice')}."
            )
            self.message.setStyleSheet("")
        return not blank

    def add_column(self) -> None:
        name, ok = QInputDialog.getText(self, "Add column", "Column name (e.g. genotype):")
        name = name.strip()
        if not ok or not name:
            return
        self.sheet = self._read()
        if name in self.sheet.columns:
            return
        self.sheet[name] = ""
        self._fill()

    def save(self) -> None:
        self._read().to_csv(self.project.sample_sheet, index=False)
        self.accept()


# --- entry point ----------------------------------------------------------------------------


def main(project: Path | None = None) -> int:
    application = QApplication.instance() or QApplication(sys.argv[:1])
    if project is not None and not (project.expanduser() / CONFIG_NAME).is_file():
        print(f"Error: {project} is not a FiberTypeQC project folder.", file=sys.stderr)
        return 2
    window = LauncherWindow(project)
    window.show()
    return application.exec()
