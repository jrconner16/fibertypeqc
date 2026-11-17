"""Focused, reviewer-oriented workspace for Phase 4.5."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256

import pandas as pd
from qtpy.QtCore import QSettings, Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QShortcut,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.review.call_overlay import legend_html, saved_call_text
from src.review.fiber_type_review import FiberTypeReviewController
from src.review.project import Project
from src.review.queues import (
    QueueSource,
    RandomAuditScope,
    build_fiber_type_queue,
    describe_reason,
    load_fiber_type_rows,
)
from src.review.roles import POOL, TEST, allowed_images, held_out_mice_released
from src.review.schemas import Domain, ObjectReviewStatus, Scope
from src.review.section_stage_widget import left_out_areas, section_verdict
from src.review.timing import ReviewTimer

CLICKED_QUEUE = "clicked"
PLAN_NAMES = {
    QueueSource.FLAGGED.value: "flagged fibers",
    QueueSource.FULL_AUDIT.value: "every fiber",
    QueueSource.RANDOM_AUDIT.value: "random sample",
}


class GuidedReviewWidget(QWidget):
    """Staged reviewer: Section (verdict, areas), Fibers (queue or click), Finish (finalize).

    One stage is shown at a time; everything else (cohort dashboard, per-domain section status,
    named regions and ROIs, nuclei) is under the Advanced menu.
    """

    def __init__(
        self,
        project: Project,
        controller: FiberTypeReviewController,
        *,
        object_changed: Callable[[str, int], None] | None = None,
        show_dashboard: Callable[[], None] | None = None,
        show_section: Callable[[], None] | None = None,
        show_region: Callable[[], None] | None = None,
        fiber_qc: pd.DataFrame | None = None,
        finalize_requested: Callable[[], None] | None = None,
        show_domain: Callable[[Domain], None] | None = None,
        focus_current_object: Callable[[], None] | None = None,
        blind: bool = False,
        reference_field_fibers: Callable[[str, bool | str], set[int]] | None = None,
        show_image: Callable[[str], None] | None = None,
        section_stage: QWidget | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.controller = controller
        self.section_stage = section_stage
        self._return_to: tuple[tuple, str, int] | None = None  # the plan to resume after a click
        # False while showing a clicked fiber: it is already on screen, so the view stays put.
        self.center_on_change = True
        # Blind reference labelling: nothing derived from the model is shown or offered.
        self.blind = blind
        # (image_id, capture drawn shapes?) -> IDs of fibers inside the reviewer's drawn fields.
        self.reference_field_fibers = reference_field_fibers
        self.show_image = show_image
        self.object_changed = object_changed
        self.show_dashboard = show_dashboard
        self.show_section = show_section
        self.show_region = show_region
        self.show_domain = show_domain
        self.focus_current_object = focus_current_object
        self.rows = load_fiber_type_rows(project, fiber_qc)
        # Evaluation roles: blind labels on test mice, guided review on pool mice, so test mice
        # are never shown with model calls while their blind labels are still needed. A project
        # without a model is labelled blind throughout (pool mice train the built model, test
        # mice evaluate it later), so every mouse is offered; `build` keeps test mice out. Once a
        # built model is promoted with its independent test recorded, guided review offers every
        # mouse so the whole cohort is reviewed; `improve` still trains on pool mice only.
        self.allowed_images = (
            None
            if project.model_version == "none" or (not blind and held_out_mice_released(project))
            else allowed_images(project, TEST if blind else POOL)
        )
        if self.allowed_images is not None:
            self.rows = self.rows[self.rows["image_id"].isin(self.allowed_images)]
        # Blind mode never shows section or mouse names (they can carry genotype, age, or
        # treatment). Sections get neutral numbers in an order unrelated to their names.
        blind_images = sorted(
            self.allowed_images
            if self.allowed_images is not None
            else {image.image_id for image in project.images},
            key=lambda image_id: sha256(f"{project.project_id}:{image_id}".encode()).hexdigest(),
        )
        self.section_codes = {image_id: n for n, image_id in enumerate(blind_images, start=1)}
        # Section order for the section bar: neutral numbers in blind mode, project order otherwise.
        self.section_order = (
            sorted(self.section_codes, key=self.section_codes.get)
            if blind
            else [
                image.image_id
                for image in project.images
                if self.allowed_images is None or image.image_id in self.allowed_images
            ]
        )
        # Only offer the classes this project's model produces (class names only; no calls). A
        # project without a model offers the classes named in its configuration instead.
        self.classes = set(project.review_classes) or (
            set(self.rows["model_fiber_type"].astype(str).str.lower())
            & {"i", "iia", "iib", "iix"}
            or {"i", "iia", "iib", "iix"}
        )
        # (shortcut key, button text, class); eMHC, hybrid, and negative only appear when
        # configured.
        self.class_buttons = tuple(
            (key, name, value)
            for key, name, value in (
                ("1", "I", "i"),
                ("2", "IIa", "iia"),
                ("3", "IIb", "iib"),
                ("4", "IIx", "iix"),
                # eMHC is never offered with myosin classes, so it can share key 1.
                ("1", project.class_labels.get("emhc", "eMHC⁺"), "emhc"),
                ("5", "Hybrid", "hybrid"),
                ("6", project.class_labels.get("negative", "Negative"), "negative"),
            )
            if value in self.classes
        )
        self.timer = ReviewTimer(
            project.review_directory,
            reviewer=controller.session.reviewer,
            mode="blind_reference" if blind else "guided_review",
        )
        active = self.controller.session.active_queue
        self._review_started = bool(active) and active != CLICKED_QUEUE
        self._settings = QSettings("FiberTypeQC", "FiberTypeQC")
        self.setMinimumWidth(410)

        self.context = QLabel()
        self.context.setWordWrap(True)
        self.blind_banner = QLabel(
            "BLIND reference labelling: model calls are hidden. Labels are saved separately "
            "for this reviewer and are not applied to results."
        )
        self.blind_banner.setWordWrap(True)
        self.blind_banner.setVisible(blind)
        # Section bar: which section is on screen, and moving between sections.
        self.section_label = QLabel()
        self.section_label.setWordWrap(True)
        self.previous_section_button = QPushButton("◀")
        self.previous_section_button.setToolTip("Previous section")
        self.next_section_bar_button = QPushButton("▶")
        self.next_section_bar_button.setToolTip("Next section")
        for button in (self.previous_section_button, self.next_section_bar_button):
            button.setMaximumWidth(40)
            button.setEnabled(show_image is not None)
        self.previous_section_button.clicked.connect(lambda: self._go_section(-1))
        self.next_section_bar_button.clicked.connect(lambda: self._go_section(1))
        # Everything beyond the three stages lives here.
        self.advanced_menu = QMenu(self)
        if not blind:  # these panels name sections or mice, or show model results
            self.advanced_menu.addAction("Cohort QC dashboard", self._show_dashboard)
            self.advanced_menu.addAction("Section status by domain", self._show_section)
            self.advanced_menu.addAction(
                "Segmentation status", lambda: self._show_domain(Domain.FIBER_SEGMENTATION)
            )
            self.advanced_menu.addAction("Named regions and analysis ROIs", self._show_region)
            self.advanced_menu.addAction("Nuclei review", lambda: self._show_domain(Domain.NUCLEI))
            self.advanced_menu.addSeparator()
        self.advanced_menu.addAction("Show the welcome guide", self._show_tutorial)
        self.advanced_button = QToolButton()
        self.advanced_button.setText("Advanced ▾")
        self.advanced_button.setPopupMode(QToolButton.InstantPopup)
        self.advanced_button.setMenu(self.advanced_menu)
        section_bar = QHBoxLayout()
        section_bar.addWidget(self.previous_section_button)
        section_bar.addWidget(self.section_label, 1)
        section_bar.addWidget(self.next_section_bar_button)
        section_bar.addWidget(self.advanced_button)
        self.section_bar = section_bar
        self.tutorial_group = QGroupBox(
            "Welcome to blind labelling" if blind else "Welcome to guided review"
        )
        tutorial_layout = QVBoxLayout(self.tutorial_group)
        tutorial = QLabel(
            "1. Choose a labelling plan.\n"
            "2. Label each fiber from the stain alone; the model's calls are hidden.\n"
            "3. Labels save immediately; Undo restores the last one."
            if blind
            else "1. Section: set each channel's background (lower contrast limit, top left), "
            "then give a verdict: looks fine, leave out an area, or leave out the section.\n"
            "2. Fibers: work through the flagged fibers, or click any fiber that looks wrong, "
            "and keep or correct the model call.\n"
            "3. Finish: finalize and open the report. Everything saves as you go."
        )
        tutorial.setWordWrap(True)
        tutorial_layout.addWidget(tutorial)
        dismiss_tutorial = QPushButton("Got it")
        dismiss_tutorial.clicked.connect(self._dismiss_tutorial)
        tutorial_layout.addWidget(dismiss_tutorial)
        self.plan_message = QLabel()
        self.plan_message.setWordWrap(True)
        self.start_flagged_button = QPushButton("Review flagged fibers")
        self.review_section_button = QPushButton("Review this section")
        self.cohort_qc_button = QPushButton("View cohort QC")
        self.change_plan_button = QPushButton("Change review plan")
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.save_status = QLabel()
        self.save_status.setWordWrap(True)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.undo_button = QPushButton("Undo last decision")
        self.undo_button.setEnabled(False)
        class_keys = " · ".join(f"{key} {name}" for key, name, _ in self.class_buttons)
        self.shortcut_label = QLabel(
            "Shortcuts: ←/→ navigate · F center fiber · "
            + ("" if blind else "K keep model · ")
            + f"{class_keys} · X exclude · U undo"
        )
        self.shortcut_label.setWordWrap(True)
        self.legend_label = QLabel(
            legend_html(sorted(set(self.rows["model_fiber_type"].astype(str).str.lower())))
        )
        self.legend_label.setTextFormat(Qt.RichText)
        self.legend_label.setWordWrap(True)
        self.legend_label.setVisible(not blind)

        self.plan_group = QGroupBox("Labelling plan" if blind else "What needs attention?")
        plan_layout = QVBoxLayout(self.plan_group)
        plan_layout.addWidget(self.plan_message)
        plan_layout.addWidget(self.start_flagged_button)
        self.start_flagged_button.setVisible(not blind)
        self.random_sample_button = QPushButton("Label a random sample")
        self.random_sample_button.setToolTip(
            "One list of computer-drawn fibers that goes through every section in turn. Labels "
            "save as you go, and clicking again resumes at the first fiber you have not "
            "labelled. Change the number, seed, and scope under Advanced review options."
        )
        self.random_sample_button.setVisible(blind)
        self.random_sample_button.clicked.connect(self.start_random_sample)
        plan_layout.addWidget(self.random_sample_button)
        self.auto_field_button = QPushButton("Label a computer-placed field (about 100 fibers)")
        self.auto_field_button.setToolTip(
            "The computer picks where the field goes on this section, from the seed under "
            "Advanced review options, so the choice is not yours. Every fiber in it is queued."
        )
        self.auto_field_button.setVisible(blind and reference_field_fibers is not None)
        self.auto_field_button.clicked.connect(self.start_auto_field_review)
        plan_layout.addWidget(self.auto_field_button)
        self.next_section_button = QPushButton("Go to next section")
        self.next_section_button.setToolTip(
            "Open the next section (shown by number only). This only changes the view: click a "
            "field button to label fibers on it."
        )
        self.next_section_button.setVisible(False)  # the section bar's ▶ does this now
        self.next_section_button.clicked.connect(self.go_to_next_section)
        plan_layout.addWidget(self.next_section_button)
        self.field_button = QPushButton("Label every fiber in my drawn fields")
        self.field_button.setToolTip(
            "Draw one or more shapes in the orange 'review_analysis_rois' layer on this section, "
            "then click here. Every fiber whose center is inside a shape is queued, and the "
            "shapes are saved as reference fields."
        )
        self.field_button.setVisible(blind and reference_field_fibers is not None)
        self.field_button.clicked.connect(self.start_field_review)
        plan_layout.addWidget(self.field_button)
        plan_layout.addWidget(self.review_section_button)
        self.cohort_qc_button.setVisible(False)  # Advanced ▸ Cohort QC dashboard
        self.click_hint = QLabel("Or click any fiber in the image to check it.")
        self.click_hint.setWordWrap(True)
        self.click_hint.setVisible(not blind)
        self.finalize_button = QPushButton("Finalize and build report")
        self.finalize_button.setToolTip(
            "Apply your decisions and exclusions, write the result tables, and open the report. "
            "Predictions are not modified; you can keep reviewing and finalize again."
        )
        self.finalize_button.setVisible(finalize_requested is not None)
        if finalize_requested is not None:
            self.finalize_button.clicked.connect(lambda _checked=False: finalize_requested())

        self.decision_group = QGroupBox("Current fiber")
        decision_layout = QVBoxLayout(self.decision_group)
        decision_layout.addWidget(self.details)
        decision_layout.addWidget(self.legend_label)
        # Two rows so nothing is cut off in a narrow panel: the classes, then keep and exclude.
        classes_row = QHBoxLayout()
        for key, name, fiber_type in self.class_buttons:
            button = QPushButton(f"{name} ({key})")
            button.clicked.connect(
                lambda _checked=False, value=fiber_type: self._record_type(value)
            )
            classes_row.addWidget(button)
        primary = QHBoxLayout()
        keep = QPushButton("Keep model call (K)")
        keep.setVisible(not blind)
        primary.addWidget(keep)
        exclude = QPushButton("Exclude (X)")
        exclude.setToolTip("Leave this fiber out of the results (damaged, cut off, not a fiber).")
        exclude.clicked.connect(lambda: self._record_special(ObjectReviewStatus.EXCLUDED))
        primary.addWidget(exclude)
        decision_layout.addLayout(classes_row)
        decision_layout.addLayout(primary)
        self.focus_button = QPushButton("Center current fiber (F)")
        self.focus_button.clicked.connect(self._focus_current_object)
        decision_layout.addWidget(self.focus_button)
        decision_layout.addWidget(self.undo_button)

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("Advanced review options")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setArrowType(Qt.RightArrow)
        # Without this the tool button shows only its arrow, with no label.
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_group = QGroupBox()
        advanced_layout = QVBoxLayout(self.advanced_group)
        advanced_actions = QHBoxLayout()
        for label, decision_status in (
            ("Uncertain", ObjectReviewStatus.UNCERTAIN),
            ("Unresolved", ObjectReviewStatus.UNRESOLVED),
        ):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, value=decision_status: self._record_special(value)
            )
            advanced_actions.addWidget(button)
        advanced_layout.addLayout(advanced_actions)
        self.queue_combo = QComboBox()
        self.queue_combo.addItems(
            [
                source.value
                for source in QueueSource
                # Every other queue is ordered by model confidence or QC flags.
                if not blind or source in (QueueSource.RANDOM_AUDIT, QueueSource.FULL_AUDIT)
            ]
        )
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 10_000)
        self.sample_spin.setValue(25)
        self.scope_combo = QComboBox()
        self.scope_combo.addItems([scope.value for scope in RandomAuditScope])
        advanced_layout.addWidget(QLabel("Queue"))
        advanced_layout.addWidget(self.queue_combo)
        advanced_layout.addWidget(QLabel("Random-audit seed / sample size / scope"))
        random_controls = QHBoxLayout()
        random_controls.addWidget(self.seed_spin)
        random_controls.addWidget(self.sample_spin)
        random_controls.addWidget(self.scope_combo)
        advanced_layout.addLayout(random_controls)
        self.advanced_group.setVisible(False)

        navigation = QHBoxLayout()
        previous = QPushButton("Previous (←)")
        next_button = QPushButton("Next (→)")
        navigation.addWidget(previous)
        navigation.addWidget(next_button)

        # Stage 2: fibers (also the whole panel in blind labelling).
        self.fibers_page = QWidget()
        fibers = QVBoxLayout(self.fibers_page)
        fibers.addWidget(self.context)
        fibers.addWidget(self.plan_group)
        fibers.addWidget(self.change_plan_button)
        fibers.addWidget(self.click_hint)
        fibers.addWidget(self.decision_group)
        fibers.addLayout(navigation)
        fibers.addWidget(self.save_status)
        fibers.addWidget(self.status)
        fibers.addWidget(self.advanced_toggle)
        fibers.addWidget(self.advanced_group)
        fibers.addStretch(1)
        fibers.addWidget(self.shortcut_label)
        # Stage 3: finish.
        self.finish_page = QWidget()
        finish = QVBoxLayout(self.finish_page)
        self.finish_summary = QLabel()
        self.finish_summary.setWordWrap(True)
        finish.addWidget(self.finish_summary)
        finish.addStretch(1)
        finish.addWidget(self.finalize_button)

        self.stages = QTabWidget()
        self.stages.setDocumentMode(True)
        self.stages.setTabBarAutoHide(True)  # blind labelling has a single stage
        if section_stage is not None and not blind:
            self.stages.addTab(section_stage, "1  Section")
        self.stages.addTab(self.fibers_page, "Fibers" if blind else "2  Fibers")
        if finalize_requested is not None and not blind:
            self.stages.addTab(self.finish_page, "3  Finish")
        self.stages.currentChanged.connect(lambda _index: self._stage_changed())

        layout = QVBoxLayout(self)
        layout.addWidget(self.blind_banner)
        layout.addLayout(section_bar)
        layout.addWidget(self.tutorial_group)
        layout.addWidget(self.stages, 1)

        self.start_flagged_button.clicked.connect(self.start_flagged_review)
        self.review_section_button.clicked.connect(self.start_section_review)
        self.cohort_qc_button.clicked.connect(self._show_dashboard)
        self.change_plan_button.clicked.connect(self._change_plan)
        keep.clicked.connect(self._keep_model_call)
        self.undo_button.clicked.connect(self.undo)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        self.queue_combo.currentTextChanged.connect(self._queue_changed)
        previous.clicked.connect(lambda: self._move(-1))
        next_button.clicked.connect(lambda: self._move(1))
        self.shortcuts: list[QShortcut] = []
        self._add_shortcut("Left", lambda: self._move(-1))
        self._add_shortcut("Right", lambda: self._move(1))
        self._add_shortcut("F", self._focus_current_object)
        if not blind:
            self._add_shortcut("K", self._keep_model_call)
        for key, _, fiber_type in self.class_buttons:
            self._add_shortcut(key, lambda value=fiber_type: self._record_type(value))
        self._add_shortcut("X", lambda: self._record_special(ObjectReviewStatus.EXCLUDED))
        self._add_shortcut("U", self.undo)
        if blind:
            # Reference sampling: the same number of computer-drawn fibers from every section.
            self.sample_spin.setValue(150)
            self.scope_combo.setCurrentText(RandomAuditScope.IMAGE.value)
        self._load_sample_settings()
        self.sample_spin.valueChanged.connect(self._update_sample_button)
        self.scope_combo.currentTextChanged.connect(self._update_sample_button)
        self._update_sample_button()
        self._restore_saved_queue()
        self.tutorial_group.setVisible(not self._tutorial_seen())
        # A resumed plan opens on Fibers; a fresh review starts with the section.
        if self._review_started:
            self.stages.setCurrentWidget(self.fibers_page)
        self.refresh()
        self.section_changed()

    # --- stages and sections -------------------------------------------------------------

    def _show_fibers(self) -> None:
        self.stages.setCurrentWidget(self.fibers_page)

    def _stage_changed(self) -> None:
        if self.stages.currentWidget() is self.finish_page:
            self.refresh_finish()

    def section_changed(self) -> None:
        """Update the section bar and summaries after the section on screen changes."""
        image_id = self.controller.session.current_image_id
        total = len(self.section_order)
        if image_id in self.section_order:
            number = self.section_order.index(image_id) + 1
            if self.blind:
                self.section_label.setText(f"Section {number} of {total}")
            else:
                image = self.project.image(image_id)
                self.section_label.setText(
                    f"Section {number} of {total} · {image.mouse_id} · {image_id}"
                )
        else:
            self.section_label.setText(f"{total} sections")
        refresh = getattr(self.section_stage, "refresh", None)
        if refresh is not None:
            refresh()
        if self.stages.currentWidget() is self.finish_page:
            self.refresh_finish()

    def _go_section(self, offset: int) -> int | None:
        """Show the next or previous section; returns its position, or None."""
        order = self.section_order
        if not order or self.show_image is None:
            return None
        current = self.controller.session.current_image_id
        index = (order.index(current) + offset) % len(order) if current in order else 0
        self.controller.session.current_image_id = order[index]
        self.show_image(order[index])
        self.section_changed()
        return index

    def refresh_finish(self) -> None:
        session = self.controller.session
        verdicts = [
            section_verdict(self.project, session, image_id) for image_id in self.section_order
        ]
        areas = sum(len(left_out_areas(session, image_id)) for image_id in self.section_order)
        decisions = session.object_decisions
        changed = sum(
            1
            for decision in decisions
            if decision.review_status
            in (ObjectReviewStatus.CORRECTED, ObjectReviewStatus.EXCLUDED)
        )
        self.finish_summary.setText(
            f"Sections: {verdicts.count('fine')} look fine, "
            f"{verdicts.count('left_out')} left out, "
            f"{verdicts.count('not_checked')} not checked yet "
            f"(of {len(verdicts)}).\n"
            f"Areas left out: {areas}.\n"
            f"Fibers checked: {len(decisions)} ({changed} changed or excluded).\n\n"
            "Finalizing applies these decisions, writes the result tables, and opens the "
            "report. Model calls are never overwritten, and you can keep reviewing and finalize "
            "again."
        )

    def select_fiber(self, image_id: str, fiber_id: int) -> bool:
        """Check one clicked fiber, then return to the plan that was under way.

        Not offered in blind labelling: hand-picked fibers would bias the reference labels.
        """
        if self.blind:
            return False
        if self.allowed_images is not None and image_id not in self.allowed_images:
            self.status.setText("This section is not part of guided review in this project.")
            return False
        rows = self.rows[self.rows["image_id"].eq(image_id) & self.rows["fiber_id"].eq(fiber_id)]
        if rows.empty:
            return False
        item = replace(
            build_fiber_type_queue(rows, QueueSource.FULL_AUDIT)[0], reason_code=CLICKED_QUEUE
        )
        if self.controller.session.active_queue != CLICKED_QUEUE:
            self._return_to = (
                self.controller.queue,
                self.controller.session.active_queue,
                self.controller.session.queue_position,
            )
        self._review_started = True
        self.controller.set_queue((item,), CLICKED_QUEUE)
        self._show_fibers()
        self.status.setText(f"Fiber {fiber_id} (clicked): decide, or → to return to the plan.")
        self.center_on_change = False
        try:
            self.refresh(notify=True)
        finally:
            self.center_on_change = True
        return True

    def _resume_after_click(self) -> bool:
        if self.controller.session.active_queue != CLICKED_QUEUE:
            return False
        if self._return_to is None or not self._return_to[1]:
            # No plan was under way: back to choosing one.
            self._return_to = None
            self._review_started = False
            self.controller.set_queue((), "")
            self._restore_saved_queue()
            self.refresh()
            return True
        queue, name, position = self._return_to
        self._return_to = None
        self.controller.set_queue(queue, name, position=position)
        self.refresh(notify=True)
        return True

    def start_flagged_review(self) -> None:
        self._review_started = True
        self._set_queue(QueueSource.FLAGGED)
        self._show_fibers()
        self.status.setText("Started flagged-fiber review.")

    def _add_shortcut(self, sequence: str, callback: Callable[[], None]) -> None:
        shortcut = QShortcut(QKeySequence(sequence), self, activated=callback)
        # Work wherever keyboard focus is (the image, another panel, or this dock floating as
        # its own window). Text boxes still receive the keys typed into them.
        shortcut.setContext(Qt.ApplicationShortcut)
        self.shortcuts.append(shortcut)

    def _sample_scope_text(self) -> str:
        return {
            RandomAuditScope.IMAGE.value: "from every section",
            RandomAuditScope.MOUSE.value: "from every mouse",
            RandomAuditScope.COHORT.value: "across the project",
        }.get(self.scope_combo.currentText(), "")

    def _update_sample_button(self, *_args) -> None:
        self.random_sample_button.setText(
            f"Label {self.sample_spin.value()} random fibers {self._sample_scope_text()}".strip()
        )

    def start_random_sample(self) -> None:
        self._review_started = True
        self._set_queue(QueueSource.RANDOM_AUDIT)
        self._show_fibers()
        count = len(self.controller.queue)
        done = self._labelled_in_queue()
        message = (
            f"Random sample: {count} fibers ({self.sample_spin.value()} "
            f"{self._sample_scope_text()}), {done} already labelled."
        )
        if done:
            message += " Resuming at the first fiber you have not labelled."
        self.status.setText(message)

    def _labelled_in_queue(self) -> int:
        decided = {(d.image_id, d.fiber_id) for d in self.controller.session.object_decisions}
        return sum((item.image_id, item.fiber_id) in decided for item in self.controller.queue)

    def go_to_next_section(self) -> None:
        """Open the next section (in the neutral order in blind labelling)."""
        index = self._go_section(1)
        if index is None:
            return
        self.status.setText(
            f"Showing section {index + 1} of {len(self.section_order)}. "
            + (
                "Click a field button to label fibers on it; the random sample keeps its own "
                "order."
                if self.blind
                else "Give it a verdict, or check its fibers."
            )
        )

    def start_auto_field_review(self) -> None:
        if not self._section_allowed():
            return
        image_id = self.controller.session.current_image_id
        seed = f"{self.project.project_id}:{image_id}:{self.seed_spin.value()}"
        queue = self._field_queue(capture=seed)
        if not queue:
            self.status.setText("This section has no fibers to place a field on.")
            return
        self._review_started = True
        self.controller.set_queue(queue, "reference_fields", position=self._first_undecided(queue))
        self._show_fibers()
        self.status.setText(f"Started labelling {len(queue)} fibers in a computer-placed field.")
        self.refresh(notify=True)

    def _first_undecided(self, queue: tuple) -> int:
        """Start a plan at its first fiber without a decision, so starting again resumes."""
        decided = {(d.image_id, d.fiber_id) for d in self.controller.session.object_decisions}
        for index, item in enumerate(queue):
            if (item.image_id, item.fiber_id) not in decided:
                return index
        return 0

    def _field_queue(self, capture: bool | str) -> tuple:
        image_id = self.controller.session.current_image_id
        if not image_id or self.reference_field_fibers is None:
            return ()
        inside = self.reference_field_fibers(image_id, capture)
        rows = self.rows[self.rows["image_id"].eq(image_id) & self.rows["fiber_id"].isin(inside)]
        return build_fiber_type_queue(rows, QueueSource.FULL_AUDIT)

    def start_field_review(self) -> None:
        if not self._section_allowed():
            return
        queue = self._field_queue(capture=True)
        if not queue:
            self.status.setText(
                "No fibers found in drawn fields. Draw a shape in the 'review_analysis_rois' "
                "layer on this section first."
            )
            return
        self._review_started = True
        self.controller.set_queue(queue, "reference_fields", position=self._first_undecided(queue))
        self._show_fibers()
        self.status.setText(f"Started labelling {len(queue)} fibers in the drawn fields.")
        self.refresh(notify=True)

    def _section_allowed(self) -> bool:
        image_id = self.controller.session.current_image_id
        if self.allowed_images is None or image_id in self.allowed_images:
            return True
        wanted, mode = ("test", "Blind labelling") if self.blind else ("pool", "Guided review")
        self.status.setText(
            f"{mode} is limited to {wanted} mice in this project, and this section is not one. "
            + (
                "Use 'Go to next section'."
                if self.blind
                else "Choose another section (see evaluation_roles.csv)."
            )
        )
        return False

    def start_section_review(self) -> None:
        if not self._section_allowed():
            return
        self._review_started = True
        self._show_fibers()
        image_id = self.controller.session.current_image_id
        section_rows = self.rows[self.rows["image_id"].eq(image_id)]
        queue = build_fiber_type_queue(section_rows, QueueSource.FULL_AUDIT)
        self.controller.set_queue(queue, "selected_section", position=self._first_undecided(queue))
        self.status.setText("Started full review for the current section.")
        self.refresh(notify=True)

    def _restore_saved_queue(self) -> None:
        """Rebuild a persisted review plan without discarding its position."""
        name = self.controller.session.active_queue
        try:
            source = QueueSource(name)
        except ValueError:
            source = None
        if source is not None:
            self._set_queue(
                source,
                notify=False,
                position=self.controller.session.queue_position,
            )
            return
        if name == "reference_fields":
            self.controller.set_queue(
                self._field_queue(capture=False),
                name,
                position=self.controller.session.queue_position,
            )
            return
        if name == "selected_section" and self.controller.session.current_image_id:
            image_id = self.controller.session.current_image_id
            section_rows = self.rows[self.rows["image_id"].eq(image_id)]
            queue = build_fiber_type_queue(section_rows, QueueSource.FULL_AUDIT)
            self.controller.set_queue(
                queue,
                name,
                position=self.controller.session.queue_position,
            )
            return
        # A new session needs a count for the plan chooser, but must not acquire
        # an active queue merely by opening the dock.
        self.controller.queue = build_fiber_type_queue(
            self.rows, QueueSource.FULL_AUDIT if self.blind else QueueSource.FLAGGED
        )

    def _change_plan(self) -> None:
        self._review_started = False
        self.undo_button.setEnabled(False)
        self.refresh()

    def _show_dashboard(self) -> None:
        if self.show_dashboard is not None:
            self.show_dashboard()

    def _show_section(self) -> None:
        if self.show_section is not None:
            self.show_section()

    def _show_region(self) -> None:
        if self.show_region is not None:
            self.show_region()

    def _show_fiber_typing(self) -> None:
        self.controller.session.active_domain = Domain.FIBER_TYPING
        self.controller.session.active_scope = Scope.OBJECT
        self._review_started = False
        self.refresh()

    def _show_domain(self, domain: Domain) -> None:
        if self.show_domain is not None:
            self.show_domain(domain)

    def _focus_current_object(self) -> None:
        if self.focus_current_object is not None:
            self.focus_current_object()

    def _tutorial_seen(self) -> bool:
        return str(
            self._settings.value(f"guided_review/{self.project.project_id}/tutorial_seen", False)
        ).lower() in {"true", "1"}

    def _dismiss_tutorial(self) -> None:
        self._settings.setValue(f"guided_review/{self.project.project_id}/tutorial_seen", True)
        self.tutorial_group.setVisible(False)

    def _show_tutorial(self) -> None:
        self.tutorial_group.setVisible(True)

    def _toggle_advanced(self, visible: bool) -> None:
        self.advanced_group.setVisible(visible)
        self.advanced_toggle.setArrowType(Qt.DownArrow if visible else Qt.RightArrow)

    def _queue_changed(self, source: str) -> None:
        if self.advanced_toggle.isChecked():
            self._set_queue(QueueSource(source))

    def _set_queue(
        self,
        source: QueueSource,
        *,
        notify: bool = True,
        position: int | None = None,
    ) -> None:
        if source is QueueSource.RANDOM_AUDIT:
            self._save_sample_settings()
        queue = build_fiber_type_queue(
            self.rows,
            source,
            seed=self.seed_spin.value(),
            sample_size=self.sample_spin.value(),
            random_scope=self.scope_combo.currentText(),
        )
        if position is None:
            position = self._first_undecided(queue)
        self.controller.set_queue(queue, source.value, position=position)
        self.queue_combo.blockSignals(True)
        self.queue_combo.setCurrentText(source.value)
        self.queue_combo.blockSignals(False)
        self.refresh(notify=notify)

    @property
    def _sample_settings_path(self):
        return self.project.review_directory / "random_sample_settings.json"

    def _save_sample_settings(self) -> None:
        self._sample_settings_path.parent.mkdir(parents=True, exist_ok=True)
        self._sample_settings_path.write_text(
            json.dumps(
                {
                    "seed": self.seed_spin.value(),
                    "sample_size": self.sample_spin.value(),
                    "scope": self.scope_combo.currentText(),
                }
            ),
            encoding="utf-8",
        )

    def _load_sample_settings(self) -> None:
        """Restore the last random sample's settings so reopening rebuilds the same sample."""
        if not self._sample_settings_path.is_file():
            return
        try:
            saved = json.loads(self._sample_settings_path.read_text(encoding="utf-8"))
            self.seed_spin.setValue(int(saved["seed"]))
            self.sample_spin.setValue(int(saved["sample_size"]))
            self.scope_combo.setCurrentText(str(saved["scope"]))
        except (ValueError, KeyError, TypeError):
            return

    def _keep_model_call(self) -> None:
        if self.blind:
            return
        self._record_special(ObjectReviewStatus.ACCEPTED)

    def _record_type(self, fiber_type: str) -> None:
        self._record(fiber_type)

    def _record_special(self, status: ObjectReviewStatus) -> None:
        reviewed_type = "exclude" if status is ObjectReviewStatus.EXCLUDED else None
        self._record(reviewed_type, status=status)

    def _record(self, fiber_type: str | None, *, status: ObjectReviewStatus | None = None) -> None:
        try:
            event = self.controller.decide(fiber_type, status=status)
            self.controller.save(self.project, event)
            self.undo_button.setEnabled(True)
            self.timer.activity(decision=True)
            if self._resume_after_click():
                self.controller.save(self.project)
                self.status.setText("Saved the clicked fiber; back to the plan. Undo available.")
                return
            self.controller.move(1)
            self.controller.save(self.project)
            self.status.setText("Autosaved. Moved to the next fiber; Undo is available.")
            self.refresh(notify=True)
        except ValueError as exc:
            self.status.setText(str(exc))

    def undo(self) -> None:
        try:
            event = self.controller.undo_last_decision()
            self.controller.save(self.project, event)
            self.undo_button.setEnabled(False)
            self.timer.activity()
            self.status.setText("Restored the previous decision.")
            self.refresh(notify=True)
        except ValueError as exc:
            self.status.setText(str(exc))

    def _move(self, offset: int) -> None:
        if self._resume_after_click():  # arrows leave a clicked fiber and return to the plan
            self.controller.save(self.project)
            return
        try:
            self.controller.move(offset)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.timer.activity()
        self.controller.save(self.project)
        self.refresh(notify=True)

    def refresh(self, *, notify: bool = False) -> None:
        item = self.controller.current_item
        count = len(self.controller.queue)
        # In blind mode the three labelling plans stay on screen, so switching is one click.
        self.plan_group.setVisible(self.blind or not self._review_started)
        self.change_plan_button.setVisible(self._review_started and not self.blind)
        # Blind labelling shows the sample settings before a plan starts, so the sample size can be
        # set first.
        show_advanced = self._review_started or self.blind
        self.advanced_toggle.setVisible(show_advanced)
        self.advanced_group.setVisible(show_advanced and self.advanced_toggle.isChecked())
        saved_count = len(self.controller.session.object_decisions)
        self.save_status.setText(
            f"Autosave is on · {saved_count} decision{'s' if saved_count != 1 else ''} saved"
        )
        if not self._review_started:
            self.context.setText("Choose which fibers to check")
            self.plan_message.setText(
                "Choose how fibers are picked. The random sample is one list through every "
                "section and resumes where you left off; a field covers every fiber in one area "
                "of the section on screen (◀ ▶ at the top change the section)."
                if self.blind
                else f"{count} flagged fibers are ready for review. Start there, or check every "
                "fiber in this section."
            )
            self.decision_group.setEnabled(False)
            return
        if item is None:
            self.context.setText("No fibers in this plan")
            self.plan_message.setText(
                "No fibers match the selected plan. Choose another review plan."
            )
            self.decision_group.setEnabled(False)
            return
        self.decision_group.setEnabled(True)
        where = (
            f"Section {self.section_codes.get(item.image_id, '?')} of {len(self.section_codes)}"
            if self.blind
            else f"{item.mouse_id} · {item.image_id}"
        )
        plan = (
            "clicked fiber"
            if self.controller.session.active_queue == CLICKED_QUEUE
            else PLAN_NAMES.get(item.queue_source.value, item.queue_source.value.replace("_", " "))
        )
        self.context.setText(
            f"{where} · {plan} · {self.controller.session.queue_position + 1} of {count}"
        )
        self.plan_message.setText(
            f"{self._labelled_in_queue()} of {count} fibers in this plan are labelled. Labels save "
            "as you go; switch plans with the buttons below."
            if self.blind
            else "A clicked fiber: decide, or press → to return to your plan."
            if self.controller.session.active_queue == CLICKED_QUEUE
            else f"{count} fibers are in the current plan; click any fiber to check it instead."
        )
        if self.blind:
            note = " | cut off by the image edge" if "cut_by_image_edge" in item.reason_code else ""
            self.details.setText(f"Fiber {item.fiber_id}{note}")
            if notify and self.object_changed is not None:
                self.object_changed(item.image_id, item.fiber_id)
            return
        values = [
            f"Model call: {item.model_fiber_type.upper()}",
            f"Why shown: {describe_reason(item.reason_code)}",
        ]
        saved = saved_call_text(
            self.controller.session.object_decisions, item.image_id, item.fiber_id
        )
        if saved is not None:
            values.insert(1, f"Your call: {saved}")
        if item.confidence is not None:
            values.append(f"confidence {item.confidence:.2f}")
        if item.probability_margin is not None:
            values.append(f"margin {item.probability_margin:.2f}")
        self.details.setText(" | ".join(values))
        if notify and self.object_changed is not None:
            self.object_changed(item.image_id, item.fiber_id)
