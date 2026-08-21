"""Focused, reviewer-oriented workspace for Phase 4.5."""

from __future__ import annotations

from collections.abc import Callable

from qtpy.QtCore import QSettings, Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QShortcut,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.review.fiber_type_review import FiberTypeReviewController
from src.review.project import Project
from src.review.queues import (
    QueueSource,
    RandomAuditScope,
    build_fiber_type_queue,
    load_fiber_type_rows,
)
from src.review.schemas import Domain, ObjectReviewStatus, Scope


class GuidedReviewWidget(QWidget):
    """One-task-at-a-time fiber reviewer with optional advanced controls."""

    def __init__(
        self,
        project: Project,
        controller: FiberTypeReviewController,
        *,
        object_changed: Callable[[str, int], None] | None = None,
        show_dashboard: Callable[[], None] | None = None,
        show_section: Callable[[], None] | None = None,
        show_region: Callable[[], None] | None = None,
        show_domain: Callable[[Domain], None] | None = None,
        focus_current_object: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.controller = controller
        self.object_changed = object_changed
        self.show_dashboard = show_dashboard
        self.show_section = show_section
        self.show_region = show_region
        self.show_domain = show_domain
        self.focus_current_object = focus_current_object
        self.rows = load_fiber_type_rows(project)
        self._review_started = bool(self.controller.session.active_queue)
        self._settings = QSettings("FiberTypeQC", "FiberTypeQC")
        self.setMinimumWidth(410)

        self.context = QLabel()
        self.context.setWordWrap(True)
        self.navigator_group = QGroupBox("Navigate review")
        navigator = QHBoxLayout(self.navigator_group)
        for label, callback in (
            ("Cohort", self._show_dashboard),
            ("Section", self._show_section),
            ("Regions", self._show_region),
            ("Fiber typing", self._show_fiber_typing),
            ("Segmentation", lambda: self._show_domain(Domain.FIBER_SEGMENTATION)),
            ("Nuclei", lambda: self._show_domain(Domain.NUCLEI)),
        ):
            button = QToolButton()
            button.setText(label)
            button.clicked.connect(callback)
            navigator.addWidget(button)
        navigator.addStretch(1)
        self.navigator = navigator
        self.review_guide_button = QToolButton()
        self.review_guide_button.setText("? Review guide")
        navigator.addWidget(self.review_guide_button)
        self.tutorial_group = QGroupBox("Welcome to guided review")
        tutorial_layout = QVBoxLayout(self.tutorial_group)
        tutorial = QLabel(
            "1. Choose a review plan.\n"
            "2. Check the current fiber and keep or correct the model call.\n"
            "3. Decisions save immediately; Undo restores the last one."
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
        self.change_plan_button = QToolButton()
        self.change_plan_button.setText("Change review plan")
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.save_status = QLabel()
        self.save_status.setWordWrap(True)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.undo_button = QPushButton("Undo last decision")
        self.undo_button.setEnabled(False)
        self.shortcut_label = QLabel(
            "Shortcuts: ←/→ navigate · F center fiber · K keep model · "
            "1 I · 2 IIa · 3 IIb · 4 IIx · U undo"
        )
        self.shortcut_label.setWordWrap(True)

        self.plan_group = QGroupBox("What needs attention?")
        plan_layout = QVBoxLayout(self.plan_group)
        plan_layout.addWidget(self.plan_message)
        plan_layout.addWidget(self.start_flagged_button)
        plan_layout.addWidget(self.review_section_button)
        plan_layout.addWidget(self.cohort_qc_button)

        self.decision_group = QGroupBox("Current fiber")
        decision_layout = QVBoxLayout(self.decision_group)
        decision_layout.addWidget(self.details)
        primary = QHBoxLayout()
        keep = QPushButton("Keep model call (K)")
        primary.addWidget(keep)
        for label, fiber_type in (
            ("I (1)", "i"),
            ("IIa (2)", "iia"),
            ("IIb (3)", "iib"),
            ("IIx (4)", "iix"),
        ):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, value=fiber_type: self._record_type(value)
            )
            primary.addWidget(button)
        decision_layout.addLayout(primary)
        self.focus_button = QPushButton("Center current fiber (F)")
        self.focus_button.clicked.connect(self._focus_current_object)
        decision_layout.addWidget(self.focus_button)
        decision_layout.addWidget(self.undo_button)

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("Advanced review options")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setArrowType(Qt.RightArrow)
        self.advanced_group = QGroupBox()
        advanced_layout = QVBoxLayout(self.advanced_group)
        advanced_actions = QHBoxLayout()
        for label, decision_status in (
            ("Uncertain", ObjectReviewStatus.UNCERTAIN),
            ("Exclude", ObjectReviewStatus.EXCLUDED),
            ("Unresolved", ObjectReviewStatus.UNRESOLVED),
        ):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, value=decision_status: self._record_special(value)
            )
            advanced_actions.addWidget(button)
        advanced_layout.addLayout(advanced_actions)
        self.queue_combo = QComboBox()
        self.queue_combo.addItems([source.value for source in QueueSource])
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

        layout = QVBoxLayout(self)
        layout.addWidget(self.context)
        layout.addWidget(self.navigator_group)
        layout.addWidget(self.tutorial_group)
        layout.addWidget(self.plan_group)
        layout.addWidget(self.change_plan_button)
        layout.addWidget(self.decision_group)
        layout.addLayout(navigation)
        layout.addWidget(self.advanced_toggle)
        layout.addWidget(self.advanced_group)
        layout.addWidget(self.save_status)
        layout.addWidget(self.status)
        layout.addWidget(self.shortcut_label)
        layout.addStretch(1)

        self.start_flagged_button.clicked.connect(self.start_flagged_review)
        self.review_section_button.clicked.connect(self.start_section_review)
        self.cohort_qc_button.clicked.connect(self._show_dashboard)
        self.review_guide_button.clicked.connect(self._show_tutorial)
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
        self._add_shortcut("K", self._keep_model_call)
        self._add_shortcut("1", lambda: self._record_type("i"))
        self._add_shortcut("2", lambda: self._record_type("iia"))
        self._add_shortcut("3", lambda: self._record_type("iib"))
        self._add_shortcut("4", lambda: self._record_type("iix"))
        self._add_shortcut("U", self.undo)
        self._restore_saved_queue()
        self.tutorial_group.setVisible(not self._tutorial_seen())
        self.refresh()

    def start_flagged_review(self) -> None:
        self._review_started = True
        self._set_queue(QueueSource.FLAGGED)
        self.status.setText("Started flagged-fiber review.")

    def _add_shortcut(self, sequence: str, callback: Callable[[], None]) -> None:
        shortcut = QShortcut(QKeySequence(sequence), self, activated=callback)
        self.shortcuts.append(shortcut)

    def start_section_review(self) -> None:
        self._review_started = True
        image_id = self.controller.session.current_image_id
        section_rows = self.rows[self.rows["image_id"].eq(image_id)]
        queue = build_fiber_type_queue(section_rows, QueueSource.FULL_AUDIT)
        self.controller.set_queue(queue, "selected_section")
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
        self.controller.queue = build_fiber_type_queue(self.rows, QueueSource.FLAGGED)

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
        position: int = 0,
    ) -> None:
        queue = build_fiber_type_queue(
            self.rows,
            source,
            seed=self.seed_spin.value(),
            sample_size=self.sample_spin.value(),
            random_scope=self.scope_combo.currentText(),
        )
        self.controller.set_queue(queue, source.value, position=position)
        self.queue_combo.blockSignals(True)
        self.queue_combo.setCurrentText(source.value)
        self.queue_combo.blockSignals(False)
        self.refresh(notify=notify)

    def _keep_model_call(self) -> None:
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
            self.status.setText("Restored the previous decision.")
            self.refresh(notify=True)
        except ValueError as exc:
            self.status.setText(str(exc))

    def _move(self, offset: int) -> None:
        try:
            self.controller.move(offset)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.controller.save(self.project)
        self.refresh(notify=True)

    def refresh(self, *, notify: bool = False) -> None:
        item = self.controller.current_item
        count = len(self.controller.queue)
        self.plan_group.setVisible(not self._review_started)
        self.change_plan_button.setVisible(self._review_started)
        self.advanced_toggle.setVisible(self._review_started)
        self.advanced_group.setVisible(self._review_started and self.advanced_toggle.isChecked())
        saved_count = len(self.controller.session.object_decisions)
        self.save_status.setText(
            f"Autosave is on · {saved_count} decision{'s' if saved_count != 1 else ''} saved"
        )
        if not self._review_started:
            self.context.setText("Fiber typing · Choose a review plan")
            self.plan_message.setText(
                f"{count} flagged fibers are ready for review. "
                "Start there, review the current section, or inspect cohort QC."
            )
            self.decision_group.setEnabled(False)
            return
        if item is None:
            self.context.setText("Fiber typing | No objects in this review plan")
            self.plan_message.setText(
                "No fibers match the selected plan. Choose another review plan."
            )
            self.decision_group.setEnabled(False)
            return
        self.decision_group.setEnabled(True)
        self.context.setText(
            f"{item.mouse_id} · {item.image_id} · Fiber typing · "
            f"{item.queue_source.value.replace('_', ' ')} · "
            f"{self.controller.session.queue_position + 1}/{count}"
        )
        self.plan_message.setText(
            f"{count} fibers are in the current plan. Start with flagged fibers, "
            "or review every fiber in the current section."
        )
        values = [
            f"Model call: {item.model_fiber_type.upper()}",
            f"Why shown: {item.reason_code.replace('_', ' ')}",
        ]
        if item.confidence is not None:
            values.append(f"confidence {item.confidence:.2f}")
        if item.probability_margin is not None:
            values.append(f"margin {item.probability_margin:.2f}")
        self.details.setText(" | ".join(values))
        if notify and self.object_changed is not None:
            self.object_changed(item.image_id, item.fiber_id)
