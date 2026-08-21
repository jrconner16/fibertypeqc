"""Qt object inspector for canonical Phase 4 fiber-type decisions."""

from __future__ import annotations

from collections.abc import Callable

from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.review.fiber_type_review import FiberTypeReviewController
from src.review.project import Project
from src.review.queues import QueueSource, build_fiber_type_queue, load_fiber_type_rows
from src.review.schemas import ObjectReviewStatus


class ObjectInspectorWidget(QWidget):
    """Queue navigation and type adjudication; segmentation remains read-only."""

    def __init__(
        self,
        project: Project,
        controller: FiberTypeReviewController,
        *,
        object_changed: Callable[[str, int], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.controller = controller
        self.object_changed = object_changed
        self.rows = load_fiber_type_rows(project)
        self.setMinimumWidth(430)

        self.queue_combo = QComboBox()
        self.queue_combo.addItems([source.value for source in QueueSource])
        self.context = QLabel()
        self.context.setWordWrap(True)
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.status = QLabel()
        self.status.setWordWrap(True)

        navigation = QHBoxLayout()
        previous = QPushButton("Previous")
        next_button = QPushButton("Next")
        navigation.addWidget(previous)
        navigation.addWidget(next_button)

        type_actions = QHBoxLayout()
        for label in ("i", "iia", "iib", "iix"):
            button = QPushButton(label.upper())
            button.clicked.connect(lambda _checked=False, value=label: self._decide(value))
            type_actions.addWidget(button)
        other_actions = QHBoxLayout()
        for label, decision_status in (
            ("Accept model", ObjectReviewStatus.ACCEPTED),
            ("Uncertain", ObjectReviewStatus.UNCERTAIN),
            ("Exclude", ObjectReviewStatus.EXCLUDED),
            ("Unresolved", ObjectReviewStatus.UNRESOLVED),
        ):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, value=decision_status: self._special_decide(value)
            )
            other_actions.addWidget(button)

        queue_group = QGroupBox("Fiber-type object queue")
        queue_layout = QVBoxLayout(queue_group)
        queue_layout.addWidget(self.queue_combo)
        queue_layout.addLayout(navigation)
        decision_group = QGroupBox("Decision")
        decision_layout = QVBoxLayout(decision_group)
        decision_layout.addLayout(type_actions)
        decision_layout.addLayout(other_actions)

        layout = QVBoxLayout(self)
        layout.addWidget(self.context)
        layout.addWidget(queue_group)
        layout.addWidget(self.details)
        layout.addWidget(decision_group)
        layout.addWidget(self.status)
        layout.addStretch(1)

        self.queue_combo.currentTextChanged.connect(self._queue_changed)
        previous.clicked.connect(lambda: self._move(-1))
        next_button.clicked.connect(lambda: self._move(1))
        self._queue_changed(self.queue_combo.currentText())

    def _queue_changed(self, source: str) -> None:
        queue = build_fiber_type_queue(self.rows, source)
        self.controller.set_queue(queue, source)
        self.refresh(notify=True)

    def _move(self, offset: int) -> None:
        try:
            self.controller.move(offset)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.refresh(notify=True)

    def _decide(self, fiber_type: str) -> None:
        try:
            event = self.controller.decide(fiber_type)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.controller.save(self.project, event)
        self.status.setText(f"Saved {fiber_type.upper()} decision.")
        self.refresh()

    def _special_decide(self, status: ObjectReviewStatus) -> None:
        item = self.controller.current_item
        if item is None:
            self.status.setText("The fiber-type queue is empty")
            return
        reviewed_type = "exclude" if status is ObjectReviewStatus.EXCLUDED else None
        event = self.controller.decide(reviewed_type, status=status)
        self.controller.save(self.project, event)
        self.status.setText(f"Saved {status.value} decision.")
        self.refresh()

    def refresh(self, *, notify: bool = False) -> None:
        item = self.controller.current_item
        count = len(self.controller.queue)
        if item is None:
            self.context.setText("Fiber typing | Queue empty")
            self.details.setText("No objects match this queue.")
            return
        self.context.setText(
            f"Fiber typing | Queue: {item.queue_source.value}"
            f" | {self.controller.session.queue_position + 1}/{count}"
        )
        values = [
            f"image={item.image_id}",
            f"fiber_id={item.fiber_id}",
            f"model={item.model_fiber_type}",
            f"reason={item.reason_code}",
        ]
        if item.confidence is not None:
            values.append(f"confidence={item.confidence:.3f}")
        if item.probability_margin is not None:
            values.append(f"margin={item.probability_margin:.3f}")
        if item.normalized_entropy is not None:
            values.append(f"entropy={item.normalized_entropy:.3f}")
        self.details.setText(" | ".join(values))
        if notify and self.object_changed is not None:
            self.object_changed(item.image_id, item.fiber_id)
