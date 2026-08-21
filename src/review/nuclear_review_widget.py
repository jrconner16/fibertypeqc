"""Napari controls for copy-on-write nuclei review."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.review.nuclear_review import NuclearReviewController, NucleusQueueSource
from src.review.schemas import NucleusAssociationStatus


class NuclearReviewWidget(QWidget):
    """Save explicit nucleus-mask and association actions from Napari layers."""

    def __init__(
        self,
        controller: NuclearReviewController,
        *,
        selected_nucleus_id: Callable[[], int] | None = None,
        draft_pixels: Callable[[], np.ndarray] | None = None,
        reset_draft: Callable[[], None] | None = None,
        review_changed: Callable[[], None] | None = None,
        add_enabled: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.selected_nucleus_id = selected_nucleus_id
        self.draft_pixels = draft_pixels
        self.reset_draft = reset_draft
        self.review_changed = review_changed
        self.setMinimumWidth(370)

        self.context = QLabel()
        self.context.setWordWrap(True)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.nucleus_id = QSpinBox()
        self.nucleus_id.setRange(1, 2_147_483_647)
        self.sync_selection = QPushButton("Use selected nucleus")
        self.delete = QPushButton("Delete selected nucleus")
        self.add = QPushButton("Save painted draft nucleus")
        self.add.setEnabled(add_enabled)
        if not add_enabled:
            self.add.setToolTip("Restart with --display-downsample 1 to paint at mask resolution.")
        self.fiber_id = QSpinBox()
        self.fiber_id.setRange(0, 2_147_483_647)
        self.association_status = QComboBox()
        self.association_status.addItems([item.value for item in NucleusAssociationStatus])
        self.association_status.setCurrentText(NucleusAssociationStatus.ASSIGNED.value)
        self.reason = QLineEdit()
        self.reason.setPlaceholderText("Optional reason")
        self.associate = QPushButton("Save association")
        self.queue_source = QComboBox()
        self.queue_source.addItems([item.value for item in NucleusQueueSource])
        self.load_queue = QPushButton("Load association queue")
        self.previous = QPushButton("Previous")
        self.next = QPushButton("Next")

        mask_group = QGroupBox("Reviewed nucleus mask")
        mask_layout = QVBoxLayout(mask_group)
        mask_layout.addWidget(
            QLabel(
                "Select a nucleus in the reviewed labels layer, or paint only in "
                "the Draft new nucleus layer before saving."
            )
        )
        selected = QHBoxLayout()
        selected.addWidget(self.nucleus_id)
        selected.addWidget(self.sync_selection)
        mask_layout.addLayout(selected)
        mask_layout.addWidget(self.delete)
        mask_layout.addWidget(self.add)

        association_group = QGroupBox("Nucleus-to-fiber association")
        association_form = QFormLayout(association_group)
        association_form.addRow("Status", self.association_status)
        association_form.addRow("Reviewed fiber ID", self.fiber_id)
        association_form.addRow("Reason", self.reason)
        association_form.addRow(self.associate)

        queue_group = QGroupBox("Association queue")
        queue_layout = QVBoxLayout(queue_group)
        queue_layout.addWidget(self.queue_source)
        queue_layout.addWidget(self.load_queue)
        navigation = QHBoxLayout()
        navigation.addWidget(self.previous)
        navigation.addWidget(self.next)
        queue_layout.addLayout(navigation)

        layout = QVBoxLayout(self)
        layout.addWidget(self.context)
        layout.addWidget(mask_group)
        layout.addWidget(association_group)
        layout.addWidget(queue_group)
        layout.addWidget(self.status)
        layout.addStretch(1)

        self.sync_selection.clicked.connect(self._sync_selection)
        self.delete.clicked.connect(self._delete)
        self.add.clicked.connect(self._add)
        self.associate.clicked.connect(self._associate)
        self.load_queue.clicked.connect(self._load_queue)
        self.previous.clicked.connect(lambda: self._move(-1))
        self.next.clicked.connect(lambda: self._move(1))
        self.association_status.currentTextChanged.connect(self._status_changed)
        self._status_changed(self.association_status.currentText())
        self.refresh()

    def _image_id(self) -> str:
        image_id = self.controller.session.current_image_id
        if not image_id:
            raise ValueError("Select an image before reviewing nuclei")
        return image_id

    def _status_changed(self, value: str) -> None:
        self.fiber_id.setEnabled(value == NucleusAssociationStatus.ASSIGNED.value)
        if not self.fiber_id.isEnabled():
            self.fiber_id.setValue(0)

    def _sync_selection(self) -> None:
        if self.selected_nucleus_id is None:
            self.status.setText("Select a label in the reviewed nuclei layer first.")
            return
        nucleus_id = self.selected_nucleus_id()
        if nucleus_id <= 0:
            self.status.setText("Select a non-background nucleus in the reviewed nuclei layer.")
            return
        self.nucleus_id.setValue(nucleus_id)
        self.status.setText(f"Selected nucleus {nucleus_id}.")

    def _delete(self) -> None:
        try:
            event = self.controller.delete_nucleus(
                self._image_id(), self.nucleus_id.value(), reason_code=self.reason.text().strip()
            )
            self.controller.save(event)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.status.setText(
            "Deleted nucleus and autosaved reviewed outputs; dependent outputs are stale."
        )
        self._changed()

    def _add(self) -> None:
        if self.draft_pixels is None:
            self.status.setText("The draft nucleus layer is unavailable.")
            return
        try:
            event = self.controller.add_nucleus(
                self._image_id(), self.draft_pixels(), reason_code=self.reason.text().strip()
            )
            self.controller.save(event)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        if self.reset_draft is not None:
            self.reset_draft()
        self.nucleus_id.setValue(int(event.target_id))
        self.status.setText(
            f"Added nucleus {event.target_id} and autosaved; assign it to a fiber "
            "or leave unresolved."
        )
        self._changed()

    def _associate(self) -> None:
        try:
            event = self.controller.set_association(
                self._image_id(),
                self.nucleus_id.value(),
                fiber_id=self.fiber_id.value(),
                status=self.association_status.currentText(),
                reason_code=self.reason.text().strip(),
            )
            self.controller.save(event)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.status.setText(
            "Association autosaved; association and fiber nucleus counts are stale."
        )
        self._changed()

    def _load_queue(self) -> None:
        try:
            self.controller.set_association_queue(self._image_id(), self.queue_source.currentText())
            self.controller.save()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.status.setText("Association queue loaded and autosaved.")
        self.refresh()

    def _move(self, offset: int) -> None:
        try:
            item = self.controller.move(offset)
            self.controller.save()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.nucleus_id.setValue(item.nucleus_id)
        self.status.setText(f"Showing nucleus {item.nucleus_id}.")
        self.refresh()

    def _changed(self) -> None:
        self.reason.clear()
        self.refresh()
        if self.review_changed is not None:
            self.review_changed()

    def refresh(self) -> None:
        image_id = self.controller.session.current_image_id or "no image selected"
        item = self.controller.current_item
        stale = self.controller.session.stale_products.get(image_id, [])
        queue_text = "No association queue loaded"
        if item is not None:
            queue_text = (
                f"Queue: {item.queue_source.value.replace('_', ' ')} · "
                f"{self.controller.session.queue_position + 1}/{len(self.controller.queue)}"
            )
        self.context.setText(
            f"Nuclei review · {image_id}\n{queue_text}\n"
            f"Stale: {', '.join(stale) if stale else 'none'}"
        )
