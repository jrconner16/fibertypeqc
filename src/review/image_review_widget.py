"""Qt controls for Phase 3 image-level review."""

from __future__ import annotations

from collections.abc import Callable

from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.review.image_review import ImageReviewController
from src.review.schemas import Domain, DomainStatus, ReviewMode


class ImageReviewWidget(QWidget):
    """Persisted image/domain actions; it deliberately exposes no object editing."""

    def __init__(
        self,
        controller: ImageReviewController,
        *,
        image_changed: Callable[[str], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.image_changed = image_changed
        self.setMinimumWidth(430)

        self.context = QLabel()
        self.context.setWordWrap(True)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([mode.value for mode in ReviewMode])
        self.mode_combo.setCurrentText(controller.session.active_review_mode.value)
        self.domain_combo = QComboBox()
        self.domain_combo.addItems([domain.value for domain in Domain])
        self.domain_combo.setCurrentText(controller.session.active_domain.value)
        self.image_combo = QComboBox()
        self.cards_layout = QVBoxLayout()
        self.cards_layout.setContentsMargins(0, 0, 0, 0)

        form = QFormLayout()
        form.addRow("Review mode", self.mode_combo)
        form.addRow("Domain", self.domain_combo)
        form.addRow("Image", self.image_combo)
        controls = QGroupBox("Review context")
        controls.setLayout(form)

        navigation = QHBoxLayout()
        previous = QPushButton("Previous")
        next_button = QPushButton("Next")
        navigation.addWidget(previous)
        navigation.addWidget(next_button)
        self.save_button = QPushButton("Save")
        navigation.addWidget(self.save_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.context)
        layout.addWidget(controls)
        layout.addLayout(navigation)
        layout.addWidget(QLabel("Image/domain status"))
        layout.addLayout(self.cards_layout)
        layout.addStretch(1)

        self.mode_combo.currentTextChanged.connect(self._mode_changed)
        self.domain_combo.currentTextChanged.connect(self._domain_changed)
        self.image_combo.currentTextChanged.connect(self._image_changed)
        previous.clicked.connect(lambda: self._move(-1))
        next_button.clicked.connect(lambda: self._move(1))
        self.save_button.clicked.connect(lambda: self.controller.save())
        self.refresh(notify=True)

    def _mode_changed(self, value: str) -> None:
        self.controller.set_mode(value)
        self.refresh(notify=True)

    def _domain_changed(self, value: str) -> None:
        self.controller.set_domain(value)
        self.refresh(notify=True)

    def _image_changed(self, value: str) -> None:
        if value:
            self.controller.set_image(value)
            self.refresh(notify=True)

    def _move(self, offset: int) -> None:
        try:
            self.controller.move(offset)
        except ValueError:
            return
        self.refresh(notify=True)

    def _set_status(self, status: DomainStatus) -> None:
        event = self.controller.set_status(status)
        self.controller.save(event)
        self.refresh()

    def refresh(self, *, notify: bool = False) -> None:
        ids = self.controller.navigation_ids()
        if not ids:
            ids = tuple(image.image_id for image in self.controller.project.images)
        self.image_combo.blockSignals(True)
        self.image_combo.clear()
        self.image_combo.addItems(ids)
        self.image_combo.setCurrentText(self.controller.current_image_id)
        self.image_combo.blockSignals(False)
        self.context.setText(
            f"<b>{self.controller.project.project_name}</b>"
            f" | Image: {self.controller.current_image_id}"
            f" | Scope: image | Domain: {self.controller.session.active_domain.value}"
            f" | Mode: {self.controller.session.active_review_mode.value}"
        )
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for card in self.controller.cards():
            group = QGroupBox(card.domain.value.replace("_", " ").title())
            rows = QVBoxLayout(group)
            rows.addWidget(
                QLabel(
                    f"Applicable: {'yes' if card.applicable else 'no'} | QC: {card.qc_status}"
                    f" | Reviewed: {card.review_status.value} | Priority: {card.review_priority}"
                )
            )
            if card.reason_codes:
                reason = QLabel(f"QC reasons: {card.reason_codes}")
                reason.setWordWrap(True)
                rows.addWidget(reason)
            actions = QHBoxLayout()
            for status in (
                DomainStatus.PASS,
                DomainStatus.REVIEW,
                DomainStatus.FAIL,
                DomainStatus.EXCLUDED,
                DomainStatus.NOT_APPLICABLE,
            ):
                button = QPushButton(status.value.replace("_", " ").title())
                button.setEnabled(
                    card.applicable and card.domain is self.controller.session.active_domain
                )
                button.clicked.connect(lambda _checked=False, value=status: self._set_status(value))
                actions.addWidget(button)
            rows.addLayout(actions)
            self.cards_layout.addWidget(group)
        if notify and self.image_changed is not None:
            self.image_changed(self.controller.current_image_id)
