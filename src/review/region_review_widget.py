"""Qt controls for explicit, non-destructive region review."""

from __future__ import annotations

from collections.abc import Callable

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.review.region_review import RegionReviewController
from src.review.schemas import Domain, RegionAction


class RegionReviewWidget(QWidget):
    """Apply an action only after a polygon exists in the shared shapes layer."""

    def __init__(
        self,
        controller: RegionReviewController,
        *,
        selected_geometry: Callable[[], dict | None],
        regions_changed: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.selected_geometry = selected_geometry
        self.regions_changed = regions_changed
        self.setMinimumWidth(360)

        self.context = QLabel()
        self.context.setWordWrap(True)
        self.domain = QComboBox()
        self.domain.addItems([item.value for item in Domain])
        self.action = QComboBox()
        self.action.addItems([item.value.replace("_", " ") for item in RegionAction])
        self.reason = QLineEdit()
        self.reason.setPlaceholderText("Optional reason, e.g. fold or edge artifact")
        self.notes = QTextEdit()
        self.notes.setPlaceholderText("Optional reviewer notes")
        self.notes.setMaximumHeight(70)
        self.apply_button = QPushButton("Apply action to selected shape")
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.region_list = QListWidget()
        self.remove_button = QPushButton("Remove selected saved region")

        form = QFormLayout()
        form.addRow("Domain", self.domain)
        form.addRow("Action", self.action)
        form.addRow("Reason", self.reason)
        form.addRow("Notes", self.notes)
        action_group = QGroupBox("Region action")
        action_layout = QVBoxLayout(action_group)
        action_layout.addWidget(
            QLabel("Draw/select a polygon first. Nothing is saved until this action is applied.")
        )
        action_layout.addLayout(form)
        action_layout.addWidget(self.apply_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.context)
        layout.addWidget(action_group)
        layout.addWidget(QLabel("Saved regions in this section"))
        layout.addWidget(self.region_list)
        layout.addWidget(self.remove_button)
        layout.addWidget(self.status)

        self.apply_button.clicked.connect(self._apply)
        self.remove_button.clicked.connect(self._remove)
        self.refresh()

    def _apply(self) -> None:
        geometry = self.selected_geometry()
        if geometry is None:
            self.status.setText("Draw a polygon and select it before applying an action.")
            return
        try:
            event = self.controller.add_region(
                image_id=self.controller.session.current_image_id or "",
                geometry=geometry,
                domain=self.domain.currentText(),
                action=self.action.currentText().replace(" ", "_"),
                reason_code=self.reason.text().strip(),
                notes=self.notes.toPlainText().strip(),
            )
            self.controller.save(event)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.status.setText("Autosaved region action.")
        self.reason.clear()
        self.notes.clear()
        self.refresh()
        if self.regions_changed is not None:
            self.regions_changed()

    def _remove(self) -> None:
        item = self.region_list.currentItem()
        if item is None:
            self.status.setText("Select a saved region to remove it.")
            return
        try:
            event = self.controller.remove_region(item.data(32))
            self.controller.save(event)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.status.setText("Removed region and autosaved the updated review record.")
        self.refresh()
        if self.regions_changed is not None:
            self.regions_changed()

    def refresh(self) -> None:
        image_id = self.controller.session.current_image_id or ""
        self.context.setText(f"Region review · {image_id} · draw in the Region shapes layer")
        self.domain.setCurrentText(self.controller.session.active_domain.value)
        self.region_list.clear()
        for region in self.controller.regions_for_image(image_id):
            item = QListWidgetItem(
                f"{region.domain.value}: {region.action.replace('_', ' ')}"
                f"{' · ' + region.reason_code if region.reason_code else ''}"
            )
            item.setData(Qt.UserRole, region.region_id)
            self.region_list.addItem(item)
