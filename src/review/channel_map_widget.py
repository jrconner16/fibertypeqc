"""Compact stain/channel legend for the Napari layer controls."""

from __future__ import annotations

from qtpy.QtGui import QColor
from qtpy.QtWidgets import QGridLayout, QLabel, QSizePolicy, QWidget

from src.review.channel_display import ChannelDisplay


class ChannelMapWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.setMaximumWidth(260)
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.layout.setHorizontalSpacing(6)
        self.layout.setVerticalSpacing(2)
        self.layout.addWidget(QLabel("Channel map"), 0, 0, 1, 3)
        self._display_widgets: list[QWidget] = []

    def set_displays(self, displays: tuple[ChannelDisplay, ...]) -> None:
        for widget in self._display_widgets:
            self.layout.removeWidget(widget)
            widget.deleteLater()
        self._display_widgets.clear()
        for row, display in enumerate(displays, start=1):
            swatch = QLabel("●")
            color = QColor.fromRgbF(*display.rgb)
            swatch.setStyleSheet(f"color: {color.name()}; font-size: 16px;")
            role = QLabel(display.role)
            channel = QLabel(f"ch {display.channel}")
            role.setWordWrap(False)
            channel.setWordWrap(False)
            self.layout.addWidget(swatch, row, 0)
            self.layout.addWidget(role, row, 1)
            self.layout.addWidget(channel, row, 2)
            self._display_widgets.extend((swatch, role, channel))
