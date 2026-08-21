"""Qt cohort dashboard for precomputed review QC."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.review.dashboard import DashboardTables, build_dashboard_model
from src.review.project import Project
from src.review.schemas import Domain
from src.review.section_selection import SelectionStrategy
from src.review.session import ReviewSession

STATUS_COLORS = {
    "pass": QColor("#d8f3dc"),
    "review": QColor("#fff3bf"),
    "fail": QColor("#ffd6d6"),
    "not_applicable": QColor("#e9ecef"),
    "complete": QColor("#d8f3dc"),
    "targeted_review": QColor("#fff3bf"),
    "no_acceptable_section": QColor("#ffd6d6"),
}


class CohortDashboardWidget(QWidget):
    """Read-only Phase 2B dashboard backed by the headless dashboard model."""

    def __init__(
        self,
        project: Project,
        tables: DashboardTables,
        *,
        strategy: SelectionStrategy | str = SelectionStrategy.ALL_PASSING,
        manual_selections: dict[str, dict[str, list[str]]] | None = None,
        session: ReviewSession | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.tables = tables
        self.manual_selections = manual_selections
        self.session = session
        self.model = build_dashboard_model(
            project,
            tables,
            strategy=strategy,
            manual_selections=manual_selections,
            session=session,
        )
        self.setMinimumWidth(760)

        self.context_label = QLabel()
        self.context_label.setWordWrap(True)
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems([item.value for item in SelectionStrategy])
        self.strategy_combo.setCurrentText(self.model.strategy.value)
        self.domain_filter = QComboBox()
        self.domain_filter.addItems(["all", *(domain.value for domain in Domain)])
        self.status_filter = QComboBox()
        self.status_filter.addItems(
            ["all", "pass", "review", "fail", "not_applicable"]
        )

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Selection strategy"))
        controls.addWidget(self.strategy_combo)
        controls.addWidget(QLabel("Domain"))
        controls.addWidget(self.domain_filter)
        controls.addWidget(QLabel("Status"))
        controls.addWidget(self.status_filter)
        controls.addStretch(1)

        self.domain_table = QTableWidget(0, 5)
        self.domain_table.setHorizontalHeaderLabels(
            ["Domain", "PASS", "REVIEW", "FAIL", "N/A"]
        )
        self.domain_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.domain_table.setSelectionMode(QTableWidget.NoSelection)
        self.domain_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )

        self.section_tree = QTreeWidget()
        self.section_tree.setColumnCount(7)
        self.section_tree.setHeaderLabels(
            ["Mouse / section", "Domain", "Status", "Score", "Priority", "Selected", "Reasons"]
        )
        self.section_tree.setAlternatingRowColors(True)
        self.section_tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.section_tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.section_tree.header().setSectionResizeMode(6, QHeaderView.Stretch)

        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText("Select a domain row to inspect explicit QC reasons.")
        self.details.setMaximumBlockCount(500)

        counts_group = QGroupBox("Domain status counts")
        counts_layout = QVBoxLayout(counts_group)
        counts_layout.addWidget(self.domain_table)
        details_group = QGroupBox("Selected QC details")
        details_layout = QVBoxLayout(details_group)
        details_layout.addWidget(self.details)

        layout = QVBoxLayout(self)
        layout.addWidget(self.context_label)
        layout.addWidget(self.summary_label)
        layout.addLayout(controls)
        layout.addWidget(counts_group)
        layout.addWidget(self.section_tree, stretch=3)
        layout.addWidget(details_group, stretch=1)

        self.strategy_combo.currentTextChanged.connect(self._strategy_changed)
        self.domain_filter.currentTextChanged.connect(self._refresh_tree)
        self.status_filter.currentTextChanged.connect(self._refresh_tree)
        self.section_tree.itemSelectionChanged.connect(self._show_selected_details)
        self.refresh()

    def _strategy_changed(self, strategy: str) -> None:
        self.model = build_dashboard_model(
            self.project,
            self.tables,
            strategy=strategy,
            manual_selections=self.manual_selections,
            session=self.session,
        )
        self.refresh()

    def refresh(self) -> None:
        self._refresh_context()
        self._refresh_summary()
        self._refresh_domain_counts()
        self._refresh_tree()

    def _refresh_context(self) -> None:
        self.context_label.setText(
            f"<b>{self.project.project_name}</b> ({self.project.project_id})"
            f" &nbsp;|&nbsp; Scope: cohort"
            f" &nbsp;|&nbsp; Domain: {self.domain_filter.currentText()}"
            f" &nbsp;|&nbsp; Strategy: {self.model.strategy.value}"
            f" &nbsp;|&nbsp; Model: {self.project.model_version}"
        )

    def _refresh_summary(self) -> None:
        summary = self.model.summary
        progress = 100.0 * summary.review_progress_fraction
        self.summary_label.setText(
            f"Mice: {summary.mouse_count} &nbsp; Sections: {summary.section_count}"
            f" &nbsp; Complete: {summary.complete_mouse_count}"
            f" &nbsp; Targeted review: {summary.targeted_review_mouse_count}"
            f" &nbsp; No acceptable section: {summary.no_acceptable_mouse_count}<br>"
            f"Review progress: {summary.reviewed_domain_rows}/"
            f"{summary.applicable_domain_rows} ({progress:.1f}%)"
            f" &nbsp; Object decisions: {summary.object_decision_count}"
            f" &nbsp; Regions: {summary.region_count}"
            f" &nbsp; Reviewed masks: {summary.reviewed_mask_count}"
        )

    def _refresh_domain_counts(self) -> None:
        self.domain_table.setRowCount(len(self.model.domain_counts))
        for row_index, row in enumerate(self.model.domain_counts.itertuples(index=False)):
            values = (
                row.domain,
                row.pass_count,
                row.review_count,
                row.fail_count,
                row.not_applicable_count,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column > 0:
                    item.setTextAlignment(Qt.AlignCenter)
                self.domain_table.setItem(row_index, column, item)

    def _refresh_tree(self) -> None:
        self._refresh_context()
        self.section_tree.clear()
        domain_filter = self.domain_filter.currentText()
        status_filter = self.status_filter.currentText()
        for mouse in self.model.mouse_rows.itertuples(index=False):
            mouse_item = QTreeWidgetItem(
                [str(mouse.mouse_id), "", str(mouse.readiness), "", "", "", ""]
            )
            self._color_status(mouse_item, str(mouse.readiness))
            self.section_tree.addTopLevelItem(mouse_item)
            mouse_sections = [
                image for image in self.project.images if image.mouse_id == mouse.mouse_id
            ]
            for image in mouse_sections:
                rows = self.model.section_rows[
                    self.model.section_rows["image_id"].eq(image.image_id)
                ]
                if domain_filter != "all":
                    rows = rows[rows["domain"].eq(domain_filter)]
                if status_filter != "all":
                    rows = rows[rows["status"].eq(status_filter)]
                if rows.empty:
                    continue
                section_item = QTreeWidgetItem(
                    [f"{image.section_id} ({image.image_id})", "", "", "", "", "", ""]
                )
                mouse_item.addChild(section_item)
                for _, row in rows.iterrows():
                    score = (
                        ""
                        if pd.isna(row["technical_quality_score"])
                        else f"{float(row['technical_quality_score']):.2f}"
                    )
                    item = QTreeWidgetItem(
                        [
                            image.image_id,
                            str(row["domain"]),
                            str(row["status"]),
                            score,
                            str(int(row["review_priority"])),
                            "yes" if bool(row["selected"]) else "",
                            "" if pd.isna(row["reason_codes"]) else str(row["reason_codes"]),
                        ]
                    )
                    item.setData(0, Qt.UserRole, row.to_dict())
                    self._color_status(item, str(row["status"]))
                    section_item.addChild(item)
            if mouse_item.childCount() == 0:
                self.section_tree.takeTopLevelItem(
                    self.section_tree.indexOfTopLevelItem(mouse_item)
                )
            else:
                mouse_item.setExpanded(True)
        self.section_tree.resizeColumnToContents(2)

    def _color_status(self, item: QTreeWidgetItem, status: str) -> None:
        color = STATUS_COLORS.get(status)
        if color is None:
            return
        for column in range(self.section_tree.columnCount()):
            item.setBackground(column, color)

    def _show_selected_details(self) -> None:
        selected = self.section_tree.selectedItems()
        if not selected:
            self.details.clear()
            return
        data: dict[str, Any] | None = selected[0].data(0, Qt.UserRole)
        if not data:
            self.details.setPlainText("Select an image/domain row for detailed QC evidence.")
            return
        details: dict[str, Any] = {
            "image_id": data.get("image_id"),
            "mouse_id": data.get("mouse_id"),
            "section_id": data.get("section_id"),
            "domain": data.get("domain"),
            "status": data.get("status"),
            "technical_quality_score": data.get("technical_quality_score"),
            "review_priority": data.get("review_priority"),
            "selected": data.get("selected"),
            "reason_codes": data.get("reason_codes") or "",
        }
        reason_json = data.get("reason_details_json")
        if isinstance(reason_json, str) and reason_json.strip():
            try:
                details["reason_details"] = json.loads(reason_json)
            except json.JSONDecodeError:
                details["reason_details"] = reason_json
        metrics = {
            key: value
            for key, value in data.items()
            if key
            not in {
                "schema_version",
                "qc_version",
                "rules_version",
                "model_version",
                "computed_at",
                "project_id",
                "image_id",
                "mouse_id",
                "section_id",
                "domain",
                "applicable",
                "status",
                "hard_fail",
                "technical_quality_score",
                "review_priority",
                "reason_codes",
                "reason_details_json",
                "artifact_paths_json",
                "selected",
            }
            and not pd.isna(value)
        }
        details["available_metrics"] = metrics
        self.details.setPlainText(json.dumps(details, indent=2, default=str, sort_keys=True))
