"""The Section stage of guided review: one plain verdict per section, and areas to leave out.

It records the same decisions as the advanced panels (section status per domain, and regions with
the action ``exclude_all_analysis``), so finalization treats them identically.
"""

from __future__ import annotations

from collections.abc import Callable

from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from src.review.image_review import ImageReviewController
from src.review.project import Project
from src.review.region_review import RegionReviewController
from src.review.schemas import Domain, DomainStatus, RegionAction, RegionKind
from src.review.session import ReviewSession

SECTION_DOMAINS = (Domain.FIBER_SEGMENTATION, Domain.FIBER_TYPING)
LEFT_OUT = frozenset({DomainStatus.EXCLUDED, DomainStatus.FAIL})
AREA_REASON = "reviewer_left_out_area"


def section_verdict(project: Project, session: ReviewSession, image_id: str) -> str:
    """'left_out', 'fine', or 'not_checked' from the section's fiber-domain statuses."""
    image = project.image(image_id)
    domains = [domain for domain in SECTION_DOMAINS if domain in image.applicable_domains]
    statuses = [session.get_status(image_id, domain) for domain in domains]
    if any(status in LEFT_OUT for status in statuses):
        return "left_out"
    if statuses and all(status is DomainStatus.PASS for status in statuses):
        return "fine"
    return "not_checked"


def left_out_areas(session: ReviewSession, image_id: str) -> list:
    return [
        region
        for region in session.regions
        if region.image_id == image_id
        and region.kind is RegionKind.REVIEW
        and region.action == RegionAction.EXCLUDE_ALL_ANALYSIS.value
    ]


class SectionStageWidget(QWidget):
    """Verdict for the section on screen: looks fine, leave out an area, or leave it all out."""

    def __init__(
        self,
        image_controller: ImageReviewController,
        region_controller: RegionReviewController,
        *,
        start_drawing: Callable[[], None] | None = None,
        stop_drawing: Callable[[], None] | None = None,
        drawn_geometry: Callable[[], dict | None] | None = None,
        regions_changed: Callable[[], None] | None = None,
        go_to_fibers: Callable[[], None] | None = None,
        section_changed: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.images = image_controller
        self.regions = region_controller
        self.start_drawing = start_drawing
        self.stop_drawing = stop_drawing
        self.drawn_geometry = drawn_geometry
        self.regions_changed = regions_changed
        self.section_changed = section_changed
        self._drawing = False

        self.qc = QLabel()
        self.qc.setWordWrap(True)
        self.verdict = QLabel()
        self.verdict.setWordWrap(True)
        contrast = QLabel(
            "Tip: set each channel's background with the lower contrast limit in the layer "
            "controls (top left); it is kept for this section."
        )
        contrast.setWordWrap(True)
        self.fine_button = QPushButton("Looks fine")
        self.fine_button.setToolTip("Mark this section as checked and usable.")
        self.area_button = QPushButton("Leave out an area…")
        self.area_button.setToolTip(
            "Draw around a fold, tear, or damaged area; fibers inside it are left out of results."
        )
        self.section_button = QPushButton("Leave out the whole section")
        self.section_button.setToolTip("No fiber from this section goes into the results.")
        self.clear_button = QPushButton("Clear verdict")
        verdicts = QVBoxLayout()
        first_row = QHBoxLayout()
        first_row.addWidget(self.fine_button)
        first_row.addWidget(self.area_button)
        verdicts.addLayout(first_row)
        verdicts.addWidget(self.section_button)

        self.draw_help = QLabel(
            "Click around the area on the image (double-click to close the shape), then press "
            "Save area."
        )
        self.draw_help.setWordWrap(True)
        self.save_area_button = QPushButton("Save area (left out)")
        self.cancel_area_button = QPushButton("Cancel")
        drawing = QHBoxLayout()
        drawing.addWidget(self.save_area_button)
        drawing.addWidget(self.cancel_area_button)
        self.areas = QLabel()
        self.areas.setWordWrap(True)
        self.remove_area_button = QPushButton("Remove last area")
        self.fibers_button = QPushButton("Check fibers →")
        self.fibers_button.setToolTip("Go to the Fibers stage for this section.")
        self.status = QLabel()
        self.status.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.qc)
        layout.addWidget(self.verdict)
        layout.addLayout(verdicts)
        layout.addWidget(self.draw_help)
        layout.addLayout(drawing)
        layout.addWidget(self.areas)
        layout.addWidget(self.remove_area_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(contrast)
        layout.addWidget(self.status)
        layout.addStretch(1)
        layout.addWidget(self.fibers_button)

        self.fine_button.clicked.connect(lambda: self.set_verdict(DomainStatus.PASS))
        self.section_button.clicked.connect(lambda: self.set_verdict(DomainStatus.EXCLUDED))
        self.clear_button.clicked.connect(lambda: self.set_verdict(DomainStatus.NOT_REVIEWED))
        self.area_button.clicked.connect(self.begin_area)
        self.save_area_button.clicked.connect(self.save_area)
        self.cancel_area_button.clicked.connect(self.cancel_area)
        self.remove_area_button.clicked.connect(self.remove_last_area)
        if go_to_fibers is not None:
            self.fibers_button.clicked.connect(lambda: go_to_fibers())
        self.refresh()

    @property
    def image_id(self) -> str:
        return self.images.current_image_id

    def set_verdict(self, status: DomainStatus) -> None:
        image = self.images.project.image(self.image_id)
        previous = self.images.session.active_domain
        try:
            for domain in SECTION_DOMAINS:
                if domain not in image.applicable_domains:
                    continue
                self.images.set_domain(domain)
                event = self.images.set_status(status, reason_code="section_stage")
                self.images.save(event)
        except ValueError as exc:
            self.status.setText(str(exc))
        finally:
            self.images.set_domain(previous)
        self.status.setText(
            {
                DomainStatus.PASS: "Saved: this section looks fine.",
                DomainStatus.EXCLUDED: "Saved: this whole section is left out of the results.",
                DomainStatus.NOT_REVIEWED: "Verdict cleared.",
            }.get(status, "Saved.")
        )
        self.refresh()
        if self.section_changed is not None:
            self.section_changed()

    def begin_area(self) -> None:
        self._drawing = True
        if self.start_drawing is not None:
            self.start_drawing()
        self.status.setText("")
        self.refresh()

    def cancel_area(self) -> None:
        self._drawing = False
        if self.stop_drawing is not None:
            self.stop_drawing()
        if self.regions_changed is not None:
            self.regions_changed()  # drops the unsaved shape
        self.refresh()

    def save_area(self) -> None:
        geometry = self.drawn_geometry() if self.drawn_geometry is not None else None
        if geometry is None:
            self.status.setText("Draw a closed shape around the area first.")
            return
        try:
            event = self.regions.add_region(
                image_id=self.image_id,
                geometry=geometry,
                domain=Domain.FIBER_SEGMENTATION,
                action=RegionAction.EXCLUDE_ALL_ANALYSIS,
                reason_code=AREA_REASON,
            )
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.regions.save(event)
        self._drawing = False
        if self.stop_drawing is not None:
            self.stop_drawing()
        if self.regions_changed is not None:
            self.regions_changed()
        self.status.setText("Saved: fibers inside this area are left out of the results.")
        self.refresh()
        if self.section_changed is not None:
            self.section_changed()

    def remove_last_area(self) -> None:
        areas = left_out_areas(self.images.session, self.image_id)
        if not areas:
            return
        self.regions.save(self.regions.remove_region(areas[-1].region_id))
        if self.regions_changed is not None:
            self.regions_changed()
        self.status.setText("Removed the last area.")
        self.refresh()
        if self.section_changed is not None:
            self.section_changed()

    def refresh(self) -> None:
        cards = {card.domain: card for card in self.images.cards(self.image_id)}
        flagged = [
            card
            for domain in SECTION_DOMAINS
            if (card := cards.get(domain)) is not None
            and card.applicable
            and card.qc_status in ("review", "fail")
        ]
        if flagged:
            reasons = "; ".join(
                sorted(
                    {
                        part.split(".")[-1].replace("_", " ")
                        for card in flagged
                        for part in card.reason_codes.split("|")
                        if part
                    }
                )
            )
            self.qc.setText(f"Automatic checks suggest a look: {reasons or 'flagged'}.")
        else:
            self.qc.setText("Automatic checks found nothing unusual in this section.")
        verdict = section_verdict(self.images.project, self.images.session, self.image_id)
        self.verdict.setText(
            {
                "fine": "Your verdict: looks fine.",
                "left_out": "Your verdict: the whole section is left out.",
                "not_checked": "Your verdict: not checked yet.",
            }[verdict]
        )
        areas = left_out_areas(self.images.session, self.image_id)
        self.areas.setText(
            f"{len(areas)} area{'s' if len(areas) != 1 else ''} left out in this section."
            if areas
            else ""
        )
        self.areas.setVisible(bool(areas))
        self.remove_area_button.setVisible(bool(areas) and not self._drawing)
        for widget in (self.draw_help, self.save_area_button, self.cancel_area_button):
            widget.setVisible(self._drawing)
        for button in (self.fine_button, self.area_button, self.section_button):
            button.setEnabled(not self._drawing)
        self.clear_button.setVisible(verdict != "not_checked" and not self._drawing)
