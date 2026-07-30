"""Headless image-level review actions and navigation for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.review.project import Project
from src.review.schemas import Domain, DomainStatus, ReviewEvent, ReviewMode, Scope, parse_enum
from src.review.session import ReviewSession
from src.review.storage import append_review_event, save_session


@dataclass(frozen=True)
class ImageDomainCard:
    """QC and reviewer state shown for one image/domain card."""

    image_id: str
    domain: Domain
    applicable: bool
    qc_status: str
    review_status: DomainStatus
    reason_codes: str
    review_priority: int


class ImageReviewController:
    """Own image-level review state without importing Qt or Napari.

    Phase 3 intentionally records only image/domain decisions. Object queues,
    regions, and mask edits remain owned by their later phases.
    """

    def __init__(self, project: Project, image_qc: pd.DataFrame, session: ReviewSession) -> None:
        if session.project_id != project.project_id:
            raise ValueError("Review session does not belong to this project")
        required = {"image_id", "domain", "applicable", "status", "reason_codes", "review_priority"}
        missing = sorted(required - set(image_qc.columns))
        if missing:
            raise ValueError(f"image QC is missing required columns: {missing}")
        self.project = project
        self.image_qc = image_qc.copy()
        self.session = session
        known_ids = {image.image_id for image in project.images}
        if not set(self.image_qc["image_id"].astype(str)).issubset(known_ids):
            raise ValueError("image QC contains image IDs outside the project")
        if session.current_image_id not in known_ids:
            session.current_image_id = project.images[0].image_id
            session.touch()

    @property
    def current_image_id(self) -> str:
        assert self.session.current_image_id is not None
        return self.session.current_image_id

    def cards(self, image_id: str | None = None) -> tuple[ImageDomainCard, ...]:
        selected_id = image_id or self.current_image_id
        image = self.project.image(selected_id)
        rows = self.image_qc[self.image_qc["image_id"].astype(str).eq(selected_id)]
        cards: list[ImageDomainCard] = []
        for domain in Domain:
            row = rows[rows["domain"].astype(str).eq(domain.value)]
            if row.empty:
                raise ValueError(f"image QC is missing {selected_id!r}/{domain.value!r}")
            value = row.iloc[0]
            applicable = domain in image.applicable_domains
            cards.append(
                ImageDomainCard(
                    image_id=selected_id,
                    domain=domain,
                    applicable=applicable,
                    qc_status=str(value["status"]),
                    review_status=self.session.get_status(selected_id, domain),
                    reason_codes=(
                        "" if pd.isna(value["reason_codes"]) else str(value["reason_codes"])
                    ),
                    review_priority=int(value["review_priority"]),
                )
            )
        return tuple(cards)

    def set_image(self, image_id: str) -> None:
        self.project.image(image_id)
        self.session.current_image_id = image_id
        self.session.touch()

    def set_domain(self, domain: Domain | str) -> None:
        self.session.active_domain = parse_enum(Domain, domain, "active domain")
        self.session.touch()

    def set_mode(self, mode: ReviewMode | str) -> None:
        self.session.active_review_mode = parse_enum(ReviewMode, mode, "review mode")
        self.session.touch()

    def navigation_ids(self) -> tuple[str, ...]:
        """Return deterministic image order appropriate for the selected review mode."""
        mode = self.session.active_review_mode
        if mode in {ReviewMode.DOMAIN_FOCUSED, ReviewMode.FULL_AUDIT}:
            return tuple(
                image.image_id
                for image in self.project.images
                if self.session.active_domain in image.applicable_domains
            )
        rows = self.image_qc[
            self.image_qc["domain"].astype(str).eq(self.session.active_domain.value)
            & self.image_qc["applicable"].astype(bool)
        ]
        if mode is ReviewMode.QC_GATED_AUTOMATIC:
            rows = rows[rows["status"].astype(str).isin({"review", "fail"})]
        else:  # flagged_review
            rows = rows[
                rows["status"].astype(str).eq("review") | rows["review_priority"].astype(int).gt(0)
            ]
        candidates = set(rows["image_id"].astype(str))
        return tuple(
            image.image_id for image in self.project.images if image.image_id in candidates
        )

    def move(self, offset: int) -> str:
        ids = self.navigation_ids()
        if not ids:
            raise ValueError("No images match the current review mode and domain")
        try:
            index = ids.index(self.current_image_id)
        except ValueError:
            index = 0 if offset >= 0 else len(ids) - 1
        self.set_image(ids[(index + offset) % len(ids)])
        return self.current_image_id

    def set_status(self, status: DomainStatus | str, *, reason_code: str = "") -> ReviewEvent:
        domain = self.session.active_domain
        image = self.project.image(self.current_image_id)
        parsed_status = parse_enum(DomainStatus, status, "image status")
        if (
            domain not in image.applicable_domains
            and parsed_status is not DomainStatus.NOT_APPLICABLE
        ):
            raise ValueError(f"{domain.value} is not applicable to image {image.image_id!r}")
        old_status = self.session.get_status(image.image_id, domain)
        self.session.set_status(image.image_id, domain, parsed_status)
        return ReviewEvent(
            image_id=image.image_id,
            scope=Scope.IMAGE,
            domain=domain,
            target_id=image.image_id,
            action="set_domain_status",
            reason_code=reason_code,
            old_value=old_status.value,
            new_value=parsed_status.value,
            reviewer=self.session.reviewer,
            model_version=self.session.model_version,
            qc_version=self.session.qc_version,
        )

    def save(self, event: ReviewEvent | None = None) -> None:
        """Persist the snapshot and optional audit event through shared storage."""
        save_session(self.project.review_state_path, self.session)
        if event is not None:
            append_review_event(self.project.review_events_path, event)
