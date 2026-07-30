"""Headless region annotation actions for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from src.review.project import Project
from src.review.schemas import (
    Domain,
    RegionAction,
    RegionAnnotation,
    ReviewEvent,
    Scope,
    parse_enum,
)
from src.review.session import ReviewSession
from src.review.storage import append_review_event, save_regions_geojson, save_session


@dataclass(frozen=True)
class RegionQueueItem:
    """A saved region that explicitly asks for later object-level review."""

    region_id: str
    image_id: str
    domain: Domain
    action: RegionAction
    reason_code: str


class RegionReviewController:
    """Record explicit region actions while keeping project predictions immutable."""

    def __init__(self, project: Project, session: ReviewSession) -> None:
        if project.project_id != session.project_id:
            raise ValueError("Review session does not belong to this project")
        self.project = project
        self.session = session

    @property
    def regions_geojson_path(self):
        return self.project.review_directory / "review_regions.geojson"

    def regions_for_image(self, image_id: str) -> tuple[RegionAnnotation, ...]:
        self.project.image(image_id)
        return tuple(region for region in self.session.regions if region.image_id == image_id)

    def queue(self, image_id: str | None = None) -> tuple[RegionQueueItem, ...]:
        regions = self.session.regions if image_id is None else self.regions_for_image(image_id)
        return tuple(
            RegionQueueItem(
                region_id=region.region_id,
                image_id=region.image_id,
                domain=region.domain,
                action=RegionAction(region.action),
                reason_code=region.reason_code,
            )
            for region in regions
            if RegionAction(region.action)
            in {RegionAction.QUEUE_OBJECTS, RegionAction.DETAILED_REVIEW, RegionAction.UNRESOLVED}
        )

    def coverage_heatmap(
        self,
        image_id: str,
        shape: tuple[int, int],
        *,
        coordinate_scale: int = 1,
    ) -> np.ndarray:
        """Return saved-region coverage for display, never an inferred QC score."""
        if len(shape) != 2 or any(size < 1 for size in shape):
            raise ValueError("heatmap shape must contain two positive dimensions")
        if coordinate_scale < 1:
            raise ValueError("coordinate_scale must be at least 1")
        from skimage.draw import polygon

        heatmap = np.zeros(shape, dtype=np.uint8)
        for region in self.regions_for_image(image_id):
            if region.geometry.get("type") != "Polygon":
                continue
            rings = region.geometry.get("coordinates", [])
            if not rings or len(rings[0]) < 3:
                continue
            points = np.asarray(rings[0], dtype=float)
            rows, columns = polygon(
                points[:, 1] / coordinate_scale,
                points[:, 0] / coordinate_scale,
                shape=shape,
            )
            heatmap[rows, columns] = np.minimum(heatmap[rows, columns] + 1, 255)
        return heatmap

    def add_region(
        self,
        *,
        image_id: str,
        geometry: dict,
        domain: Domain | str,
        action: RegionAction | str,
        reason_code: str = "",
        notes: str = "",
    ) -> ReviewEvent:
        image = self.project.image(image_id)
        parsed_domain = parse_enum(Domain, domain, "region domain")
        if parsed_domain not in image.applicable_domains:
            raise ValueError(f"{parsed_domain.value} is not applicable to image {image_id!r}")
        parsed_action = parse_enum(RegionAction, action, "region action")
        region = RegionAnnotation(
            region_id=str(uuid4()),
            image_id=image_id,
            geometry=geometry,
            domain=parsed_domain,
            action=parsed_action.value,
            reason_code=reason_code,
            notes=notes,
            reviewer=self.session.reviewer,
        )
        self.session.add_region(region)
        self.session.active_scope = Scope.REGION
        self.session.active_domain = parsed_domain
        return ReviewEvent(
            image_id=image_id,
            scope=Scope.REGION,
            domain=parsed_domain,
            target_id=region.region_id,
            action="add_region_annotation",
            reason_code=reason_code,
            new_value=region.to_dict(),
            reviewer=self.session.reviewer,
            model_version=self.session.model_version,
            qc_version=self.session.qc_version,
        )

    def remove_region(self, region_id: str) -> ReviewEvent:
        region = self.session.remove_region(region_id)
        if region is None:
            raise ValueError(f"Unknown region_id: {region_id}")
        return ReviewEvent(
            image_id=region.image_id,
            scope=Scope.REGION,
            domain=region.domain,
            target_id=region.region_id,
            action="remove_region_annotation",
            reason_code=region.reason_code,
            old_value=region.to_dict(),
            reviewer=self.session.reviewer,
            model_version=self.session.model_version,
            qc_version=self.session.qc_version,
        )

    def save(self, event: ReviewEvent | None = None) -> None:
        save_session(self.project.review_state_path, self.session)
        save_regions_geojson(self.regions_geojson_path, self.session.regions)
        if event is not None:
            append_review_event(self.project.review_events_path, event)
