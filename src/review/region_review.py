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
    RegionKind,
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


@dataclass(frozen=True)
class RegionAssignment:
    """One centroid's relationship to the named analysis ROIs in its image."""

    object_id: int
    status: str
    region_id: str = ""
    region_name: str = ""
    region_role: str = ""


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

    def analysis_rois(self, image_id: str) -> tuple[RegionAnnotation, ...]:
        return tuple(
            region
            for region in self.regions_for_image(image_id)
            if region.kind is RegionKind.ANALYSIS_ROI
        )

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
        kind: RegionKind | str | None = None,
    ) -> np.ndarray:
        """Return saved-region coverage for display, never an inferred QC score."""
        if len(shape) != 2 or any(size < 1 for size in shape):
            raise ValueError("heatmap shape must contain two positive dimensions")
        if coordinate_scale < 1:
            raise ValueError("coordinate_scale must be at least 1")
        from skimage.draw import polygon

        parsed_kind = None if kind is None else parse_enum(RegionKind, kind, "region kind")
        heatmap = np.zeros(shape, dtype=np.uint8)
        for region in self.regions_for_image(image_id):
            if parsed_kind is not None and region.kind is not parsed_kind:
                continue
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

    def assign_centroids(
        self,
        image_id: str,
        centroids: dict[int, tuple[float, float]],
    ) -> tuple[RegionAssignment, ...]:
        """Assign centroids without silently resolving overlap or boundary cases."""
        rois = self.analysis_rois(image_id)
        assignments: list[RegionAssignment] = []
        for object_id, (x, y) in sorted(centroids.items()):
            matches = [_point_in_polygon(x, y, roi.geometry) for roi in rois]
            if any(match == "boundary" for match in matches):
                assignments.append(RegionAssignment(object_id, "boundary"))
                continue
            contained = [roi for roi, match in zip(rois, matches, strict=True) if match == "inside"]
            if not contained:
                assignments.append(RegionAssignment(object_id, "outside"))
            elif len(contained) > 1:
                assignments.append(RegionAssignment(object_id, "ambiguous"))
            else:
                roi = contained[0]
                assignments.append(
                    RegionAssignment(object_id, "assigned", roi.region_id, roi.name, roi.role)
                )
        return tuple(assignments)

    def add_region(
        self,
        *,
        image_id: str,
        geometry: dict,
        domain: Domain | str,
        action: RegionAction | str,
        reason_code: str = "",
        notes: str = "",
        kind: RegionKind | str = RegionKind.REVIEW,
        name: str = "",
        role: str = "",
    ) -> ReviewEvent:
        image = self.project.image(image_id)
        parsed_domain = parse_enum(Domain, domain, "region domain")
        if parsed_domain not in image.applicable_domains:
            raise ValueError(f"{parsed_domain.value} is not applicable to image {image_id!r}")
        parsed_action = parse_enum(RegionAction, action, "region action")
        parsed_kind = parse_enum(RegionKind, kind, "region kind")
        if parsed_kind is RegionKind.ANALYSIS_ROI:
            parsed_action = RegionAction.ANALYSIS_ROI
        region = RegionAnnotation(
            region_id=str(uuid4()),
            image_id=image_id,
            geometry=geometry,
            domain=parsed_domain,
            action=parsed_action.value,
            reason_code=reason_code,
            kind=parsed_kind,
            name=name,
            role=role,
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


def _point_in_polygon(x: float, y: float, geometry: dict) -> str:
    """Classify a point against the outer ring of a simple GeoJSON polygon."""
    if geometry.get("type") != "Polygon" or not geometry.get("coordinates"):
        return "outside"
    ring = geometry["coordinates"][0]
    if len(ring) < 3:
        return "outside"
    inside = False
    for start, end in zip(ring, ring[1:] + ring[:1], strict=False):
        x1, y1 = start
        x2, y2 = end
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if (
            abs(cross) < 1e-9
            and min(x1, x2) <= x <= max(x1, x2)
            and min(y1, y2) <= y <= max(y1, y2)
        ):
            return "boundary"
        if (y1 > y) != (y2 > y):
            intercept = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < intercept:
                inside = not inside
    return "inside" if inside else "outside"
