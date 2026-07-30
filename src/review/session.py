"""Serializable shared state for all review workspaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.review.invalidation import EditKind, StaleProduct, invalidated_products
from src.review.schemas import (
    REVIEW_SCHEMA_VERSION,
    Domain,
    DomainStatus,
    FiberTypeDecision,
    NucleusAssociationDecision,
    RegionAnnotation,
    ReviewMode,
    Scope,
    parse_enum,
    utc_now,
)


@dataclass
class ReviewSession:
    project_id: str
    model_version: str
    qc_version: str = ""
    reviewer: str = ""
    current_image_id: str | None = None
    active_domain: Domain = Domain.FIBER_SEGMENTATION
    active_scope: Scope = Scope.IMAGE
    active_review_mode: ReviewMode = ReviewMode.FLAGGED_REVIEW
    active_queue: str = ""
    queue_position: int = 0
    image_statuses: dict[str, dict[str, str]] = field(default_factory=dict)
    regions: list[RegionAnnotation] = field(default_factory=list)
    object_decisions: list[FiberTypeDecision] = field(default_factory=list)
    nucleus_association_decisions: list[NucleusAssociationDecision] = field(default_factory=list)
    reviewed_nucleus_next_ids: dict[str, int] = field(default_factory=dict)
    reviewed_mask_paths: dict[str, dict[str, str]] = field(default_factory=dict)
    stale_products: dict[str, list[str]] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    schema_version: str = REVIEW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.project_id.strip():
            raise ValueError("session project_id must not be empty")
        if self.queue_position < 0:
            raise ValueError("queue_position must be non-negative")
        self.active_domain = parse_enum(Domain, self.active_domain, "active_domain")
        self.active_scope = parse_enum(Scope, self.active_scope, "active_scope")
        self.active_review_mode = parse_enum(
            ReviewMode, self.active_review_mode, "active_review_mode"
        )

    def set_status(
        self,
        image_id: str,
        domain: Domain | str,
        status: DomainStatus | str,
    ) -> None:
        if not image_id.strip():
            raise ValueError("image_id must not be empty")
        parsed_domain = parse_enum(Domain, domain, "status domain")
        parsed_status = parse_enum(DomainStatus, status, "domain status")
        self.image_statuses.setdefault(image_id, {})[parsed_domain.value] = parsed_status.value
        self.touch()

    def get_status(self, image_id: str, domain: Domain | str) -> DomainStatus:
        parsed_domain = parse_enum(Domain, domain, "status domain")
        value = self.image_statuses.get(image_id, {}).get(
            parsed_domain.value, DomainStatus.NOT_REVIEWED.value
        )
        return parse_enum(DomainStatus, value, "stored domain status")

    def add_region(self, region: RegionAnnotation) -> None:
        if any(existing.region_id == region.region_id for existing in self.regions):
            raise ValueError(f"Duplicate region_id: {region.region_id}")
        self.regions.append(region)
        self.touch()

    def remove_region(self, region_id: str) -> RegionAnnotation | None:
        """Remove one saved region so a mistaken annotation is reversible."""
        for index, region in enumerate(self.regions):
            if region.region_id == region_id:
                self.regions.pop(index)
                self.touch()
                return region
        return None

    def record_fiber_type_decision(self, decision: FiberTypeDecision) -> None:
        """Add or update a decision while keeping the original model value immutable."""
        for index, existing in enumerate(self.object_decisions):
            if existing.image_id == decision.image_id and existing.fiber_id == decision.fiber_id:
                if existing.model_fiber_type != decision.model_fiber_type:
                    raise ValueError(
                        "Cannot change model_fiber_type while updating a reviewed decision: "
                        f"{existing.model_fiber_type!r} != {decision.model_fiber_type!r}"
                    )
                self.object_decisions[index] = decision
                self.touch()
                return
        self.object_decisions.append(decision)
        self.touch()

    def remove_fiber_type_decision(self, image_id: str, fiber_id: int) -> FiberTypeDecision | None:
        """Remove one canonical decision for an immediate, reversible UI undo."""
        for index, decision in enumerate(self.object_decisions):
            if decision.image_id == image_id and decision.fiber_id == fiber_id:
                self.object_decisions.pop(index)
                self.touch()
                return decision
        return None

    def record_nucleus_association_decision(self, decision: NucleusAssociationDecision) -> None:
        for index, existing in enumerate(self.nucleus_association_decisions):
            if (
                existing.image_id == decision.image_id
                and existing.nucleus_id == decision.nucleus_id
            ):
                self.nucleus_association_decisions[index] = decision
                self.touch()
                return
        self.nucleus_association_decisions.append(decision)
        self.touch()

    def remove_nucleus_association_decision(
        self, image_id: str, nucleus_id: int
    ) -> NucleusAssociationDecision | None:
        """Remove a superseded decision when its reviewed nucleus is deleted."""
        for index, decision in enumerate(self.nucleus_association_decisions):
            if decision.image_id == image_id and decision.nucleus_id == nucleus_id:
                self.nucleus_association_decisions.pop(index)
                self.touch()
                return decision
        return None

    def allocate_reviewed_nucleus_id(self, image_id: str, observed_max_id: int) -> int:
        """Allocate an ID that will not be reused after later reviewed deletions."""
        if observed_max_id < 0:
            raise ValueError("observed_max_id must not be negative")
        next_id = max(observed_max_id, self.reviewed_nucleus_next_ids.get(image_id, 0)) + 1
        self.reviewed_nucleus_next_ids[image_id] = next_id
        self.touch()
        return next_id

    def mark_stale(self, image_id: str, edit_kind: EditKind | str) -> frozenset[StaleProduct]:
        products = invalidated_products(edit_kind)
        existing = set(self.stale_products.get(image_id, []))
        existing.update(product.value for product in products)
        self.stale_products[image_id] = sorted(existing)
        self.touch()
        return products

    def clear_stale(self, image_id: str, products: set[StaleProduct | str] | None = None) -> None:
        if products is None:
            self.stale_products.pop(image_id, None)
        else:
            values = {
                product.value if isinstance(product, StaleProduct) else StaleProduct(product).value
                for product in products
            }
            remaining = [
                product
                for product in self.stale_products.get(image_id, [])
                if product not in values
            ]
            if remaining:
                self.stale_products[image_id] = remaining
            else:
                self.stale_products.pop(image_id, None)
        self.touch()

    def set_queue(self, queue_name: str, position: int = 0) -> None:
        if position < 0:
            raise ValueError("queue position must be non-negative")
        self.active_queue = queue_name
        self.queue_position = position
        self.touch()

    def touch(self) -> None:
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "model_version": self.model_version,
            "qc_version": self.qc_version,
            "reviewer": self.reviewer,
            "current_image_id": self.current_image_id,
            "active_domain": self.active_domain.value,
            "active_scope": self.active_scope.value,
            "active_review_mode": self.active_review_mode.value,
            "active_queue": self.active_queue,
            "queue_position": self.queue_position,
            "image_statuses": self.image_statuses,
            "regions": [region.to_dict() for region in self.regions],
            "object_decisions": [decision.to_dict() for decision in self.object_decisions],
            "nucleus_association_decisions": [
                decision.to_dict() for decision in self.nucleus_association_decisions
            ],
            "reviewed_nucleus_next_ids": self.reviewed_nucleus_next_ids,
            "reviewed_mask_paths": self.reviewed_mask_paths,
            "stale_products": self.stale_products,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewSession:
        if not isinstance(data, dict):
            raise ValueError("Review state must be a JSON object")
        schema_version = data.get("schema_version")
        if schema_version != REVIEW_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported review schema_version {schema_version!r}; "
                f"expected {REVIEW_SCHEMA_VERSION!r}"
            )
        image_statuses = data.get("image_statuses", {})
        if not isinstance(image_statuses, dict):
            raise ValueError("image_statuses must be a mapping")
        normalized_statuses: dict[str, dict[str, str]] = {}
        for image_id, statuses in image_statuses.items():
            if not isinstance(statuses, dict):
                raise ValueError(f"image_statuses[{image_id!r}] must be a mapping")
            normalized_statuses[str(image_id)] = {}
            for domain, status in statuses.items():
                parsed_domain = parse_enum(Domain, domain, "stored status domain")
                parsed_status = parse_enum(DomainStatus, status, "stored domain status")
                normalized_statuses[str(image_id)][parsed_domain.value] = parsed_status.value

        raw_regions = data.get("regions", [])
        if not isinstance(raw_regions, list):
            raise ValueError("regions must be a list")
        stale_products = data.get("stale_products", {})
        if not isinstance(stale_products, dict):
            raise ValueError("stale_products must be a mapping")
        normalized_stale: dict[str, list[str]] = {}
        for image_id, products in stale_products.items():
            if not isinstance(products, list):
                raise ValueError(f"stale_products[{image_id!r}] must be a list")
            normalized_stale[str(image_id)] = sorted(
                {StaleProduct(product).value for product in products}
            )

        object_decisions = data.get("object_decisions", [])
        nucleus_association_decisions = data.get("nucleus_association_decisions", [])
        reviewed_nucleus_next_ids = data.get("reviewed_nucleus_next_ids", {})
        reviewed_mask_paths = data.get("reviewed_mask_paths", {})
        if not isinstance(object_decisions, list):
            raise ValueError("object_decisions must be a list")
        if not isinstance(nucleus_association_decisions, list):
            raise ValueError("nucleus_association_decisions must be a list")
        if not isinstance(reviewed_nucleus_next_ids, dict):
            raise ValueError("reviewed_nucleus_next_ids must be a mapping")
        normalized_next_ids: dict[str, int] = {}
        for image_id, value in reviewed_nucleus_next_ids.items():
            numeric_value = int(value)
            if numeric_value < 0:
                raise ValueError(f"reviewed_nucleus_next_ids[{image_id!r}] must not be negative")
            normalized_next_ids[str(image_id)] = numeric_value
        if not isinstance(reviewed_mask_paths, dict):
            raise ValueError("reviewed_mask_paths must be a mapping")
        return cls(
            schema_version=REVIEW_SCHEMA_VERSION,
            project_id=str(data.get("project_id", "")),
            model_version=str(data.get("model_version", "")),
            qc_version=str(data.get("qc_version", "")),
            reviewer=str(data.get("reviewer", "")),
            current_image_id=data.get("current_image_id"),
            active_domain=parse_enum(Domain, data.get("active_domain", ""), "active_domain"),
            active_scope=parse_enum(Scope, data.get("active_scope", ""), "active_scope"),
            active_review_mode=parse_enum(
                ReviewMode, data.get("active_review_mode", ""), "active_review_mode"
            ),
            active_queue=str(data.get("active_queue", "")),
            queue_position=int(data.get("queue_position", 0)),
            image_statuses=normalized_statuses,
            regions=[RegionAnnotation.from_dict(region) for region in raw_regions],
            object_decisions=[
                FiberTypeDecision.from_dict(decision) for decision in object_decisions
            ],
            nucleus_association_decisions=[
                NucleusAssociationDecision.from_dict(decision)
                for decision in nucleus_association_decisions
            ],
            reviewed_nucleus_next_ids=normalized_next_ids,
            reviewed_mask_paths=dict(reviewed_mask_paths),
            stale_products=normalized_stale,
            created_at=str(data.get("created_at", "")) or utc_now(),
            updated_at=str(data.get("updated_at", "")) or utc_now(),
        )
