"""Controlled values and serializable records for review state."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, TypeVar
from uuid import uuid4

REVIEW_SCHEMA_VERSION = "review_state.v1"
PROJECT_SCHEMA_VERSION = "review_project.v1"
EVENT_SCHEMA_VERSION = "review_event.v1"


class Scope(StrEnum):
    IMAGE = "image"
    REGION = "region"
    OBJECT = "object"


class Domain(StrEnum):
    FIBER_SEGMENTATION = "fiber_segmentation"
    FIBER_TYPING = "fiber_typing"
    NUCLEI = "nuclei"


class NucleiSubdomain(StrEnum):
    SEGMENTATION = "nuclei_segmentation"
    ASSOCIATION = "nucleus_association"


class ReviewMode(StrEnum):
    QC_GATED_AUTOMATIC = "qc_gated_automatic"
    FLAGGED_REVIEW = "flagged_review"
    DOMAIN_FOCUSED = "domain_focused"
    FULL_AUDIT = "full_audit"


class DomainStatus(StrEnum):
    NOT_REVIEWED = "not_reviewed"
    PASS = "pass"
    REVIEW = "review"
    FAIL = "fail"
    EXCLUDED = "excluded"
    NOT_APPLICABLE = "not_applicable"


class ObjectReviewStatus(StrEnum):
    ACCEPTED = "accepted"
    CORRECTED = "corrected"
    UNCERTAIN = "uncertain"
    EXCLUDED = "excluded"
    UNRESOLVED = "unresolved"


class RegionAction(StrEnum):
    """Explicit actions a reviewer can attach to a drawn region."""

    EXCLUDE_DOMAIN = "exclude_domain"
    QUEUE_OBJECTS = "queue_objects"
    DETAILED_REVIEW = "detailed_review"
    IGNORE_NUCLEI = "ignore_nuclei"
    IGNORE_FIBER_TYPING = "ignore_fiber_typing"
    EXCLUDE_ALL_ANALYSIS = "exclude_all_analysis"
    RECOMMEND_RERUN = "recommend_regional_rerun"
    UNRESOLVED = "unresolved"


EnumT = TypeVar("EnumT", bound=StrEnum)


def parse_enum(enum_type: type[EnumT], value: str | EnumT, field_name: str) -> EnumT:
    """Parse a controlled value and raise a contextual validation error."""
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{field_name} must be one of: {choices}; got {value!r}") from exc


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class RegionAnnotation:
    region_id: str
    image_id: str
    geometry: dict[str, Any]
    domain: Domain
    action: str
    reason_code: str
    notes: str = ""
    reviewer: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.region_id.strip():
            raise ValueError("region_id must not be empty")
        if not self.image_id.strip():
            raise ValueError("image_id must not be empty")
        if not isinstance(self.geometry, dict) or "type" not in self.geometry:
            raise ValueError("region geometry must be a GeoJSON-like mapping with a type")
        if not self.action.strip():
            raise ValueError("region action must not be empty")
        object.__setattr__(self, "domain", parse_enum(Domain, self.domain, "region domain"))
        object.__setattr__(
            self, "action", parse_enum(RegionAction, self.action, "region action").value
        )
        if not self.timestamp:
            object.__setattr__(self, "timestamp", utc_now())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["domain"] = self.domain.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegionAnnotation:
        return cls(
            region_id=str(data.get("region_id", "")),
            image_id=str(data.get("image_id", "")),
            geometry=data.get("geometry", {}),
            domain=parse_enum(Domain, data.get("domain", ""), "region domain"),
            action=str(data.get("action", "")),
            reason_code=str(data.get("reason_code", "")),
            notes=str(data.get("notes", "")),
            reviewer=str(data.get("reviewer", "")),
            timestamp=str(data.get("timestamp", "")),
        )


@dataclass(frozen=True)
class ReviewEvent:
    image_id: str
    scope: Scope
    domain: Domain
    target_id: str
    action: str
    reason_code: str = ""
    old_value: Any = None
    new_value: Any = None
    reviewer: str = ""
    timestamp: str = ""
    model_version: str = ""
    qc_version: str = ""
    subdomain: str = ""
    event_id: str = ""
    schema_version: str = EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.image_id.strip():
            raise ValueError("event image_id must not be empty")
        if not self.action.strip():
            raise ValueError("event action must not be empty")
        object.__setattr__(self, "scope", parse_enum(Scope, self.scope, "event scope"))
        object.__setattr__(self, "domain", parse_enum(Domain, self.domain, "event domain"))
        if not self.timestamp:
            object.__setattr__(self, "timestamp", utc_now())
        if not self.event_id:
            object.__setattr__(self, "event_id", str(uuid4()))

    def to_csv_row(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "image_id": self.image_id,
            "scope": self.scope.value,
            "domain": self.domain.value,
            "subdomain": self.subdomain,
            "target_id": self.target_id,
            "action": self.action,
            "reason_code": self.reason_code,
            "old_value": json.dumps(self.old_value, sort_keys=True),
            "new_value": json.dumps(self.new_value, sort_keys=True),
            "reviewer": self.reviewer,
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "qc_version": self.qc_version,
        }


@dataclass(frozen=True)
class FiberTypeDecision:
    image_id: str
    fiber_id: int
    model_fiber_type: str
    reviewed_fiber_type: str
    review_status: ObjectReviewStatus
    queue_source: str = ""
    reason_code: str = ""
    reviewer: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.image_id.strip():
            raise ValueError("fiber-type decision image_id must not be empty")
        if self.fiber_id <= 0:
            raise ValueError("fiber_id must be positive")
        if not self.model_fiber_type.strip():
            raise ValueError("model_fiber_type must not be empty")
        object.__setattr__(
            self,
            "review_status",
            parse_enum(ObjectReviewStatus, self.review_status, "object review_status"),
        )
        if not self.timestamp:
            object.__setattr__(self, "timestamp", utc_now())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["review_status"] = self.review_status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FiberTypeDecision:
        return cls(
            image_id=str(data.get("image_id", "")),
            fiber_id=int(data.get("fiber_id", 0)),
            model_fiber_type=str(data.get("model_fiber_type", "")),
            reviewed_fiber_type=str(data.get("reviewed_fiber_type", "")),
            review_status=parse_enum(
                ObjectReviewStatus,
                data.get("review_status", ""),
                "object review_status",
            ),
            queue_source=str(data.get("queue_source", "")),
            reason_code=str(data.get("reason_code", "")),
            reviewer=str(data.get("reviewer", "")),
            timestamp=str(data.get("timestamp", "")),
        )
