"""Headless reviewed-nuclei masks, association decisions, and queues for Phase 6."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from src.review.invalidation import EditKind
from src.review.project import Project
from src.review.schemas import (
    Domain,
    NucleusAssociationDecision,
    NucleusAssociationStatus,
    ReviewEvent,
    Scope,
)
from src.review.session import ReviewSession
from src.review.storage import (
    append_review_event,
    atomic_write_dataframe,
    materialize_reviewed_mask,
    save_session,
)


class NucleusQueueSource(StrEnum):
    UNASSIGNED = "unassigned_nuclei"
    AMBIGUOUS = "ambiguous_associations"
    FULL = "full_nuclei_audit"


@dataclass(frozen=True)
class NucleusQueueItem:
    image_id: str
    nucleus_id: int
    model_fiber_id: int
    assignment_status: str
    queue_source: NucleusQueueSource


class NuclearReviewController:
    """Keep predicted masks read-only while recording reviewed nuclear work."""

    def __init__(self, project: Project, session: ReviewSession) -> None:
        if session.project_id != project.project_id:
            raise ValueError("Review session does not belong to this project")
        self.project = project
        self.session = session
        self.queue: tuple[NucleusQueueItem, ...] = ()

    @property
    def reviewed_associations_path(self) -> Path:
        return self.project.review_directory / "reviewed_nucleus_associations.csv"

    def _predicted_mask_path(self, image_id: str) -> Path:
        path = self.project.image(image_id).outputs.get("nuclei_labels")
        if path is None:
            raise ValueError(f"Image {image_id!r} has no nuclei_labels artifact")
        return path

    def nuclei_mask_path(self, image_id: str) -> Path:
        stored = self.session.reviewed_mask_paths.get(image_id, {}).get(Domain.NUCLEI.value)
        return Path(stored) if stored else self._predicted_mask_path(image_id)

    def load_nuclei_labels(self, image_id: str) -> np.ndarray:
        return np.asarray(tifffile.imread(self.nuclei_mask_path(image_id)), dtype=np.int32)

    def ensure_reviewed_mask(self, image_id: str) -> Path:
        predicted = self._predicted_mask_path(image_id)
        reviewed = self.project.reviewed_mask_path(image_id, Domain.NUCLEI)
        materialize_reviewed_mask(predicted, reviewed)
        self.session.reviewed_mask_paths.setdefault(image_id, {})[Domain.NUCLEI.value] = str(
            reviewed
        )
        self.session.touch()
        return reviewed

    def delete_nucleus(
        self, image_id: str, nucleus_id: int, *, reason_code: str = ""
    ) -> ReviewEvent:
        if nucleus_id <= 0:
            raise ValueError("nucleus_id must be positive")
        reviewed = self.ensure_reviewed_mask(image_id)
        labels = np.asarray(tifffile.imread(reviewed), dtype=np.int32)
        pixel_count = int(np.count_nonzero(labels == nucleus_id))
        if pixel_count == 0:
            raise ValueError(f"nucleus_id {nucleus_id} is not present in {image_id!r}")
        labels[labels == nucleus_id] = 0
        _atomic_write_labels(reviewed, labels)
        removed_association = self.session.remove_nucleus_association_decision(image_id, nucleus_id)
        self.session.mark_stale(image_id, EditKind.NUCLEUS_MASK)
        return ReviewEvent(
            image_id=image_id,
            scope=Scope.OBJECT,
            domain=Domain.NUCLEI,
            target_id=str(nucleus_id),
            action="delete_nucleus",
            reason_code=reason_code,
            old_value={
                "pixel_count": pixel_count,
                "association_decision": removed_association.to_dict()
                if removed_association is not None
                else None,
            },
            new_value={"pixel_count": 0, "reviewed_mask": str(reviewed)},
            reviewer=self.session.reviewer,
            model_version=self.session.model_version,
            qc_version=self.session.qc_version,
        )

    def add_nucleus(
        self,
        image_id: str,
        pixels: np.ndarray,
        *,
        reason_code: str = "",
    ) -> ReviewEvent:
        """Add one painted nucleus to the copy-on-write reviewed mask.

        ``pixels`` must be a boolean mask in the nucleus-label image coordinate
        system.  It cannot overlap an existing reviewed nucleus.
        """
        reviewed = self.ensure_reviewed_mask(image_id)
        labels = np.asarray(tifffile.imread(reviewed), dtype=np.int32)
        painted = np.asarray(pixels)
        if painted.dtype != np.bool_:
            raise ValueError("pixels must be a boolean mask")
        if painted.shape != labels.shape:
            raise ValueError(
                f"pixels shape {painted.shape} does not match nucleus mask shape {labels.shape}"
            )
        pixel_count = int(np.count_nonzero(painted))
        if pixel_count == 0:
            raise ValueError("pixels must contain at least one painted pixel")
        if np.any(labels[painted] != 0):
            raise ValueError("added nucleus pixels must not overlap an existing reviewed nucleus")
        nucleus_id = self.session.allocate_reviewed_nucleus_id(image_id, int(labels.max(initial=0)))
        labels[painted] = nucleus_id
        _atomic_write_labels(reviewed, labels)
        self.session.mark_stale(image_id, EditKind.NUCLEUS_MASK)
        return ReviewEvent(
            image_id=image_id,
            scope=Scope.OBJECT,
            domain=Domain.NUCLEI,
            target_id=str(nucleus_id),
            action="add_nucleus",
            reason_code=reason_code,
            old_value=None,
            new_value={
                "nucleus_id": nucleus_id,
                "pixel_count": pixel_count,
                "reviewed_mask": str(reviewed),
            },
            reviewer=self.session.reviewer,
            model_version=self.session.model_version,
            qc_version=self.session.qc_version,
        )

    def association_queue(
        self,
        image_id: str,
        source: NucleusQueueSource | str,
    ) -> tuple[NucleusQueueItem, ...]:
        parsed_source = NucleusQueueSource(source)
        table_path = self.project.image(image_id).outputs.get("nuclei_table")
        if table_path is None:
            raise ValueError(f"Image {image_id!r} has no nuclei_table artifact")
        table = self.reviewed_associations(image_id)
        required = {"nucleus_id", "assigned_fiber_id", "assignment_status"}
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"nuclei table is missing required columns: {missing}")
        work = table.copy()
        if parsed_source is NucleusQueueSource.UNASSIGNED:
            work = work[
                work["reviewed_association_status"].eq(NucleusAssociationStatus.UNASSIGNED.value)
            ]
        elif parsed_source is NucleusQueueSource.AMBIGUOUS:
            work = work[
                work["reviewed_association_status"].eq(NucleusAssociationStatus.AMBIGUOUS.value)
            ]
        return tuple(
            NucleusQueueItem(
                image_id=image_id,
                nucleus_id=int(row.nucleus_id),
                model_fiber_id=_positive_int(row.assigned_fiber_id),
                assignment_status=str(row.reviewed_association_status),
                queue_source=parsed_source,
            )
            for row in work.sort_values("nucleus_id").itertuples(index=False)
            if int(row.nucleus_id) > 0
        )

    @property
    def current_item(self) -> NucleusQueueItem | None:
        return self.queue[self.session.queue_position] if self.queue else None

    def set_association_queue(
        self,
        image_id: str,
        source: NucleusQueueSource | str,
        *,
        position: int = 0,
    ) -> None:
        """Select a deterministic per-image association queue and persist its position."""
        parsed_source = NucleusQueueSource(source)
        self.queue = self.association_queue(image_id, parsed_source)
        self.session.current_image_id = image_id
        self.session.active_domain = Domain.NUCLEI
        self.session.active_scope = Scope.OBJECT
        self.session.set_queue(
            f"nuclei:{image_id}:{parsed_source.value}",
            position=min(position, len(self.queue) - 1) if self.queue else 0,
        )

    def move(self, offset: int) -> NucleusQueueItem:
        if not self.queue:
            raise ValueError("The nucleus association queue is empty")
        self.session.set_queue(
            self.session.active_queue,
            (self.session.queue_position + offset) % len(self.queue),
        )
        return self.queue[self.session.queue_position]

    def reviewed_associations(self, image_id: str) -> pd.DataFrame:
        """Return model associations overlaid with canonical manual decisions.

        Model columns are retained unchanged.  Reviewed columns make the effective
        association explicit without overwriting prediction outputs.
        """
        table_path = self.project.image(image_id).outputs.get("nuclei_table")
        if table_path is None:
            raise ValueError(f"Image {image_id!r} has no nuclei_table artifact")
        table = pd.read_csv(table_path, low_memory=False)
        required = {"nucleus_id", "assigned_fiber_id", "assignment_status"}
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"nuclei table is missing required columns: {missing}")
        if table["nucleus_id"].duplicated().any():
            raise ValueError(f"nuclei table has duplicate nucleus IDs: {table_path}")

        present_ids = set(np.unique(self.load_nuclei_labels(image_id)).tolist()) - {0}
        table = table[table["nucleus_id"].map(_positive_int).isin(present_ids)].copy()
        table["model_row_available"] = True
        known_ids = set(table["nucleus_id"].map(_positive_int))
        missing_ids = sorted(present_ids - known_ids)
        if missing_ids:
            additions = pd.DataFrame(
                {
                    "nucleus_id": missing_ids,
                    "assigned_fiber_id": pd.NA,
                    "assignment_status": pd.NA,
                    "model_row_available": False,
                }
            )
            table = pd.concat([table, additions], ignore_index=True, sort=False)
        decisions = {
            decision.nucleus_id: decision
            for decision in self.session.nucleus_association_decisions
            if decision.image_id == image_id and decision.nucleus_id in present_ids
        }
        reviewed_ids: list[int] = []
        reviewed_statuses: list[str] = []
        reason_codes: list[str] = []
        reviewers: list[str] = []
        timestamps: list[str] = []
        for row in table.itertuples(index=False):
            nucleus_id = _positive_int(row.nucleus_id)
            decision = decisions.get(nucleus_id)
            if decision is None:
                reviewed_ids.append(_positive_int(row.assigned_fiber_id))
                reviewed_statuses.append(_model_association_status(row.assignment_status).value)
                reason_codes.append("")
                reviewers.append("")
                timestamps.append("")
            else:
                reviewed_ids.append(decision.reviewed_fiber_id)
                reviewed_statuses.append(decision.association_status.value)
                reason_codes.append(decision.reason_code)
                reviewers.append(decision.reviewer)
                timestamps.append(decision.timestamp)
        table["reviewed_fiber_id"] = reviewed_ids
        table["reviewed_association_status"] = reviewed_statuses
        table["review_reason_code"] = reason_codes
        table["review_reviewer"] = reviewers
        table["review_timestamp"] = timestamps
        return table

    def set_association(
        self,
        image_id: str,
        nucleus_id: int,
        *,
        fiber_id: int = 0,
        status: NucleusAssociationStatus | str = NucleusAssociationStatus.ASSIGNED,
        reason_code: str = "",
    ) -> ReviewEvent:
        labels = self.load_nuclei_labels(image_id)
        if not np.any(labels == nucleus_id):
            raise ValueError(f"nucleus_id {nucleus_id} is not present in {image_id!r}")
        parsed_status = NucleusAssociationStatus(status)
        if parsed_status is NucleusAssociationStatus.ASSIGNED:
            fibers_path = self.project.image(image_id).outputs.get("fiber_labels")
            if fibers_path is None:
                raise ValueError(f"Image {image_id!r} has no fiber_labels artifact")
            fibers = np.asarray(tifffile.imread(fibers_path), dtype=np.int32)
            if not np.any(fibers == fiber_id):
                raise ValueError(f"fiber_id {fiber_id} is not present in {image_id!r}")
        else:
            fiber_id = 0
        decision = NucleusAssociationDecision(
            image_id=image_id,
            nucleus_id=nucleus_id,
            reviewed_fiber_id=fiber_id,
            association_status=parsed_status,
            reason_code=reason_code,
            reviewer=self.session.reviewer,
        )
        previous = next(
            (
                item
                for item in self.session.nucleus_association_decisions
                if item.image_id == image_id and item.nucleus_id == nucleus_id
            ),
            None,
        )
        self.session.record_nucleus_association_decision(decision)
        self.session.mark_stale(image_id, EditKind.NUCLEUS_REASSIGNMENT)
        return ReviewEvent(
            image_id=image_id,
            scope=Scope.OBJECT,
            domain=Domain.NUCLEI,
            target_id=str(nucleus_id),
            action="set_nucleus_association",
            reason_code=reason_code,
            old_value=previous.to_dict() if previous else None,
            new_value=decision.to_dict(),
            reviewer=self.session.reviewer,
            model_version=self.session.model_version,
            qc_version=self.session.qc_version,
            subdomain="nucleus_association",
        )

    def save(self, event: ReviewEvent | None = None) -> None:
        save_session(self.project.review_state_path, self.session)
        tables = [
            self.reviewed_associations(image.image_id)
            for image in self.project.images
            if "nuclei_table" in image.outputs and "nuclei_labels" in image.outputs
        ]
        rows = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        atomic_write_dataframe(self.reviewed_associations_path, rows)
        if event is not None:
            append_review_event(self.project.review_events_path, event)


def _atomic_write_labels(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tif"
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        tifffile.imwrite(temporary_path, labels)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _positive_int(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _model_association_status(value: object) -> NucleusAssociationStatus:
    normalized = str(value).strip().lower()
    if normalized == NucleusAssociationStatus.ASSIGNED.value:
        return NucleusAssociationStatus.ASSIGNED
    if normalized == NucleusAssociationStatus.AMBIGUOUS.value:
        return NucleusAssociationStatus.AMBIGUOUS
    if "unassigned" in normalized or "interstitial" in normalized:
        return NucleusAssociationStatus.UNASSIGNED
    return NucleusAssociationStatus.UNRESOLVED
