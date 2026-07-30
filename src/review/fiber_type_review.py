"""Canonical object-level fiber-type decisions for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.fiber_type_labels import REVIEW_TYPES, normalize_review_label
from src.review.project import Project
from src.review.queues import FiberQueueItem
from src.review.schemas import Domain, FiberTypeDecision, ObjectReviewStatus, ReviewEvent, Scope
from src.review.session import ReviewSession
from src.review.storage import append_review_event, save_session


def _legacy_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class FiberTypeDecisionUndo:
    item: FiberQueueItem
    queue_position: int
    previous: FiberTypeDecision | None
    decision: FiberTypeDecision


class FiberTypeReviewController:
    def __init__(self, session: ReviewSession, queue: tuple[FiberQueueItem, ...] = ()) -> None:
        self.session = session
        self.queue = queue
        self._last_undo: FiberTypeDecisionUndo | None = None
        self.session.active_domain = Domain.FIBER_TYPING
        self.session.active_scope = Scope.OBJECT

    @property
    def current_item(self) -> FiberQueueItem | None:
        return self.queue[self.session.queue_position] if self.queue else None

    def set_queue(
        self,
        queue: tuple[FiberQueueItem, ...],
        name: str,
        *,
        position: int = 0,
    ) -> None:
        self.queue = queue
        if queue:
            position = min(position, len(queue) - 1)
        else:
            position = 0
        self.session.set_queue(name, position=position)
        self.session.active_domain = Domain.FIBER_TYPING
        self.session.active_scope = Scope.OBJECT
        self._last_undo = None

    def move(self, offset: int) -> FiberQueueItem:
        if not self.queue:
            raise ValueError("The fiber-type queue is empty")
        self.session.set_queue(
            self.session.active_queue, (self.session.queue_position + offset) % len(self.queue)
        )
        return self.queue[self.session.queue_position]

    def decide(
        self,
        reviewed_type: str | None = None,
        *,
        status: ObjectReviewStatus | str | None = None,
        reason_code: str = "",
    ) -> ReviewEvent:
        item = self.current_item
        if item is None:
            raise ValueError("Select a non-empty fiber-type queue before recording a decision")
        normalized = (
            item.model_fiber_type
            if reviewed_type is None
            else normalize_review_label(reviewed_type)
        )
        if normalized not in REVIEW_TYPES and normalized != item.model_fiber_type:
            raise ValueError(f"reviewed_type must be one of: {', '.join(REVIEW_TYPES)}")
        if status is None:
            status = (
                ObjectReviewStatus.EXCLUDED
                if normalized == "exclude"
                else ObjectReviewStatus.UNCERTAIN
                if normalized == "uncertain"
                else ObjectReviewStatus.ACCEPTED
                if normalized == item.model_fiber_type
                else ObjectReviewStatus.CORRECTED
            )
        parsed_status = ObjectReviewStatus(status)
        decision = FiberTypeDecision(
            image_id=item.image_id,
            fiber_id=item.fiber_id,
            model_fiber_type=item.model_fiber_type,
            reviewed_fiber_type=normalized,
            review_status=parsed_status,
            queue_source=item.queue_source.value,
            reason_code=reason_code or item.reason_code,
            reviewer=self.session.reviewer,
        )
        previous = next(
            (
                value
                for value in self.session.object_decisions
                if value.image_id == item.image_id and value.fiber_id == item.fiber_id
            ),
            None,
        )
        self.session.record_fiber_type_decision(decision)
        self._last_undo = FiberTypeDecisionUndo(
            item=item,
            queue_position=self.session.queue_position,
            previous=previous,
            decision=decision,
        )
        return ReviewEvent(
            image_id=item.image_id,
            scope=Scope.OBJECT,
            domain=Domain.FIBER_TYPING,
            target_id=str(item.fiber_id),
            action="record_fiber_type_decision",
            reason_code=decision.reason_code,
            old_value=previous.to_dict() if previous else None,
            new_value=decision.to_dict(),
            reviewer=self.session.reviewer,
            model_version=self.session.model_version,
            qc_version=self.session.qc_version,
        )

    def undo_last_decision(self) -> ReviewEvent:
        """Restore the prior decision, if any, and return an audit event."""
        undo = self._last_undo
        if undo is None:
            raise ValueError("There is no decision available to undo")
        current = self.session.remove_fiber_type_decision(undo.item.image_id, undo.item.fiber_id)
        if current is None:
            raise ValueError("The decision available to undo is no longer present")
        if undo.previous is not None:
            self.session.record_fiber_type_decision(undo.previous)
        self.session.set_queue(self.session.active_queue, undo.queue_position)
        self._last_undo = None
        return ReviewEvent(
            image_id=undo.item.image_id,
            scope=Scope.OBJECT,
            domain=Domain.FIBER_TYPING,
            target_id=str(undo.item.fiber_id),
            action="undo_fiber_type_decision",
            old_value=current.to_dict(),
            new_value=undo.previous.to_dict() if undo.previous else None,
            reviewer=self.session.reviewer,
            model_version=self.session.model_version,
            qc_version=self.session.qc_version,
        )

    def save(self, project: Project, event: ReviewEvent | None = None) -> None:
        save_session(project.review_state_path, self.session)
        if event is not None:
            append_review_event(project.review_events_path, event)


def load_legacy_fiber_type_decisions(
    path: Path | str,
    *,
    image_id: str,
) -> tuple[FiberTypeDecision, ...]:
    """Adapt the legacy Napari CSV without treating it as the canonical format."""
    table = pd.read_csv(path, low_memory=False)
    id_column = "fiber_id" if "fiber_id" in table.columns else "label"
    if id_column not in table.columns:
        raise ValueError("Legacy review CSV must contain fiber_id or label")
    prediction_column = next(
        (
            name
            for name in ("predicted_type", "fiber_type", "model_prediction")
            if name in table.columns
        ),
        None,
    )
    if prediction_column is None:
        raise ValueError("Legacy review CSV must retain a model prediction column")
    decisions: list[FiberTypeDecision] = []
    for _, row in table.iterrows():
        corrected = normalize_review_label(row.get("corrected_type", ""))
        excluded = _legacy_bool(row.get("is_excluded", False))
        uncertain = _legacy_bool(row.get("is_uncertain", False))
        if excluded:
            corrected = "exclude"
        elif uncertain:
            corrected = "uncertain"
        if not corrected:
            continue
        model = normalize_review_label(row[prediction_column])
        status = (
            ObjectReviewStatus.EXCLUDED
            if corrected == "exclude"
            else ObjectReviewStatus.UNCERTAIN
            if corrected == "uncertain"
            else ObjectReviewStatus.ACCEPTED
            if corrected == model
            else ObjectReviewStatus.CORRECTED
        )
        decisions.append(
            FiberTypeDecision(
                image_id=image_id,
                fiber_id=int(row[id_column]),
                model_fiber_type=model,
                reviewed_fiber_type=corrected,
                review_status=status,
                queue_source="legacy_review_csv",
            )
        )
    return tuple(decisions)
