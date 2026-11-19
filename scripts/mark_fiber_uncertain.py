"""Mark one reviewed fiber as uncertain, outside the review window.

Use this when finalization refuses a review decision whose class the project cannot produce
(for example Type I on a panel without a Type I stain) and the fiber cannot be reviewed again in
napari. The fiber's final type becomes unresolved; no class is chosen for it. The previous
decision stays in the append-only event log (`review_events.csv`), and the new event records the
reason given.

Used to correct the public demo's example review (v0.5.2): fiber 39 of `demo_section_b` had been
recorded as Type I on the three-class TA panel.

    uv run python -m scripts.mark_fiber_uncertain PROJECT IMAGE_ID FIBER_ID --reason REASON
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.review.project import load_project
from src.review.schemas import Domain, FiberTypeDecision, ObjectReviewStatus, ReviewEvent, Scope
from src.review.storage import append_review_event, load_session, save_session


def mark_uncertain(project_path: Path, image_id: str, fiber_id: int, reason: str) -> ReviewEvent:
    project = load_project(project_path)
    session = load_session(project.review_state_path, expected_project_id=project.project_id)
    previous = next(
        (
            decision
            for decision in session.object_decisions
            if decision.image_id == image_id and decision.fiber_id == fiber_id
        ),
        None,
    )
    if previous is None:
        raise SystemExit(f"No review decision for fiber {fiber_id} in {image_id}.")
    decision = FiberTypeDecision(
        image_id=image_id,
        fiber_id=fiber_id,
        model_fiber_type=previous.model_fiber_type,
        reviewed_fiber_type="uncertain",
        review_status=ObjectReviewStatus.UNCERTAIN,
        queue_source=previous.queue_source,
        reason_code=reason,
        reviewer=previous.reviewer,
    )
    session.record_fiber_type_decision(decision)
    event = ReviewEvent(
        image_id=image_id,
        scope=Scope.OBJECT,
        domain=Domain.FIBER_TYPING,
        target_id=str(fiber_id),
        action="record_fiber_type_decision",
        reason_code=reason,
        old_value=previous.to_dict(),
        new_value=decision.to_dict(),
        reviewer=previous.reviewer,
        model_version=session.model_version,
        qc_version=session.qc_version,
    )
    append_review_event(project.review_events_path, event)
    save_session(project.review_state_path, session)
    return event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("project", type=Path, help="project folder or review project.yaml")
    parser.add_argument("image_id")
    parser.add_argument("fiber_id", type=int)
    parser.add_argument("--reason", required=True, help="reason code recorded in the event log")
    args = parser.parse_args(argv)
    event = mark_uncertain(args.project, args.image_id, args.fiber_id, args.reason)
    print(f"{args.image_id} fiber {args.fiber_id}: marked uncertain ({event.reason_code}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
