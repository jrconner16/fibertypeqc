"""Edge-of-image fibers: queue handling, opt-in exclusion, and the call overlay helpers."""

import pandas as pd

from src.review.call_overlay import CALL_COLORS, call_color_dict, current_calls, legend_html
from src.review.queues import QueueSource, build_fiber_type_queue, describe_reason
from src.review.schemas import FiberTypeDecision, ObjectReviewStatus


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": ["a"] * 3,
            "mouse_id": ["m"] * 3,
            "fiber_id": [1, 2, 3],
            "model_fiber_type": ["iia", "iib", "iix"],
            "confidence": [0.5, 0.5, 0.9],
            "probability_margin": [0.1, 0.1, 0.8],
            "normalized_entropy": [0.9, 0.9, 0.1],
            "needs_review": [True, True, False],
            "typing_signal_qc_flags": ["", "", ""],
            "technical_flags": ["", "", ""],
            "touches_image_border": [True, False, False],
        }
    )


def test_flagged_queue_skips_fibers_cut_by_the_image_edge():
    queue = build_fiber_type_queue(_rows(), QueueSource.FLAGGED)

    assert [item.fiber_id for item in queue] == [2]


def test_section_review_keeps_edge_fibers_and_says_why():
    queue = build_fiber_type_queue(_rows(), QueueSource.FULL_AUDIT)

    assert [item.fiber_id for item in queue] == [1, 2, 3]
    assert "cut off by the image edge" in describe_reason(queue[0].reason_code)
    assert "edge" not in describe_reason(queue[1].reason_code)


def _decision(fiber_id: int, reviewed: str, status: ObjectReviewStatus) -> FiberTypeDecision:
    return FiberTypeDecision(
        image_id="a",
        fiber_id=fiber_id,
        model_fiber_type="iib",
        reviewed_fiber_type=reviewed,
        review_status=status,
    )


def test_current_calls_apply_review_decisions_for_one_image():
    model = {1: "iib", 2: "iib", 3: "iib", 4: "iib"}
    decisions = [
        _decision(1, "iix", ObjectReviewStatus.CORRECTED),
        _decision(2, "exclude", ObjectReviewStatus.EXCLUDED),
        _decision(3, "iib", ObjectReviewStatus.ACCEPTED),
        _decision(4, "", ObjectReviewStatus.UNRESOLVED),
    ]

    assert current_calls(model, decisions, "a") == {
        1: "iix",
        2: "excluded",
        3: "iib",
        4: "unresolved",
    }
    assert current_calls(model, decisions, "other") == model


def test_call_colors_and_legend():
    colors = call_color_dict({1: "iia", 2: "excluded", 3: "something_else"})

    assert colors[1] == CALL_COLORS["iia"]
    assert colors[2] == CALL_COLORS["excluded"]
    assert colors[0][3] == 0.0 and colors[None][3] == 0.0
    legend = legend_html(["iia", "iib", "iix"])
    assert "IIa" in legend and "excluded" in legend and "> I<" not in legend
