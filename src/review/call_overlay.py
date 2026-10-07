"""Colors for the reviewer's fiber-call overlay (no GUI imports; testable headlessly)."""

from __future__ import annotations

import pandas as pd

from src.fiber_type_labels import normalize_review_label
from src.review.schemas import FiberTypeDecision, ObjectReviewStatus

# RGBA per current call. Class colors match the stain display colors, so an outline that
# disagrees with the stain inside it stands out.
CALL_COLORS: dict[str, tuple[float, float, float, float]] = {
    "i": (0.2, 0.4, 1.0, 1.0),
    "iia": (0.1, 0.9, 0.2, 1.0),
    "iib": (1.0, 0.0, 0.8, 1.0),
    "iix": (1.0, 0.55, 0.0, 1.0),
    "excluded": (1.0, 0.1, 0.1, 1.0),
    "unresolved": (0.6, 0.6, 0.6, 1.0),
}
OTHER_COLOR = (1.0, 1.0, 1.0, 1.0)
CALL_NAMES = {
    "i": "I",
    "iia": "IIa",
    "iib": "IIb",
    "iix": "IIx",
    "excluded": "excluded",
    "unresolved": "uncertain / unresolved",
}


def model_calls_from_table(table: pd.DataFrame) -> dict[int, str]:
    """Fiber ID -> model call from a per-image fiber table."""
    id_column = "fiber_id" if "fiber_id" in table.columns else "label"
    return {
        int(fiber_id): normalize_review_label(call)
        for fiber_id, call in zip(table[id_column], table["fiber_type"], strict=True)
        if int(fiber_id) > 0
    }


def current_calls(
    model_calls: dict[int, str], decisions: list[FiberTypeDecision], image_id: str
) -> dict[int, str]:
    """Model calls with this image's review decisions applied."""
    calls = dict(model_calls)
    for decision in decisions:
        if decision.image_id != image_id or decision.fiber_id not in calls:
            continue
        reviewed = str(decision.reviewed_fiber_type or "").strip().lower()
        status = decision.review_status
        if status is ObjectReviewStatus.EXCLUDED or reviewed == "exclude":
            calls[decision.fiber_id] = "excluded"
        elif status in (ObjectReviewStatus.UNCERTAIN, ObjectReviewStatus.UNRESOLVED):
            calls[decision.fiber_id] = "unresolved"
        elif status is not ObjectReviewStatus.ACCEPTED and reviewed:
            calls[decision.fiber_id] = normalize_review_label(reviewed)
    return calls


def call_color_dict(calls: dict[int, str]) -> dict[int | None, tuple[float, float, float, float]]:
    """Label -> RGBA mapping for a direct label colormap; background is transparent."""
    colors: dict[int | None, tuple[float, float, float, float]] = {
        fiber_id: CALL_COLORS.get(str(call).lower(), OTHER_COLOR)
        for fiber_id, call in calls.items()
    }
    colors[0] = (0.0, 0.0, 0.0, 0.0)
    colors[None] = (0.0, 0.0, 0.0, 0.0)
    return colors


def legend_html(classes: list[str]) -> str:
    """One-line colored legend for the classes in use plus the review outcomes."""
    names = [name for name in CALL_COLORS if name in classes or name in ("excluded", "unresolved")]
    parts = []
    for name in names:
        red, green, blue, _ = CALL_COLORS[name]
        color = f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"
        parts.append(f'<span style="color:{color}">&#9632;</span> {CALL_NAMES[name]}')
    return "Outline colors: " + " &nbsp; ".join(parts)
