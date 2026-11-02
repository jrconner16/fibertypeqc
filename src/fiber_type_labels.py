from __future__ import annotations

import pandas as pd

INTERNAL_TO_BIOLOGICAL = {
    "type1": "iib",
    "type2": "iia",
    "unknown": "iix_candidate",
    "mixed": "hybrid_candidate",
    "hybrid": "hybrid",
    "uncertain": "uncertain",
    "exclude": "exclude",
    "i": "i",
    "iib": "iib",
    "iia": "iia",
    "iix": "iix",
    "iix_candidate": "iix_candidate",
    "hybrid_candidate": "hybrid_candidate",
    "unclassified": "unclassified",
    "negative": "negative",
    "": "",
}

BIOLOGICAL_TO_INTERNAL_MARKER = {
    "iib": "type1",
    "iia": "type2",
}

REVIEW_TYPES = ["i", "iib", "iia", "iix", "hybrid", "uncertain", "exclude"]

# Classes a project without a model may be labelled with (``classes`` in the project config).
# ``negative`` means negative for every measured myosin marker; it is never read as IIx or IIb.
# ``unclassified`` (no call yet) is not a class: it is what every fiber has before a model exists,
# and it is distinct from the legacy rule call ``unknown`` (read as IIx) and from a reviewer's
# ``uncertain``.
MYOSIN_CLASSES = ("i", "iia", "iib", "iix")
CONFIGURABLE_CLASSES = (*MYOSIN_CLASSES, "hybrid", "negative")
UNCLASSIFIED = "unclassified"
CLASS_DISPLAY = {"i": "I", "iia": "IIa", "iib": "IIb", "iix": "IIx", "hybrid": "Hybrid"}


def negative_display_label(markers: tuple[str, ...] | list[str]) -> str:
    """How ``negative`` is shown for a panel, e.g. ``I⁻/IIa⁻`` for Type I and IIa markers."""
    shown = [f"{CLASS_DISPLAY[marker]}⁻" for marker in MYOSIN_CLASSES if marker in markers]
    return "/".join(shown) if shown else "Negative"


def default_classes(markers: tuple[str, ...] | list[str]) -> list[str]:
    """One class per measured myosin marker, plus ``negative``; ``hybrid`` only on request."""
    return [marker for marker in MYOSIN_CLASSES if marker in markers] + ["negative"]


def validate_classes(classes: object, markers: tuple[str, ...] | list[str]) -> list[str]:
    """Check a configured class list against the vocabulary and the panel's markers."""
    if not isinstance(classes, list) or not all(isinstance(value, str) for value in classes):
        raise ValueError("'classes' must be a list of class names")
    names = [value.strip().lower() for value in classes]
    unknown = [name for name in names if name not in CONFIGURABLE_CLASSES]
    if unknown:
        raise ValueError(
            f"'classes' has unsupported names {unknown}; use {', '.join(CONFIGURABLE_CLASSES)}"
        )
    if len(set(names)) != len(names):
        raise ValueError("'classes' lists a class more than once")
    if len(names) < 2:
        raise ValueError("'classes' needs at least two classes")
    unmeasured = [name for name in names if name in MYOSIN_CLASSES and name not in markers]
    if unmeasured:
        raise ValueError(
            f"'classes' includes {unmeasured}, but the panel has no channel for that marker"
        )
    return names


def to_biological_label(value: object) -> str:
    if pd.isna(value):
        return ""
    key = str(value).strip().lower()
    return INTERNAL_TO_BIOLOGICAL.get(key, key)


def normalize_review_label(value: object) -> str:
    label = to_biological_label(value)
    if label == "iix_candidate":
        return "iix"
    if label == "hybrid_candidate":
        return "hybrid"
    return label
