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
# On a regeneration panel (eMHC, no myosin marker) the classes are ``emhc`` and ``negative``,
# which then means eMHC-negative. eMHC is not combined with myosin classes: it is independent of
# fiber type, and one call per fiber cannot hold both.
# ``unclassified`` (no call yet) is not a class: it is what every fiber has before a model exists,
# and it is distinct from the legacy rule call ``unknown`` (read as IIx) and from a reviewer's
# ``uncertain``.
MYOSIN_CLASSES = ("i", "iia", "iib", "iix")
EMHC_CLASS = "emhc"
CONFIGURABLE_CLASSES = (*MYOSIN_CLASSES, EMHC_CLASS, "hybrid", "negative")
UNCLASSIFIED = "unclassified"
CLASS_DISPLAY = {
    "i": "I", "iia": "IIa", "iib": "IIb", "iix": "IIx", "emhc": "eMHC⁺", "hybrid": "Hybrid",
}  # fmt: skip


def negative_display_label(markers: tuple[str, ...] | list[str]) -> str:
    """How ``negative`` is shown for a panel, e.g. ``I⁻/IIa⁻`` for Type I and IIa markers, or
    ``eMHC⁻`` on a regeneration panel (eMHC without a myosin marker)."""
    shown = [f"{CLASS_DISPLAY[marker]}⁻" for marker in MYOSIN_CLASSES if marker in markers]
    if not shown and EMHC_CLASS in markers:
        return "eMHC⁻"
    return "/".join(shown) if shown else "Negative"


def default_classes(markers: tuple[str, ...] | list[str]) -> list[str]:
    """One class per measured myosin marker, plus ``negative``; ``hybrid`` only on request.

    A panel with eMHC and no myosin marker gets ``emhc`` and ``negative`` (eMHC⁻)."""
    myosin = [marker for marker in MYOSIN_CLASSES if marker in markers]
    if not myosin and EMHC_CLASS in markers:
        return [EMHC_CLASS, "negative"]
    return myosin + ["negative"]


def class_labels_for(classes: list[str] | tuple[str, ...], negative_label: str) -> dict[str, str]:
    """Display names a project stores for its classes: ``negative`` always, ``emhc`` when used."""
    labels = {"negative": negative_label} if negative_label else {}
    if EMHC_CLASS in classes:
        labels[EMHC_CLASS] = CLASS_DISPLAY[EMHC_CLASS]
    return labels


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
    if EMHC_CLASS in names:
        if EMHC_CLASS not in markers:
            raise ValueError("'classes' includes emhc, but the panel has no eMHC channel")
        others = [name for name in names if name not in (EMHC_CLASS, "negative")]
        if others:
            raise ValueError(
                f"'classes' combines emhc with {others}; eMHC is labelled on its own "
                "(classes: [emhc, negative])"
            )
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
