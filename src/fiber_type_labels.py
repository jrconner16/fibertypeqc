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
    "iib": "iib",
    "iia": "iia",
    "iix": "iix",
    "iix_candidate": "iix_candidate",
    "hybrid_candidate": "hybrid_candidate",
    "": "",
}

BIOLOGICAL_TO_INTERNAL_MARKER = {
    "iib": "type1",
    "iia": "type2",
}

REVIEW_TYPES = ["iib", "iia", "iix", "hybrid", "uncertain", "exclude"]


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
