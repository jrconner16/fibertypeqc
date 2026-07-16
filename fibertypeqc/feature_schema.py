"""Named feature contracts used to guard classifier compatibility."""

from __future__ import annotations

LEGACY_FROZEN_ALPHA_FEATURE_SCHEMA = "legacy_type1_type2.v1"
MULTIPANEL_FEATURE_SCHEMA = "multiplanel_features.v1"

SEMANTIC_MARKER_NAMES = {
    "i": "type_i",
    "iia": "type_iia",
    "iib": "type_iib",
    "iix": "type_iix",
    "emhc": "emhc",
}
SEMANTIC_DIAGNOSTIC_SUFFIXES = (
    "mean",
    "p75",
    "p90",
    "pctl",
    "coverage_high",
    "snr_mean",
    "snr_p90",
)

# The legacy classifier's input contract is intentionally frozen.  New semantic
# features will be introduced under ``MULTIPANEL_FEATURE_SCHEMA`` in Phase 3.
LEGACY_FROZEN_ALPHA_FEATURES = (
    "type1_mean",
    "type2_mean",
)
