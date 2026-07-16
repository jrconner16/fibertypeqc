"""Named feature contracts used to guard classifier compatibility."""

from __future__ import annotations

LEGACY_FROZEN_ALPHA_FEATURE_SCHEMA = "legacy_type1_type2.v1"
MULTIPANEL_FEATURE_SCHEMA = "multiplanel_features.v1"

# The legacy classifier's input contract is intentionally frozen.  New semantic
# features will be introduced under ``MULTIPANEL_FEATURE_SCHEMA`` in Phase 3.
LEGACY_FROZEN_ALPHA_FEATURES = (
    "type1_mean",
    "type2_mean",
)
