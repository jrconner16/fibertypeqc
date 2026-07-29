"""Central dependency rules for reviewed edits."""

from __future__ import annotations

from enum import StrEnum


class EditKind(StrEnum):
    FIBER_MASK = "fiber_mask"
    NUCLEUS_MASK = "nucleus_mask"
    NUCLEUS_REASSIGNMENT = "nucleus_reassignment"
    FIBER_TYPE_CORRECTION = "fiber_type_correction"
    REGION_EXCLUSION = "region_exclusion"


class StaleProduct(StrEnum):
    FIBER_GEOMETRY_FEATURES = "fiber_geometry_features"
    FIBER_TYPE_PREDICTION_FEATURES = "fiber_type_prediction_features"
    NUCLEUS_FEATURES = "nucleus_features"
    NUCLEUS_ASSOCIATIONS = "nucleus_associations"
    FIBER_NUCLEUS_COUNTS = "fiber_nucleus_counts"


INVALIDATION_DEPENDENCIES: dict[EditKind, frozenset[StaleProduct]] = {
    EditKind.FIBER_MASK: frozenset(
        {
            StaleProduct.FIBER_GEOMETRY_FEATURES,
            StaleProduct.FIBER_TYPE_PREDICTION_FEATURES,
            StaleProduct.NUCLEUS_ASSOCIATIONS,
            StaleProduct.FIBER_NUCLEUS_COUNTS,
        }
    ),
    EditKind.NUCLEUS_MASK: frozenset(
        {
            StaleProduct.NUCLEUS_FEATURES,
            StaleProduct.NUCLEUS_ASSOCIATIONS,
            StaleProduct.FIBER_NUCLEUS_COUNTS,
        }
    ),
    EditKind.NUCLEUS_REASSIGNMENT: frozenset(
        {
            StaleProduct.NUCLEUS_ASSOCIATIONS,
            StaleProduct.FIBER_NUCLEUS_COUNTS,
        }
    ),
    EditKind.FIBER_TYPE_CORRECTION: frozenset(),
    EditKind.REGION_EXCLUSION: frozenset(),
}


def invalidated_products(edit_kind: EditKind | str) -> frozenset[StaleProduct]:
    try:
        parsed = EditKind(edit_kind)
    except ValueError as exc:
        choices = ", ".join(item.value for item in EditKind)
        raise ValueError(f"edit_kind must be one of: {choices}; got {edit_kind!r}") from exc
    return INVALIDATION_DEPENDENCIES[parsed]
