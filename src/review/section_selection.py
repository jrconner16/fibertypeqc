"""Domain-specific mouse/section selection from precomputed image QC."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pandas as pd

from src.review.project import Project
from src.review.schemas import Domain

SECTION_SELECTION_SCHEMA_VERSION = "review_section_selection.v1"
SECTION_SELECTION_COLUMNS = [
    "schema_version",
    "qc_version",
    "rules_version",
    "model_version",
    "computed_at",
    "project_id",
    "mouse_id",
    "domain",
    "strategy",
    "selected_image_ids",
    "eligible_image_ids",
    "requires_manual_review",
    "reason_code",
]


class SelectionStrategy(StrEnum):
    ALL_PASSING = "all_passing"
    BEST_PASSING = "best_passing"
    MANUAL = "manual"


@dataclass(frozen=True)
class SelectionContext:
    qc_version: str
    rules_version: str
    computed_at: str


def _validate_image_qc(image_qc: pd.DataFrame) -> SelectionContext:
    required = {
        "image_id",
        "mouse_id",
        "domain",
        "applicable",
        "hard_fail",
        "technical_quality_score",
        "review_priority",
        "qc_version",
        "rules_version",
        "computed_at",
    }
    missing = sorted(required - set(image_qc.columns))
    if missing:
        raise ValueError(f"image_qc is missing required columns: {missing}")
    if image_qc.empty:
        raise ValueError("image_qc must not be empty")
    return SelectionContext(
        qc_version=str(image_qc["qc_version"].iloc[0]),
        rules_version=str(image_qc["rules_version"].iloc[0]),
        computed_at=str(image_qc["computed_at"].iloc[0]),
    )


def _manual_ids(
    selections: Mapping[str, Mapping[str, list[str]]] | None,
    mouse_id: str,
    domain: Domain,
) -> list[str]:
    if selections is None:
        return []
    mouse = selections.get(mouse_id, {})
    if not isinstance(mouse, Mapping):
        raise ValueError(f"Manual selections for mouse {mouse_id!r} must be a mapping")
    values = mouse.get(domain.value, [])
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(
            f"Manual selection {mouse_id}.{domain.value} must be a list of image IDs"
        )
    return values


def select_sections(
    project: Project,
    image_qc: pd.DataFrame,
    *,
    strategy: SelectionStrategy | str = SelectionStrategy.ALL_PASSING,
    manual_selections: Mapping[str, Mapping[str, list[str]]] | None = None,
) -> pd.DataFrame:
    """Select eligible sections independently for every mouse/domain."""
    try:
        parsed_strategy = SelectionStrategy(strategy)
    except ValueError as exc:
        choices = ", ".join(item.value for item in SelectionStrategy)
        raise ValueError(f"strategy must be one of: {choices}; got {strategy!r}") from exc
    context = _validate_image_qc(image_qc)
    manifest_order = {
        image.image_id: index for index, image in enumerate(project.images)
    }
    mouse_order = list(dict.fromkeys(image.mouse_id for image in project.images))
    rows: list[dict[str, Any]] = []

    for mouse_id in mouse_order:
        mouse_image_ids = [
            image.image_id for image in project.images if image.mouse_id == mouse_id
        ]
        for domain in Domain:
            candidates = image_qc[
                image_qc["image_id"].isin(mouse_image_ids)
                & image_qc["domain"].eq(domain.value)
            ].copy()
            applicable = candidates[candidates["applicable"].astype(bool)].copy()
            eligible = applicable[~applicable["hard_fail"].astype(bool)].copy()
            eligible = eligible.sort_values(
                "image_id",
                key=lambda values: values.map(manifest_order),
                kind="stable",
            )
            eligible_ids = eligible["image_id"].astype(str).tolist()
            selected: list[str] = []
            requires_manual = False
            reason = ""

            if applicable.empty:
                reason = "domain_not_applicable"
            elif eligible.empty:
                requires_manual = True
                reason = "no_passing_sections"
            elif parsed_strategy is SelectionStrategy.ALL_PASSING:
                selected = eligible_ids
                reason = "all_passing_sections"
            elif parsed_strategy is SelectionStrategy.BEST_PASSING:
                ranked = eligible.assign(
                    _manifest_order=eligible["image_id"].map(manifest_order)
                ).sort_values(
                    [
                        "technical_quality_score",
                        "review_priority",
                        "_manifest_order",
                    ],
                    ascending=[False, True, True],
                    kind="stable",
                )
                selected = [str(ranked.iloc[0]["image_id"])]
                reason = "best_passing_section"
            else:
                selected = _manual_ids(manual_selections, mouse_id, domain)
                invalid = [image_id for image_id in selected if image_id not in eligible_ids]
                if invalid:
                    raise ValueError(
                        f"Manual selection for {mouse_id}/{domain.value} contains "
                        f"ineligible image IDs: {invalid}"
                    )
                if selected:
                    reason = "manual_selection"
                else:
                    requires_manual = True
                    reason = "manual_selection_required"

            rows.append(
                {
                    "schema_version": SECTION_SELECTION_SCHEMA_VERSION,
                    "qc_version": context.qc_version,
                    "rules_version": context.rules_version,
                    "model_version": project.model_version,
                    "computed_at": context.computed_at,
                    "project_id": project.project_id,
                    "mouse_id": mouse_id,
                    "domain": domain.value,
                    "strategy": parsed_strategy.value,
                    "selected_image_ids": "|".join(selected),
                    "eligible_image_ids": "|".join(eligible_ids),
                    "requires_manual_review": requires_manual,
                    "reason_code": reason,
                }
            )
    return pd.DataFrame(rows, columns=SECTION_SELECTION_COLUMNS)
