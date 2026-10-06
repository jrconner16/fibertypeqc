"""Turn a reviewed project into analysis-ready, provenance-carrying fiber tables.

Finalization only reads predictions, label masks, and review state and writes new files; it never
modifies ``*_fibers.csv``. Every model column is preserved next to the final value, and each fiber
carries exactly one ``value_source`` (predicted, reviewed, excluded, or unresolved) and reason.

Policies (approved for the release; recorded in the finalization manifest):

- Precedence: exclusion (image/section, then region, then fiber) over the reviewer's decision, over
  the prediction.
- Region and ROI membership is decided by the fiber centroid.
- Flagged fibers that nobody reviewed keep the model call and are counted as unreviewed.
- Unresolved fibers have no final type; summaries leave them out of composition denominators and
  report their count.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops

from fibertypeqc.artifacts import file_sha256, git_commit
from src.fiber_type_labels import normalize_review_label
from src.review.fiber_type_review import load_legacy_fiber_type_decisions
from src.review.invalidation import StaleProduct
from src.review.project import Project, ProjectImage
from src.review.region_review import _point_in_polygon
from src.review.schemas import (
    Domain,
    DomainStatus,
    FiberTypeDecision,
    ObjectReviewStatus,
    RegionAction,
    RegionKind,
)
from src.review.session import ReviewSession
from src.review.storage import atomic_write_dataframe

FINALIZATION_SCHEMA_VERSION = "fibertypeqc.finalization.v1"
FIBER_INPUT_KEYS = ("fiber_labels", "fiber_table")
VALUE_SOURCES = ("predicted", "reviewed", "excluded", "unresolved")
POLICIES = {
    "precedence": "image_or_section_exclusion > region_exclusion > fiber_decision > prediction",
    "region_membership": "centroid_inside",
    "boundary_centroid": "inside_for_exclusion; unassigned_for_roi",
    "flagged_unreviewed": "keep_prediction_and_report",
    "unresolved_in_composition": "excluded_from_denominator_and_reported",
}
EXCLUDING_DOMAIN_STATUSES = frozenset({DomainStatus.EXCLUDED, DomainStatus.FAIL})
FIBER_DOMAINS = frozenset({Domain.FIBER_SEGMENTATION, Domain.FIBER_TYPING})
FIBER_EXCLUDING_ACTIONS = frozenset(
    {RegionAction.EXCLUDE_DOMAIN, RegionAction.IGNORE_FIBER_TYPING}
)
# Fiber-mask edits make these stale; finalizing would apply decisions to outdated features.
FIBER_STALE_PRODUCTS = frozenset(
    {
        StaleProduct.FIBER_GEOMETRY_FEATURES.value,
        StaleProduct.FIBER_TYPE_PREDICTION_FEATURES.value,
    }
)
FINAL_COLUMNS = (
    "final_type",
    "value_source",
    "exclusion_reason",
    "review_status",
    "reviewer",
    "decided_at",
    "flagged_unreviewed",
    "roi_name",
    "roi_role",
    "roi_status",
)


class FinalizationError(ValueError):
    """Review state cannot be applied safely to the current predictions."""


@dataclass
class ImageFinalization:
    image_id: str
    table: pd.DataFrame
    input_digests: dict[str, str]
    verification: str
    counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def image_input_fingerprints(image: ProjectImage) -> dict[str, str]:
    """Digests of the inputs fiber review decisions refer to."""
    return {
        key: file_sha256(image.outputs[key]) for key in FIBER_INPUT_KEYS if key in image.outputs
    }


def _fiber_id_column(table: pd.DataFrame) -> str:
    for name in ("fiber_id", "label"):
        if name in table.columns:
            return name
    raise FinalizationError("Fiber table has neither a fiber_id nor a label column.")


def _centroids(labels: np.ndarray) -> dict[int, tuple[float, float]]:
    """Fiber centroids as (x, y) in full-resolution image coordinates."""
    return {
        int(region.label): (float(region.centroid[1]), float(region.centroid[0]))
        for region in regionprops(labels)
    }


def image_border_fiber_ids(labels: np.ndarray) -> set[int]:
    """IDs of fibers with at least one pixel on the image edge."""
    edge = np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
    return {int(value) for value in np.unique(edge) if value > 0}


def _has_review_activity(session: ReviewSession, image_id: str) -> bool:
    return (
        any(decision.image_id == image_id for decision in session.object_decisions)
        or any(region.image_id == image_id for region in session.regions)
        or bool(session.image_statuses.get(image_id))
    )


def _verify_inputs(
    image: ProjectImage, session: ReviewSession, require_verified: bool
) -> tuple[dict[str, str], str]:
    digests = image_input_fingerprints(image)
    recorded = session.input_fingerprints.get(image.image_id)
    if recorded is None and not _has_review_activity(session, image.image_id):
        # Never opened and nothing decided: there are no decisions to verify.
        return digests, "not_reviewed"
    if recorded is None:
        if require_verified:
            raise FinalizationError(
                f"{image.image_id}: review session has no input fingerprints; reopen the image in "
                "the reviewer to record them, or finalize without --require-verified."
            )
        return digests, "unverified"
    changed = sorted(key for key, value in recorded.items() if digests.get(key) != value)
    if changed:
        raise FinalizationError(
            f"{image.image_id}: {', '.join(changed)} changed since review; decisions may refer to "
            "different fibers. Re-review the image or restore the reviewed inputs."
        )
    return digests, "verified"


def _region_effects(
    session: ReviewSession, image_id: str, centroids: dict[int, tuple[float, float]]
) -> tuple[dict[int, str], set[int], dict[int, tuple[str, str, str]]]:
    """Return region exclusions, region-unresolved fibers, and ROI assignments."""
    excluded: dict[int, str] = {}
    unresolved: set[int] = set()
    regions = [region for region in session.regions if region.image_id == image_id]
    rois = [region for region in regions if region.kind is RegionKind.ANALYSIS_ROI]
    for region in regions:
        if region.kind is RegionKind.ANALYSIS_ROI:
            continue
        action = RegionAction(region.action)
        applies_to_fibers = action is RegionAction.EXCLUDE_ALL_ANALYSIS or (
            region.domain in FIBER_DOMAINS
            and action in (FIBER_EXCLUDING_ACTIONS | {RegionAction.UNRESOLVED})
        )
        if not applies_to_fibers:
            continue
        for fiber_id, (x, y) in centroids.items():
            if _point_in_polygon(x, y, region.geometry) == "outside":
                continue
            if action is RegionAction.UNRESOLVED:
                unresolved.add(fiber_id)
            else:
                excluded.setdefault(fiber_id, f"region:{action.value}:{region.region_id}")
    assignments: dict[int, tuple[str, str, str]] = {}
    if rois:
        for fiber_id, (x, y) in centroids.items():
            matches = [_point_in_polygon(x, y, roi.geometry) for roi in rois]
            contained = [roi for roi, match in zip(rois, matches, strict=True) if match == "inside"]
            if "boundary" in matches:
                assignments[fiber_id] = ("", "", "boundary")
            elif not contained:
                assignments[fiber_id] = ("", "", "outside")
                excluded.setdefault(fiber_id, "outside_analysis_roi")
            elif len(contained) > 1:
                assignments[fiber_id] = ("", "", "ambiguous")
            else:
                assignments[fiber_id] = (contained[0].name, contained[0].role, "assigned")
    return excluded, unresolved, assignments


def _image_exclusion(
    session: ReviewSession, image: ProjectImage, selected: set[str] | None
) -> str:
    for domain in (Domain.FIBER_SEGMENTATION, Domain.FIBER_TYPING):
        status = session.get_status(image.image_id, domain)
        if status in EXCLUDING_DOMAIN_STATUSES:
            return f"image:{domain.value}:{status.value}"
    if selected is not None and image.image_id not in selected:
        return "section_not_selected"
    return ""


def finalize_image(
    project: Project,
    session: ReviewSession,
    image: ProjectImage,
    *,
    selected_image_ids: set[str] | None = None,
    require_verified: bool = False,
    exclude_image_border_fibers: bool = False,
) -> ImageFinalization:
    if "fiber_table" not in image.outputs or "fiber_labels" not in image.outputs:
        raise FinalizationError(f"{image.image_id}: fiber_table and fiber_labels are required.")
    stale = FIBER_STALE_PRODUCTS & set(session.stale_products.get(image.image_id, []))
    if stale:
        raise FinalizationError(
            f"{image.image_id}: {', '.join(sorted(stale))} are stale after review edits; "
            "re-quantify before finalizing."
        )
    digests, verification = _verify_inputs(image, session, require_verified)
    predictions = pd.read_csv(image.outputs["fiber_table"])
    id_column = _fiber_id_column(predictions)
    labels = np.asarray(tifffile.imread(image.outputs["fiber_labels"]), dtype=np.int64)
    centroids = _centroids(labels)
    # Opt-in: fibers cut by the image edge (cropped fields) cannot be typed reliably.
    border_ids = image_border_fiber_ids(labels) if exclude_image_border_fibers else set()
    missing = sorted(set(predictions[id_column].astype(int)) - set(centroids))
    if missing:
        raise FinalizationError(
            f"{image.image_id}: {len(missing)} fibers in the table are absent from the label mask."
        )

    decisions: dict[int, FiberTypeDecision] = {
        decision.fiber_id: decision
        for decision in session.object_decisions
        if decision.image_id == image.image_id
    }
    model_calls = dict(
        zip(predictions[id_column].astype(int), predictions["fiber_type"].astype(str), strict=True)
    )
    mismatched = sorted(
        fiber_id
        for fiber_id, decision in decisions.items()
        if fiber_id not in model_calls
        or normalize_review_label(model_calls[fiber_id])
        != normalize_review_label(decision.model_fiber_type)
    )
    if mismatched:
        raise FinalizationError(
            f"{image.image_id}: {len(mismatched)} review decisions refer to model calls that no "
            "longer match the fiber table; predictions changed after review."
        )

    image_reason = _image_exclusion(session, image, selected_image_ids)
    region_excluded, region_unresolved, roi = _region_effects(session, image.image_id, centroids)
    flagged = (
        predictions["needs_review"].fillna(False).astype(bool)
        if "needs_review" in predictions.columns
        else pd.Series(False, index=predictions.index)
    )

    rows: list[dict[str, Any]] = []
    for fiber_id, model_call, is_flagged in zip(
        predictions[id_column].astype(int), predictions["fiber_type"].astype(str), flagged,
        strict=True,
    ):
        decision = decisions.get(fiber_id)
        row: dict[str, Any] = {
            "final_type": model_call,
            "value_source": "predicted",
            "exclusion_reason": "",
            "review_status": decision.review_status.value if decision else "",
            "reviewer": decision.reviewer if decision else "",
            "decided_at": decision.timestamp if decision else "",
            "flagged_unreviewed": bool(is_flagged and decision is None),
        }
        roi_name, roi_role, roi_status = roi.get(fiber_id, ("", "", ""))
        row.update({"roi_name": roi_name, "roi_role": roi_role, "roi_status": roi_status})
        if image_reason:
            row.update(final_type="", value_source="excluded", exclusion_reason=image_reason)
        elif fiber_id in region_excluded:
            row.update(
                final_type="", value_source="excluded", exclusion_reason=region_excluded[fiber_id]
            )
        elif fiber_id in border_ids:
            row.update(final_type="", value_source="excluded", exclusion_reason="edge_of_image")
        elif decision is not None:
            reviewed = normalize_review_label(decision.reviewed_fiber_type)
            status = decision.review_status
            if status is ObjectReviewStatus.EXCLUDED or reviewed == "exclude":
                row.update(final_type="", value_source="excluded", exclusion_reason="fiber_review")
            elif status in (ObjectReviewStatus.UNCERTAIN, ObjectReviewStatus.UNRESOLVED) or (
                reviewed == "uncertain"
            ):
                row.update(final_type="", value_source="unresolved")
            elif status is ObjectReviewStatus.ACCEPTED:
                row.update(value_source="reviewed")
            else:
                row.update(final_type=reviewed or model_call, value_source="reviewed")
        elif fiber_id in region_unresolved:
            row.update(final_type="", value_source="unresolved")
        rows.append(row)

    final = pd.DataFrame(rows, columns=list(FINAL_COLUMNS), index=predictions.index)
    clashing = sorted(set(FINAL_COLUMNS) & set(predictions.columns))
    if clashing:
        raise FinalizationError(f"{image.image_id}: fiber table already has {clashing}.")
    table = pd.concat([predictions, final], axis=1)
    table.insert(0, "section_id", image.section_id)
    table.insert(0, "mouse_id", image.mouse_id)
    table.insert(0, "image_id", image.image_id)
    counts = {source: int((table["value_source"] == source).sum()) for source in VALUE_SOURCES}
    counts["flagged_unreviewed"] = int(table["flagged_unreviewed"].sum())
    warnings = []
    if verification == "unverified":
        warnings.append("review session has no input fingerprints for this image")
    roi_statuses = Counter(status for _, _, status in roi.values())
    for status in ("boundary", "ambiguous"):
        if roi_statuses.get(status):
            warnings.append(
                f"{roi_statuses[status]} fiber centroids are {status} for analysis ROIs"
            )
    return ImageFinalization(image.image_id, table, digests, verification, counts, warnings)


def import_legacy_reviews(
    project: Project, session: ReviewSession, reviews: dict[str, Path]
) -> int:
    """Add decisions from legacy per-image review CSVs to the session (in memory)."""
    imported = 0
    for image_id, path in reviews.items():
        image = project.image(image_id)
        predictions = pd.read_csv(image.outputs["fiber_table"])
        id_column = _fiber_id_column(predictions)
        model_calls = dict(
            zip(
                predictions[id_column].astype(int),
                predictions["fiber_type"].astype(str),
                strict=True,
            )
        )
        existing = {d.fiber_id for d in session.object_decisions if d.image_id == image_id}
        for decision in load_legacy_fiber_type_decisions(
            path, image_id=image_id, model_calls=model_calls
        ):
            if decision.fiber_id in existing:
                raise FinalizationError(
                    f"{image_id}: fiber {decision.fiber_id} has decisions in both the review "
                    "session and the legacy CSV; resolve the conflict first."
                )
            session.record_fiber_type_decision(decision)
            imported += 1
    return imported


def load_selected_image_ids(path: Path) -> set[str] | None:
    """Fiber-typing section selection from a project QC directory, if present."""
    if not path.is_file():
        return None
    table = pd.read_csv(path, keep_default_na=False)
    rows = table[table["domain"].eq(Domain.FIBER_TYPING.value)]
    if rows.empty:
        return None
    selected: set[str] = set()
    for value in rows["selected_image_ids"]:
        selected.update(item for item in str(value).split("|") if item)
    return selected


def finalize_project(
    project: Project,
    session: ReviewSession,
    output_dir: Path,
    *,
    section_selection_path: Path | None = None,
    require_verified: bool = False,
    exclude_image_border_fibers: bool = False,
) -> dict[str, Any]:
    """Finalize every fiber-typing image and write tables plus a manifest."""
    if session.project_id != project.project_id:
        raise FinalizationError("Review session belongs to a different project.")
    if session.model_version != project.model_version:
        raise FinalizationError(
            f"Review session used model {session.model_version!r}, but the project declares "
            f"{project.model_version!r}."
        )
    if project.qc_version and session.qc_version and session.qc_version != project.qc_version:
        raise FinalizationError(
            f"Review session used QC {session.qc_version!r}, but the project declares "
            f"{project.qc_version!r}."
        )
    selected = (
        load_selected_image_ids(section_selection_path)
        if section_selection_path is not None
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    finalized = [
        finalize_image(
            project,
            session,
            image,
            selected_image_ids=selected,
            require_verified=require_verified,
            exclude_image_border_fibers=exclude_image_border_fibers,
        )
        for image in project.images
        if Domain.FIBER_TYPING in image.applicable_domains
    ]
    for item in finalized:
        atomic_write_dataframe(output_dir / f"{item.image_id}_fibers_finalized.csv", item.table)
    if finalized:
        combined = pd.concat([item.table for item in finalized], ignore_index=True)
        atomic_write_dataframe(output_dir / "final_fiber_table.csv", combined)
    manifest = {
        "schema_version": FINALIZATION_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "project_id": project.project_id,
        "model_version": project.model_version,
        "qc_version": project.qc_version,
        "review_schema_version": session.schema_version,
        "policies": POLICIES,
        "section_selection_applied": selected is not None,
        "exclude_image_border_fibers": bool(exclude_image_border_fibers),
        "images": [
            {
                "image_id": item.image_id,
                "verification": item.verification,
                "input_sha256": item.input_digests,
                "counts": item.counts,
                "warnings": item.warnings,
            }
            for item in finalized
        ],
    }
    (output_dir / "finalization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
