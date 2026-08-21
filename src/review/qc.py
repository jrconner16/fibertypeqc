"""Headless artifact-derived QC metrics for review projects."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile

from src.review.project import Project, ProjectImage
from src.review.qc_rules import QCRuleConfig, RuleSeverity, evaluate_rules
from src.review.schemas import Domain

IMAGE_QC_SCHEMA_VERSION = "review_image_qc.v1"
FIBER_QC_SCHEMA_VERSION = "review_fiber_qc.v1"
NUCLEUS_QC_SCHEMA_VERSION = "review_nucleus_qc.v1"

IMAGE_METRIC_COLUMNS = [
    "fiber_labels_available",
    "fiber_labels_readable",
    "fiber_labels_valid",
    "fiber_table_available",
    "fiber_table_readable",
    "fiber_table_ids_valid",
    "nuclei_labels_available",
    "nuclei_labels_readable",
    "nuclei_labels_valid",
    "nuclei_table_available",
    "nuclei_table_readable",
    "nuclei_table_ids_valid",
    "label_shape_match",
    "fiber_count",
    "image_pixel_count",
    "segmented_pixel_count",
    "segmented_image_fraction",
    "median_fiber_area_px",
    "border_touching_fiber_count",
    "border_touching_fiber_fraction",
    "fiber_id_mismatch_fraction",
    "typing_row_count",
    "prediction_available",
    "unknown_fraction",
    "needs_review_fraction",
    "probability_row_count",
    "probability_coverage",
    "mean_max_probability",
    "mean_probability_margin",
    "mean_normalized_entropy",
    "type_counts_json",
    "nucleus_count",
    "nucleus_pixel_count",
    "nucleus_image_fraction",
    "median_nucleus_area_px",
    "nucleus_id_mismatch_fraction",
    "nucleus_association_available",
    "unassigned_nucleus_fraction",
    "ambiguous_nucleus_fraction",
    "mean_association_overlap",
    "assigned_nuclei_per_fiber",
]

IMAGE_QC_COLUMNS = [
    "schema_version",
    "qc_version",
    "rules_version",
    "model_version",
    "computed_at",
    "project_id",
    "image_id",
    "mouse_id",
    "section_id",
    "domain",
    "applicable",
    "status",
    "hard_fail",
    "technical_quality_score",
    "review_priority",
    "reason_codes",
    "reason_details_json",
    "artifact_paths_json",
    *IMAGE_METRIC_COLUMNS,
]

FIBER_QC_COLUMNS = [
    "schema_version",
    "qc_version",
    "rules_version",
    "model_version",
    "computed_at",
    "project_id",
    "image_id",
    "mouse_id",
    "section_id",
    "fiber_id",
    "area_px",
    "touches_image_border",
    "predicted_type",
    "prob_i",
    "prob_iia",
    "prob_iib",
    "prob_iix",
    "max_probability",
    "probability_margin",
    "normalized_entropy",
    "needs_review",
    "technical_reason_codes",
    "review_priority",
]

NUCLEUS_QC_COLUMNS = [
    "schema_version",
    "qc_version",
    "rules_version",
    "model_version",
    "computed_at",
    "project_id",
    "image_id",
    "mouse_id",
    "section_id",
    "nucleus_id",
    "area_px",
    "assigned_fiber_id",
    "assignment_status",
    "association_category",
    "overlap_fraction",
    "distance_to_boundary_px",
    "normalized_radial_position",
    "technical_reason_codes",
    "review_priority",
]

PROBABILITY_COLUMNS = ("prob_i", "prob_iia", "prob_iib", "prob_iix")
PREDICTION_COLUMNS = ("predicted_type", "fiber_type", "model_prediction")


@dataclass(frozen=True)
class QCResult:
    image_qc: pd.DataFrame
    fiber_qc: pd.DataFrame
    nucleus_qc: pd.DataFrame


@dataclass(frozen=True)
class _LabelArtifact:
    path: Path | None
    available: bool
    readable: bool | None
    valid: bool | None
    labels: np.ndarray | None


@dataclass(frozen=True)
class _TableArtifact:
    path: Path | None
    available: bool
    readable: bool | None
    table: pd.DataFrame | None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def _json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _read_label_artifact(path: Path | None) -> _LabelArtifact:
    if path is None or not path.is_file():
        return _LabelArtifact(path, False, None, None, None)
    try:
        raw = np.asarray(tifffile.imread(path))
    except Exception:
        return _LabelArtifact(path, True, False, None, None)
    valid = bool(
        raw.ndim == 2
        and raw.size > 0
        and np.issubdtype(raw.dtype, np.number)
        and np.all(np.isfinite(raw))
        and np.all(raw >= 0)
        and np.all(raw == np.floor(raw))
    )
    labels = raw.astype(np.int64, copy=False) if valid else None
    return _LabelArtifact(path, True, True, valid, labels)


def _read_table_artifact(path: Path | None) -> _TableArtifact:
    if path is None or not path.is_file():
        return _TableArtifact(path, False, None, None)
    try:
        table = pd.read_csv(path, low_memory=False)
    except Exception:
        return _TableArtifact(path, True, False, None)
    return _TableArtifact(path, True, True, table)


def _clean_id_table(
    table: pd.DataFrame | None,
    candidates: tuple[str, ...],
) -> tuple[pd.DataFrame, str | None, bool | None]:
    if table is None:
        return pd.DataFrame(), None, None
    id_column = next((column for column in candidates if column in table.columns), None)
    if id_column is None:
        return table.iloc[0:0].copy(), None, False
    numeric = pd.to_numeric(table[id_column], errors="coerce")
    valid_values = (
        numeric.notna()
        & np.isfinite(numeric)
        & numeric.gt(0)
        & numeric.eq(np.floor(numeric))
    )
    valid_ids = numeric[valid_values].astype(np.int64)
    duplicates = valid_ids.duplicated(keep=False)
    ids_valid = bool(valid_values.all() and not duplicates.any())
    clean = table.loc[valid_values].copy()
    clean["_object_id"] = valid_ids.to_numpy()
    clean = clean.drop_duplicates("_object_id", keep="first").reset_index(drop=True)
    return clean, id_column, ids_valid


def _mask_geometry(labels: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, set[int]]:
    if labels is None:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), set()
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    if not ids.size:
        return ids.astype(np.int64), counts.astype(np.int64), set()
    border_values = np.concatenate(
        (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1])
    )
    border_ids = {int(value) for value in np.unique(border_values) if value > 0}
    return ids.astype(np.int64), counts.astype(np.int64), border_ids


def _id_mismatch_fraction(mask_ids: np.ndarray, table_ids: np.ndarray) -> float | None:
    mask_set = {int(value) for value in mask_ids}
    table_set = {int(value) for value in table_ids}
    union = mask_set | table_set
    if not union:
        return None
    return len(mask_set ^ table_set) / len(union)


def _as_bool(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    return None


def _probability_metrics(row: pd.Series) -> dict[str, float | None]:
    values: list[float] = []
    output: dict[str, float | None] = {column: None for column in PROBABILITY_COLUMNS}
    for column in PROBABILITY_COLUMNS:
        if column not in row.index:
            continue
        value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        numeric = float(value)
        output[column] = numeric
        if np.isfinite(numeric) and numeric >= 0:
            values.append(numeric)
    if len(values) < 2 or sum(values) <= 0:
        return {
            **output,
            "max_probability": None,
            "probability_margin": None,
            "normalized_entropy": None,
        }
    probabilities = np.asarray(values, dtype=np.float64)
    probabilities /= probabilities.sum()
    ordered = np.sort(probabilities)
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * np.log(positive)).sum() / np.log(len(probabilities)))
    return {
        **output,
        "max_probability": float(ordered[-1]),
        "probability_margin": float(ordered[-1] - ordered[-2]),
        "normalized_entropy": entropy,
    }


def _base_provenance(
    project: Project,
    image: ProjectImage,
    rules: QCRuleConfig,
    computed_at: str,
) -> dict[str, Any]:
    return {
        "qc_version": rules.qc_version,
        "rules_version": rules.rules_version,
        "model_version": project.model_version,
        "computed_at": computed_at,
        "project_id": project.project_id,
        "image_id": image.image_id,
        "mouse_id": image.mouse_id,
        "section_id": image.section_id,
    }


def _image_row(
    *,
    project: Project,
    image: ProjectImage,
    domain: Domain,
    applicable: bool,
    rules: QCRuleConfig,
    computed_at: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "schema_version": IMAGE_QC_SCHEMA_VERSION,
        **_base_provenance(project, image, rules, computed_at),
        "domain": domain.value,
        "applicable": applicable,
        "status": "not_applicable",
        "hard_fail": False,
        "technical_quality_score": None,
        "review_priority": 0,
        "reason_codes": "",
        "reason_details_json": "[]",
        "artifact_paths_json": _json(
            {key: str(value) for key, value in sorted(image.outputs.items())}
        ),
        **{column: None for column in IMAGE_METRIC_COLUMNS},
    }
    row.update(metrics)
    if not applicable:
        return row

    reasons = evaluate_rules(rules, domain, row)
    hard_count = sum(
        reason["severity"] == RuleSeverity.HARD_FAIL.value for reason in reasons
    )
    review_count = sum(
        reason["severity"] == RuleSeverity.REVIEW.value for reason in reasons
    )
    if hard_count:
        status = "fail"
        score = 0.0
    elif review_count:
        status = "review"
        score = 0.5
    else:
        status = "pass"
        score = 1.0
    row.update(
        {
            "status": status,
            "hard_fail": hard_count > 0,
            "technical_quality_score": score,
            "review_priority": 100 * hard_count + 10 * review_count,
            "reason_codes": "|".join(reason["reason_code"] for reason in reasons),
            "reason_details_json": _json(reasons),
        }
    )
    return row


def _fiber_metrics_and_rows(
    *,
    project: Project,
    image: ProjectImage,
    rules: QCRuleConfig,
    computed_at: str,
    labels_artifact: _LabelArtifact,
    table_artifact: _TableArtifact,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    labels = labels_artifact.labels
    mask_ids, areas, border_ids = _mask_geometry(labels)
    clean_table, _, table_ids_valid = _clean_id_table(
        table_artifact.table, ("fiber_id", "label")
    )
    table_ids = (
        clean_table["_object_id"].to_numpy(dtype=np.int64)
        if "_object_id" in clean_table
        else np.array([], dtype=np.int64)
    )
    segmentation_metrics: dict[str, Any] = {
        "fiber_labels_available": labels_artifact.available,
        "fiber_labels_readable": labels_artifact.readable,
        "fiber_labels_valid": labels_artifact.valid,
        "fiber_table_available": table_artifact.available,
        "fiber_table_readable": table_artifact.readable,
        "fiber_table_ids_valid": table_ids_valid,
    }
    if labels is not None:
        fiber_count = int(mask_ids.size)
        image_pixels = int(labels.size)
        segmented_pixels = int((labels > 0).sum())
        segmentation_metrics.update(
            {
                "fiber_count": fiber_count,
                "image_pixel_count": image_pixels,
                "segmented_pixel_count": segmented_pixels,
                "segmented_image_fraction": segmented_pixels / image_pixels,
                "median_fiber_area_px": float(np.median(areas)) if areas.size else None,
                "border_touching_fiber_count": len(border_ids),
                "border_touching_fiber_fraction": (
                    len(border_ids) / fiber_count if fiber_count else None
                ),
                "fiber_id_mismatch_fraction": (
                    _id_mismatch_fraction(mask_ids, table_ids)
                    if table_artifact.table is not None
                    else None
                ),
            }
        )

    prediction_column = next(
        (column for column in PREDICTION_COLUMNS if column in clean_table.columns),
        None,
    )
    typing_count = int(len(clean_table)) if table_artifact.table is not None else None
    typing_metrics: dict[str, Any] = {
        "fiber_table_available": table_artifact.available,
        "fiber_table_readable": table_artifact.readable,
        "fiber_table_ids_valid": table_ids_valid,
        "typing_row_count": typing_count,
        "prediction_available": (
            prediction_column is not None if table_artifact.table is not None else None
        ),
    }

    probability_rows: list[dict[str, float | None]] = []
    if typing_count is not None and typing_count > 0:
        predictions = (
            clean_table[prediction_column].fillna("").astype(str).str.strip().str.lower()
            if prediction_column is not None
            else pd.Series([""] * typing_count)
        )
        unknown = predictions.isin({"", "unknown", "uncertain"})
        typing_metrics["unknown_fraction"] = float(unknown.mean())
        typing_metrics["type_counts_json"] = _json(
            {str(key): int(value) for key, value in predictions.value_counts().items()}
        )
        if "needs_review" in clean_table.columns:
            parsed_needs_review = clean_table["needs_review"].map(_as_bool)
            known = parsed_needs_review.notna()
            typing_metrics["needs_review_fraction"] = (
                float(parsed_needs_review[known].astype(bool).mean()) if known.any() else None
            )
        for _, table_row in clean_table.iterrows():
            probability_rows.append(_probability_metrics(table_row))
        usable = [
            values
            for values in probability_rows
            if values["max_probability"] is not None
        ]
        typing_metrics.update(
            {
                "probability_row_count": len(usable),
                "probability_coverage": len(usable) / typing_count,
                "mean_max_probability": (
                    float(np.mean([values["max_probability"] for values in usable]))
                    if usable
                    else None
                ),
                "mean_probability_margin": (
                    float(np.mean([values["probability_margin"] for values in usable]))
                    if usable
                    else None
                ),
                "mean_normalized_entropy": (
                    float(np.mean([values["normalized_entropy"] for values in usable]))
                    if usable
                    else None
                ),
            }
        )
    elif typing_count == 0:
        typing_metrics.update(
            {
                "probability_row_count": 0,
                "probability_coverage": None,
            }
        )

    table_lookup = (
        clean_table.set_index("_object_id", drop=False) if not clean_table.empty else None
    )
    probability_lookup = {
        int(clean_table.iloc[index]["_object_id"]): values
        for index, values in enumerate(probability_rows)
    }
    fiber_rows: list[dict[str, Any]] = []
    for fiber_id, area in zip(mask_ids, areas, strict=True):
        object_id = int(fiber_id)
        table_row = (
            table_lookup.loc[object_id]
            if table_lookup is not None and object_id in table_lookup.index
            else None
        )
        reasons = []
        if table_artifact.table is not None and table_row is None:
            reasons.append("fiber.object_missing_table_row")
        probability = probability_lookup.get(object_id, {})
        fiber_rows.append(
            {
                "schema_version": FIBER_QC_SCHEMA_VERSION,
                **_base_provenance(project, image, rules, computed_at),
                "fiber_id": object_id,
                "area_px": int(area),
                "touches_image_border": object_id in border_ids,
                "predicted_type": (
                    str(table_row[prediction_column])
                    if table_row is not None
                    and prediction_column is not None
                    and not pd.isna(table_row[prediction_column])
                    else ""
                ),
                **{column: probability.get(column) for column in PROBABILITY_COLUMNS},
                "max_probability": probability.get("max_probability"),
                "probability_margin": probability.get("probability_margin"),
                "normalized_entropy": probability.get("normalized_entropy"),
                "needs_review": (
                    _as_bool(table_row["needs_review"])
                    if table_row is not None and "needs_review" in table_row.index
                    else None
                ),
                "technical_reason_codes": "|".join(reasons),
                "review_priority": 10 * len(reasons),
            }
        )
    return segmentation_metrics, typing_metrics, fiber_rows, clean_table


def _nucleus_metrics_and_rows(
    *,
    project: Project,
    image: ProjectImage,
    rules: QCRuleConfig,
    computed_at: str,
    nuclei_artifact: _LabelArtifact,
    nuclei_table_artifact: _TableArtifact,
    fiber_labels: np.ndarray | None,
    fiber_count: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = nuclei_artifact.labels
    mask_ids, areas, _ = _mask_geometry(labels)
    clean_table, _, table_ids_valid = _clean_id_table(
        nuclei_table_artifact.table, ("nucleus_id", "label")
    )
    table_ids = (
        clean_table["_object_id"].to_numpy(dtype=np.int64)
        if "_object_id" in clean_table
        else np.array([], dtype=np.int64)
    )
    association_columns = {
        "assigned_fiber_id",
        "assignment_status",
        "overlap_fraction",
    }
    association_available = (
        association_columns.issubset(clean_table.columns)
        if nuclei_table_artifact.table is not None
        else None
    )
    metrics: dict[str, Any] = {
        "nuclei_labels_available": nuclei_artifact.available,
        "nuclei_labels_readable": nuclei_artifact.readable,
        "nuclei_labels_valid": nuclei_artifact.valid,
        "nuclei_table_available": nuclei_table_artifact.available,
        "nuclei_table_readable": nuclei_table_artifact.readable,
        "nuclei_table_ids_valid": table_ids_valid,
        "nucleus_association_available": association_available,
        "label_shape_match": (
            labels.shape == fiber_labels.shape
            if labels is not None and fiber_labels is not None
            else None
        ),
    }
    if labels is not None:
        metrics.update(
            {
                "nucleus_count": int(mask_ids.size),
                "nucleus_pixel_count": int((labels > 0).sum()),
                "nucleus_image_fraction": float((labels > 0).sum() / labels.size),
                "median_nucleus_area_px": (
                    float(np.median(areas)) if areas.size else None
                ),
                "nucleus_id_mismatch_fraction": (
                    _id_mismatch_fraction(mask_ids, table_ids)
                    if nuclei_table_artifact.table is not None
                    else None
                ),
            }
        )
    if not clean_table.empty and association_available:
        statuses = clean_table["assignment_status"].fillna("").astype(str)
        assigned_ids = pd.to_numeric(
            clean_table["assigned_fiber_id"], errors="coerce"
        ).fillna(0)
        unassigned = statuses.eq("unassigned_or_interstitial") | assigned_ids.eq(0)
        ambiguous = statuses.eq("ambiguous")
        overlaps = pd.to_numeric(clean_table["overlap_fraction"], errors="coerce")
        metrics.update(
            {
                "unassigned_nucleus_fraction": float(unassigned.mean()),
                "ambiguous_nucleus_fraction": float(ambiguous.mean()),
                "mean_association_overlap": (
                    float(overlaps.mean()) if overlaps.notna().any() else None
                ),
                "assigned_nuclei_per_fiber": (
                    float(statuses.eq("assigned").sum() / fiber_count)
                    if fiber_count
                    else None
                ),
            }
        )

    table_lookup = (
        clean_table.set_index("_object_id", drop=False) if not clean_table.empty else None
    )
    nucleus_rows: list[dict[str, Any]] = []
    association_fields = (
        "assigned_fiber_id",
        "assignment_status",
        "association_category",
        "overlap_fraction",
        "distance_to_boundary_px",
        "normalized_radial_position",
    )
    for nucleus_id, area in zip(mask_ids, areas, strict=True):
        object_id = int(nucleus_id)
        table_row = (
            table_lookup.loc[object_id]
            if table_lookup is not None and object_id in table_lookup.index
            else None
        )
        reasons = []
        if nuclei_table_artifact.table is not None and table_row is None:
            reasons.append("nucleus.object_missing_table_row")
        row = {
            "schema_version": NUCLEUS_QC_SCHEMA_VERSION,
            **_base_provenance(project, image, rules, computed_at),
            "nucleus_id": object_id,
            "area_px": int(area),
            "technical_reason_codes": "|".join(reasons),
            "review_priority": 10 * len(reasons),
        }
        for field in association_fields:
            value = table_row[field] if table_row is not None and field in table_row.index else None
            row[field] = None if pd.isna(value) else value
        nucleus_rows.append(row)
    return metrics, nucleus_rows


def generate_project_qc(project: Project, rules: QCRuleConfig) -> QCResult:
    """Generate deterministic, headless QC tables from declared prediction artifacts."""
    computed_at = datetime.now(UTC).isoformat()
    image_rows: list[dict[str, Any]] = []
    fiber_rows: list[dict[str, Any]] = []
    nucleus_rows: list[dict[str, Any]] = []

    for image in project.images:
        fiber_labels_artifact = _read_label_artifact(image.outputs.get("fiber_labels"))
        fiber_table_artifact = _read_table_artifact(image.outputs.get("fiber_table"))
        nuclei_labels_artifact = _read_label_artifact(image.outputs.get("nuclei_labels"))
        nuclei_table_artifact = _read_table_artifact(image.outputs.get("nuclei_table"))

        segmentation_metrics, typing_metrics, image_fibers, _ = _fiber_metrics_and_rows(
            project=project,
            image=image,
            rules=rules,
            computed_at=computed_at,
            labels_artifact=fiber_labels_artifact,
            table_artifact=fiber_table_artifact,
        )
        fiber_rows.extend(image_fibers)
        fiber_count = segmentation_metrics.get("fiber_count")
        nuclei_metrics, image_nuclei = _nucleus_metrics_and_rows(
            project=project,
            image=image,
            rules=rules,
            computed_at=computed_at,
            nuclei_artifact=nuclei_labels_artifact,
            nuclei_table_artifact=nuclei_table_artifact,
            fiber_labels=fiber_labels_artifact.labels,
            fiber_count=fiber_count,
        )
        nucleus_rows.extend(image_nuclei)

        domain_metrics = {
            Domain.FIBER_SEGMENTATION: segmentation_metrics,
            Domain.FIBER_TYPING: typing_metrics,
            Domain.NUCLEI: nuclei_metrics,
        }
        for domain in Domain:
            image_rows.append(
                _image_row(
                    project=project,
                    image=image,
                    domain=domain,
                    applicable=domain in image.applicable_domains,
                    rules=rules,
                    computed_at=computed_at,
                    metrics=domain_metrics[domain],
                )
            )

    return QCResult(
        image_qc=pd.DataFrame(image_rows, columns=IMAGE_QC_COLUMNS),
        fiber_qc=pd.DataFrame(fiber_rows, columns=FIBER_QC_COLUMNS),
        nucleus_qc=pd.DataFrame(nucleus_rows, columns=NUCLEUS_QC_COLUMNS),
    )
