"""Deterministic, GUI-independent fiber-type review queues."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from src.fiber_type_labels import to_biological_label
from src.review.project import Project
from src.review.schemas import Domain


class QueueSource(StrEnum):
    FLAGGED = "flagged"
    LOW_CONFIDENCE = "low_confidence"
    HIGH_ENTROPY = "high_entropy"
    LOW_PROBABILITY_MARGIN = "low_probability_margin"
    PROBABILITY_CONFLICTS = "probability_conflicts"
    RANDOM_AUDIT = "random_audit"
    FULL_AUDIT = "full_audit"


class RandomAuditScope(StrEnum):
    IMAGE = "image"
    MOUSE = "mouse"
    COHORT = "cohort"


@dataclass(frozen=True)
class FiberQueueItem:
    image_id: str
    mouse_id: str
    fiber_id: int
    model_fiber_type: str
    queue_source: QueueSource
    reason_code: str = ""
    confidence: float | None = None
    probability_margin: float | None = None
    normalized_entropy: float | None = None


def _id_column(table: pd.DataFrame) -> str:
    for name in ("fiber_id", "label"):
        if name in table.columns:
            return name
    raise ValueError("Fiber table must contain fiber_id or label")


def _prediction_column(table: pd.DataFrame) -> str:
    for name in ("predicted_type", "fiber_type", "model_prediction"):
        if name in table.columns:
            return name
    raise ValueError("Fiber table must contain predicted_type, fiber_type, or model_prediction")


def _number(row: pd.Series, names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            value = float(row[name])
            if np.isfinite(value):
                return value
    return None


def _as_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _probability_values(row: pd.Series) -> list[float]:
    values: list[float] = []
    for name in ("prob_i", "prob_iia", "prob_iib", "prob_iix"):
        if name in row.index and pd.notna(row[name]):
            value = float(row[name])
            if np.isfinite(value) and value >= 0:
                values.append(value)
    total = sum(values)
    return [value / total for value in values] if len(values) >= 2 and total > 0 else []


def _probability_metrics(row: pd.Series) -> tuple[float | None, float | None]:
    values = _probability_values(row)
    if not values:
        return None, None
    ordered = sorted(values)
    margin = ordered[-1] - ordered[-2]
    entropy = -sum(value * np.log(value) for value in values if value > 0) / np.log(len(values))
    return float(margin), float(entropy)


REASON_LABELS = {
    "flagged_by_model_or_qc": "low model confidence or margin",
    "fiber.weak_laminin_rim": "faint laminin outline",
    "fiber.thick_laminin": "thick laminin",
    "fiber.object_missing_table_row": "fiber missing from the table",
}


def describe_reason(reason_code: str) -> str:
    """Plain-language text for a queue item's reason codes."""
    parts = [part for part in str(reason_code).split("|") if part]
    return "; ".join(REASON_LABELS.get(part, part.replace("_", " ")) for part in parts)


def load_fiber_type_rows(project: Project, fiber_qc: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load canonical object rows while retaining model outputs as read-only input.

    ``fiber_qc`` (the project QC fiber table) adds per-fiber technical review reasons such as
    weak or thick laminin; they only affect which fibers are queued for review.
    """
    records: list[dict[str, object]] = []
    for image in project.images:
        if Domain.FIBER_TYPING not in image.applicable_domains:
            continue
        table_path = image.outputs.get("fiber_table")
        if table_path is None:
            raise ValueError(
                f"Image {image.image_id!r} applies fiber_typing but has no fiber_table"
            )
        try:
            table = pd.read_csv(table_path, low_memory=False)
        except Exception as exc:
            raise ValueError(f"Fiber table is unreadable: {table_path}: {exc}") from exc
        id_column = _id_column(table)
        prediction_column = _prediction_column(table)
        for _, row in table.iterrows():
            try:
                fiber_id = int(row[id_column])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid fiber ID in {table_path}: {row[id_column]!r}") from exc
            if fiber_id <= 0:
                continue
            margin, entropy = _probability_metrics(row)
            records.append(
                {
                    "image_id": image.image_id,
                    "mouse_id": image.mouse_id,
                    "fiber_id": fiber_id,
                    "model_fiber_type": to_biological_label(row[prediction_column]),
                    "confidence": _number(
                        row, ("model_confidence", "confidence", "max_probability")
                    ),
                    "probability_margin": _number(row, ("model_margin", "probability_margin"))
                    if _number(row, ("model_margin", "probability_margin")) is not None
                    else margin,
                    "normalized_entropy": _number(row, ("normalized_entropy",))
                    if _number(row, ("normalized_entropy",)) is not None
                    else entropy,
                    "needs_review": _as_bool(row.get("needs_review", False)),
                    "typing_signal_qc_flags": _as_text(row.get("typing_signal_qc_flags", "")),
                }
            )
    result = pd.DataFrame(records)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "image_id",
                "mouse_id",
                "fiber_id",
                "model_fiber_type",
                "confidence",
                "probability_margin",
                "normalized_entropy",
                "needs_review",
                "typing_signal_qc_flags",
            ]
        )
    if result.duplicated(["image_id", "fiber_id"]).any():
        raise ValueError("Fiber table IDs must be unique within each image")
    result["technical_flags"] = ""
    if fiber_qc is not None and not fiber_qc.empty and not result.empty:
        reasons = fiber_qc.loc[:, ["image_id", "fiber_id", "technical_reason_codes"]].copy()
        reasons["technical_reason_codes"] = reasons["technical_reason_codes"].fillna("").astype(str)
        reasons = reasons[reasons["technical_reason_codes"].ne("")]
        lookup = {
            (str(row.image_id), int(row.fiber_id)): row.technical_reason_codes
            for row in reasons.itertuples(index=False)
        }
        result["technical_flags"] = [
            lookup.get((str(image_id), int(fiber_id)), "")
            for image_id, fiber_id in zip(result["image_id"], result["fiber_id"], strict=True)
        ]
    return result


def build_fiber_type_queue(
    rows: pd.DataFrame,
    source: QueueSource | str,
    *,
    seed: int = 0,
    sample_size: int = 25,
    random_scope: RandomAuditScope | str = RandomAuditScope.COHORT,
) -> tuple[FiberQueueItem, ...]:
    """Build a stable queue; random audits are reproducible from their seed."""
    parsed_source = QueueSource(source)
    parsed_scope = RandomAuditScope(random_scope)
    required = {"image_id", "mouse_id", "fiber_id", "model_fiber_type"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Fiber rows are missing required columns: {missing}")
    work = rows.copy().sort_values(["image_id", "fiber_id"], kind="stable")
    reason = ""
    if parsed_source is QueueSource.FLAGGED:
        flags = work.get("typing_signal_qc_flags", pd.Series("", index=work.index)).fillna("")
        needs_review = (
            work.get("needs_review", pd.Series(False, index=work.index)).fillna(False).astype(bool)
        )
        model_flagged = needs_review | flags.astype(str).str.strip().ne("")
        technical = (
            work.get("technical_flags", pd.Series("", index=work.index)).fillna("").astype(str)
        )
        work = work.assign(
            queue_reason=[
                "|".join(
                    part for part in ("flagged_by_model_or_qc" if flagged else "", tech) if part
                )
                for flagged, tech in zip(model_flagged, technical, strict=True)
            ]
        )
        work = work[work["queue_reason"].ne("")]
        reason = "flagged_by_model_or_qc"
    elif parsed_source is QueueSource.LOW_CONFIDENCE:
        work = work[work["confidence"].notna()].sort_values(
            ["confidence", "image_id", "fiber_id"], kind="stable"
        )
        reason = "low_confidence"
    elif parsed_source is QueueSource.HIGH_ENTROPY:
        work = work[work["normalized_entropy"].notna()].sort_values(
            ["normalized_entropy", "image_id", "fiber_id"],
            ascending=[False, True, True],
            kind="stable",
        )
        reason = "high_entropy"
    elif parsed_source is QueueSource.LOW_PROBABILITY_MARGIN:
        work = work[work["probability_margin"].notna()].sort_values(
            ["probability_margin", "image_id", "fiber_id"], kind="stable"
        )
        reason = "low_probability_margin"
    elif parsed_source is QueueSource.PROBABILITY_CONFLICTS:
        work = work[work["probability_margin"].notna() & work["probability_margin"].le(0.05)]
        reason = "probability_conflict"
    elif parsed_source is QueueSource.RANDOM_AUDIT:
        if sample_size < 1:
            raise ValueError("sample_size must be positive")
        groups = {
            RandomAuditScope.COHORT: ["_cohort"],
            RandomAuditScope.MOUSE: ["mouse_id"],
            RandomAuditScope.IMAGE: ["image_id"],
        }
        work = work.assign(_cohort="cohort")
        selected: list[pd.DataFrame] = []
        for group_key, group in work.groupby(groups[parsed_scope], sort=True, dropna=False):
            group = group.sort_values(["image_id", "fiber_id"], kind="stable")
            count = min(sample_size, len(group))
            indices = random.Random(f"{seed}:{group_key}").sample(list(group.index), count)
            selected.append(group.loc[indices])
        work = (
            pd.concat(selected).sort_values(["image_id", "fiber_id"], kind="stable")
            if selected
            else work.iloc[0:0]
        )
        reason = f"random_audit:{parsed_scope.value}:seed={seed}"
    elif parsed_source is QueueSource.FULL_AUDIT:
        reason = "full_audit"

    return tuple(
        FiberQueueItem(
            image_id=str(row.image_id),
            mouse_id=str(row.mouse_id),
            fiber_id=int(row.fiber_id),
            model_fiber_type=str(row.model_fiber_type),
            queue_source=parsed_source,
            reason_code=getattr(row, "queue_reason", reason),
            confidence=None if pd.isna(row.confidence) else float(row.confidence),
            probability_margin=None
            if pd.isna(row.probability_margin)
            else float(row.probability_margin),
            normalized_entropy=None
            if pd.isna(row.normalized_entropy)
            else float(row.normalized_entropy),
        )
        for row in work.itertuples(index=False)
    )
