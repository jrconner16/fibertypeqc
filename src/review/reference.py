"""Blind reference labels: field membership and export (no GUI imports).

Reference labels live under ``<review project>/reference/<reviewer>/`` and are never applied to
results. The export gathers every reviewer's labels into one table, keeps the model's output in
separate ``model_*`` columns, and reports which drawn fields were labelled exhaustively.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops

from fibertypeqc.artifacts import file_sha256, git_commit
from src.review.project import Project
from src.review.queues import load_fiber_type_rows
from src.review.region_review import _point_in_polygon
from src.review.schemas import RegionAnnotation, RegionKind
from src.review.session import ReviewSession
from src.review.storage import atomic_write_dataframe, load_session
from src.review.timing import SESSIONS_FILENAME

REFERENCE_SCHEMA_VERSION = "fibertypeqc.reference_export.v1"
REFERENCE_FIELD_ROLE = "reference_field"
LABEL_COLUMNS = (
    "image_id",
    "mouse_id",
    "section_id",
    "fiber_id",
    "reviewer",
    "reference_label",
    "sampling",
    "field_name",
    "blind",
    "inputs_unchanged",
    "labelled_at",
    "model_fiber_type",
    "model_confidence",
    "model_margin",
    "model_needs_review",
)


def fiber_centroids(labels: np.ndarray) -> dict[int, tuple[float, float]]:
    """Fiber centroids as (x, y) in full-resolution image coordinates."""
    return {
        int(region.label): (float(region.centroid[1]), float(region.centroid[0]))
        for region in regionprops(labels)
    }


def reference_fields(session: ReviewSession, image_id: str) -> list[RegionAnnotation]:
    return [
        region
        for region in session.regions
        if region.image_id == image_id
        and region.kind is RegionKind.ANALYSIS_ROI
        and region.role == REFERENCE_FIELD_ROLE
    ]


def fibers_by_field(
    fields: list[RegionAnnotation], centroids: dict[int, tuple[float, float]]
) -> dict[int, str]:
    """Fiber ID -> field name for fibers whose centroid is inside exactly one field."""
    assigned: dict[int, str] = {}
    for fiber_id, (x, y) in centroids.items():
        inside = [f.name for f in fields if _point_in_polygon(x, y, f.geometry) == "inside"]
        if len(inside) == 1:
            assigned[fiber_id] = inside[0]
    return assigned


def _reviewer_sessions(project: Project) -> dict[str, tuple[Project, ReviewSession]]:
    root = project.root / "reference"
    found: dict[str, tuple[Project, ReviewSession]] = {}
    for directory in sorted(path for path in root.glob("*") if path.is_dir()):
        view = project.for_reference(directory.name)
        if view.review_state_path.is_file():
            session = load_session(view.review_state_path, expected_project_id=project.project_id)
            found[directory.name] = (view, session)
    return found


def _kappa(first: pd.Series, second: pd.Series) -> float | None:
    """Cohen's kappa for two reviewers' labels on the same fibers."""
    categories = sorted(set(first) | set(second))
    table = pd.crosstab(
        pd.Categorical(first, categories=categories),
        pd.Categorical(second, categories=categories),
        dropna=False,
    ).to_numpy(dtype=float)
    total = table.sum()
    if not total:
        return None
    observed = np.trace(table) / total
    expected = float((table.sum(axis=0) * table.sum(axis=1)).sum()) / total**2
    return None if expected >= 1 else float((observed - expected) / (1 - expected))


def export_reference(project: Project, output_dir: Path) -> dict[str, Any]:
    """Write reference labels, field coverage, reviewer agreement, and review time."""
    from src.review.finalization import image_input_fingerprints

    sessions = _reviewer_sessions(project)
    if not sessions:
        raise ValueError(
            f"No blind reference labels found under {project.root / 'reference'}. "
            "Label some with: python -m fibertypeqc review PROJECT --blind --reviewer NAME"
        )
    model = load_fiber_type_rows(project).set_index(["image_id", "fiber_id"])
    centroids: dict[str, dict[int, tuple[float, float]]] = {}
    digests: dict[str, dict[str, str]] = {}

    def image_centroids(image_id: str) -> dict[int, tuple[float, float]]:
        if image_id not in centroids:
            path = project.image(image_id).outputs["fiber_labels"]
            centroids[image_id] = fiber_centroids(np.asarray(tifffile.imread(path)))
        return centroids[image_id]

    label_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    time_tables: list[pd.DataFrame] = []
    for reviewer, (view, session) in sessions.items():
        field_of: dict[str, dict[int, str]] = {}
        for image in project.images:
            fields = reference_fields(session, image.image_id)
            if fields:
                field_of[image.image_id] = fibers_by_field(fields, image_centroids(image.image_id))
        labelled: dict[str, set[int]] = {}
        for decision in session.object_decisions:
            image = project.image(decision.image_id)
            if decision.image_id not in digests:
                digests[decision.image_id] = image_input_fingerprints(image)
            recorded = session.input_fingerprints.get(decision.image_id)
            field_name = field_of.get(decision.image_id, {}).get(decision.fiber_id, "")
            sampling = (
                "reference_field"
                if field_name
                else "random_sample"
                if decision.queue_source == "random_audit"
                else "whole_section"
                if decision.queue_source == "full_audit"
                else decision.queue_source
            )
            key = (decision.image_id, decision.fiber_id)
            predicted = model.loc[key].to_dict() if key in model.index else {}
            label_rows.append(
                {
                    "image_id": decision.image_id,
                    "mouse_id": image.mouse_id,
                    "section_id": image.section_id,
                    "fiber_id": decision.fiber_id,
                    "reviewer": reviewer,
                    "reference_label": decision.reviewed_fiber_type,
                    "sampling": sampling,
                    "field_name": field_name,
                    "blind": True,
                    "inputs_unchanged": recorded is not None
                    and recorded == digests[decision.image_id],
                    "labelled_at": decision.timestamp,
                    "model_fiber_type": predicted.get("model_fiber_type"),
                    "model_confidence": predicted.get("confidence"),
                    "model_margin": predicted.get("probability_margin"),
                    "model_needs_review": predicted.get("needs_review"),
                }
            )
            labelled.setdefault(decision.image_id, set()).add(decision.fiber_id)
        for image_id, assignment in field_of.items():
            for name in sorted(set(assignment.values())):
                members = {fiber for fiber, field in assignment.items() if field == name}
                done = len(members & labelled.get(image_id, set()))
                field_rows.append(
                    {
                        "image_id": image_id,
                        "reviewer": reviewer,
                        "field_name": name,
                        "n_fibers": len(members),
                        "n_labelled": done,
                        "exhaustive": bool(members) and done == len(members),
                    }
                )
        sessions_path = view.review_directory / SESSIONS_FILENAME
        if sessions_path.is_file():
            time_tables.append(pd.read_csv(sessions_path))

    labels = pd.DataFrame(label_rows, columns=list(LABEL_COLUMNS)).sort_values(
        ["image_id", "fiber_id", "reviewer"], kind="stable"
    )
    fields = pd.DataFrame(
        field_rows,
        columns=["image_id", "reviewer", "field_name", "n_fibers", "n_labelled", "exhaustive"],
    )
    agreement_rows = []
    wide = labels.pivot_table(
        index=["image_id", "fiber_id"], columns="reviewer", values="reference_label", aggfunc="last"
    )
    for first, second in combinations(sorted(sessions), 2):
        shared = wide[[first, second]].dropna()
        agreement_rows.append(
            {
                "reviewer_a": first,
                "reviewer_b": second,
                "n_shared_fibers": len(shared),
                "agreement": float((shared[first] == shared[second]).mean())
                if len(shared)
                else None,
                "cohen_kappa": _kappa(shared[first], shared[second]) if len(shared) else None,
            }
        )
    agreement = pd.DataFrame(
        agreement_rows,
        columns=["reviewer_a", "reviewer_b", "n_shared_fibers", "agreement", "cohen_kappa"],
    )
    time = pd.concat(time_tables, ignore_index=True) if time_tables else pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "reference_labels.csv": labels,
        "reference_fields.csv": fields,
        "reviewer_agreement.csv": agreement,
        "review_sessions.csv": time,
    }
    for name, table in written.items():
        atomic_write_dataframe(output_dir / name, table)
    manifest = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "project_id": project.project_id,
        "model_version": project.model_version,
        "reviewers": {
            reviewer: int((labels["reviewer"] == reviewer).sum()) for reviewer in sessions
        },
        "n_labels": int(len(labels)),
        "n_fibers_labelled": int(labels[["image_id", "fiber_id"]].drop_duplicates().shape[0]),
        "n_fields": int(len(fields)),
        "n_exhaustive_fields": int(fields["exhaustive"].sum()) if len(fields) else 0,
        "note": (
            "model_* columns are the model's output for comparison; reviewers did not see them. "
            "Labels with sampling other than reference_field or whole_section are a sample, not "
            "an exhaustive field."
        ),
        "files_sha256": {name: file_sha256(output_dir / name) for name in written},
    }
    (output_dir / "reference_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
