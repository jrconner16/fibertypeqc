"""Nucleus measurements and conservative association with cached fiber labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


def associate_nuclei(
    nuclei_labels: np.ndarray,
    fiber_labels: np.ndarray,
    *,
    min_overlap_fraction: float = 0.5,
    boundary_distance_px: float = 3.0,
    central_normalized_radius: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure nuclei and create explicit nucleus-to-fiber association rows.

    Nuclei with no fiber overlap are marked ``unassigned_or_interstitial``. Nuclei
    overlapping multiple fibers without a clear majority are marked ``ambiguous``.
    Assigned nuclei are categorized by distance to the fiber mask boundary; this is
    a geometric category, not an automatic myonucleus call.
    """
    nuclei = np.asarray(nuclei_labels, dtype=np.int32)
    fibers = np.asarray(fiber_labels, dtype=np.int32)
    if nuclei.shape != fibers.shape:
        raise ValueError(
            f"Nuclei and fiber labels must have the same shape: {nuclei.shape} != {fibers.shape}"
        )
    if not 0.0 <= min_overlap_fraction <= 1.0:
        raise ValueError("min_overlap_fraction must be between 0 and 1")

    max_nucleus = int(nuclei.max())
    areas = np.bincount(nuclei.ravel(), minlength=max_nucleus + 1)
    ys, xs = np.nonzero(nuclei)
    nucleus_ids = nuclei[ys, xs]
    y_sum = np.bincount(nucleus_ids, weights=ys, minlength=max_nucleus + 1)
    x_sum = np.bincount(nucleus_ids, weights=xs, minlength=max_nucleus + 1)
    fiber_distance = distance_transform_edt(fibers > 0)
    fiber_areas = np.bincount(fibers.ravel())
    nuclei_rows: list[dict[str, object]] = []
    links_rows: list[dict[str, object]] = []

    for nucleus_id in range(1, max_nucleus + 1):
        area = int(areas[nucleus_id])
        if area == 0:
            continue
        mask = nucleus_ids == nucleus_id
        nucleus_fibers = fibers[ys[mask], xs[mask]]
        nucleus_fibers = nucleus_fibers[nucleus_fibers > 0]
        if nucleus_fibers.size == 0:
            assigned_fiber = 0
            overlap_fraction = 0.0
            assignment_status = "unassigned_or_interstitial"
        else:
            fiber_ids, counts = np.unique(nucleus_fibers, return_counts=True)
            top = int(np.argmax(counts))
            assigned_fiber = int(fiber_ids[top])
            overlap_fraction = float(counts[top] / area)
            assignment_status = (
                "assigned" if overlap_fraction >= min_overlap_fraction else "ambiguous"
            )

        centroid_y = float(y_sum[nucleus_id] / area)
        centroid_x = float(x_sum[nucleus_id] / area)
        cy = min(max(int(round(centroid_y)), 0), fibers.shape[0] - 1)
        cx = min(max(int(round(centroid_x)), 0), fibers.shape[1] - 1)
        boundary_distance = (
            float(fiber_distance[cy, cx]) if assignment_status == "assigned" else float("nan")
        )
        normalized_radius = float("nan")
        category = assignment_status
        if assignment_status == "assigned":
            equivalent_radius = float(np.sqrt(fiber_areas[assigned_fiber] / np.pi))
            normalized_radius = boundary_distance / max(equivalent_radius, 1e-6)
            if boundary_distance <= boundary_distance_px:
                category = "boundary_associated"
            elif normalized_radius >= central_normalized_radius:
                category = "central_interior"
            else:
                category = "peripheral_associated"
            links_rows.append(
                {
                    "nucleus_id": nucleus_id,
                    "fiber_id": assigned_fiber,
                    "assignment_status": assignment_status,
                    "association_category": category,
                    "overlap_fraction": overlap_fraction,
                }
            )

        nuclei_rows.append(
            {
                "nucleus_id": nucleus_id,
                "area_px": area,
                "centroid_y_px": centroid_y,
                "centroid_x_px": centroid_x,
                "assigned_fiber_id": assigned_fiber,
                "assignment_status": assignment_status,
                "association_category": category,
                "overlap_fraction": overlap_fraction,
                "distance_to_boundary_px": boundary_distance,
                "normalized_radial_position": normalized_radius,
            }
        )

    nuclei_table = pd.DataFrame(nuclei_rows)
    links_table = pd.DataFrame(links_rows)
    if links_table.empty:
        links_table = pd.DataFrame(
            columns=[
                "nucleus_id",
                "fiber_id",
                "assignment_status",
                "association_category",
                "overlap_fraction",
            ]
        )
    return nuclei_table, links_table


def summarize_fiber_nuclei(
    fiber_labels: np.ndarray,
    nuclei_table: pd.DataFrame,
) -> pd.DataFrame:
    """Return per-fiber counts for associated and central nuclei."""
    fiber_ids = np.unique(fiber_labels)
    fiber_ids = fiber_ids[fiber_ids > 0]
    out = pd.DataFrame({"fiber_id": fiber_ids.astype(np.int32)})
    if nuclei_table.empty:
        out["associated_nuclei_count"] = 0
        out["central_nuclei_count"] = 0
        out["centrally_nucleated"] = False
        return out
    assigned = nuclei_table[nuclei_table["assignment_status"].eq("assigned")].copy()
    counts = assigned.groupby("assigned_fiber_id").size().rename("associated_nuclei_count")
    central = (
        assigned[assigned["association_category"].eq("central_interior")]
        .groupby("assigned_fiber_id")
        .size()
        .rename("central_nuclei_count")
    )
    out = out.merge(counts, left_on="fiber_id", right_index=True, how="left")
    out = out.merge(central, left_on="fiber_id", right_index=True, how="left")
    out[["associated_nuclei_count", "central_nuclei_count"]] = (
        out[["associated_nuclei_count", "central_nuclei_count"]].fillna(0).astype(int)
    )
    out["centrally_nucleated"] = out["central_nuclei_count"] > 0
    return out
