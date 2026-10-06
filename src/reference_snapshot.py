"""Freeze a project's outputs as a reference snapshot and check later runs against it.

A snapshot records, per section, the fiber count, class counts, a digest of the label mask, and a
digest of the fiber calls, plus the model digest and software versions. Checking compares a run
with a snapshot at two levels:

- **exact**: when a section's mask is identical to the reference, its fiber calls must be
  identical too (same model, same features, same calls);
- **tolerant**: when segmentation was re-run and the mask differs (another device or Cellpose
  build), the fiber count and class proportions must agree within stated tolerances.

Snapshots contain image IDs and counts from your data; keep snapshots of private data private.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile

from fibertypeqc.artifacts import git_commit

SNAPSHOT_SCHEMA = "fibertypeqc.reference_snapshot.v1"
DEFAULT_COUNT_TOLERANCE = 0.02  # relative difference in fiber count
DEFAULT_COMPOSITION_TOLERANCE = 0.02  # absolute difference in any class proportion


def _mask_digest(path: Path) -> str:
    labels = np.ascontiguousarray(tifffile.imread(path))
    digest = sha256(str(labels.shape).encode())
    digest.update(labels.astype(np.int64).tobytes())
    return digest.hexdigest()


def _calls_digest(fibers: pd.DataFrame) -> str:
    id_column = "fiber_id" if "fiber_id" in fibers.columns else "label"
    rows = sorted(zip(fibers[id_column].astype(int), fibers["fiber_type"].astype(str), strict=True))
    return sha256("\n".join(f"{fiber}:{call}" for fiber, call in rows).encode()).hexdigest()


def section_snapshot(section_dir: Path, image_id: str) -> dict[str, Any]:
    fibers = pd.read_csv(section_dir / f"{image_id}_fibers.csv")
    run = json.loads((section_dir / f"{image_id}_run.json").read_text(encoding="utf-8"))
    counts = fibers["fiber_type"].astype(str).value_counts().sort_index()
    return {
        "image_id": image_id,
        "n_fibers": int(len(fibers)),
        "class_counts": {str(name): int(value) for name, value in counts.items()},
        "mask_sha256": _mask_digest(section_dir / f"{image_id}_cellpose_labels.tif"),
        "calls_sha256": _calls_digest(fibers),
        "model_sha256": run.get("classifier_sha256"),
        "source_image_sha256": run.get("source_image_sha256"),
        "segmentation_device": run.get("segmentation", {}).get("device"),
        "dependency_versions": run.get("dependency_versions", {}),
    }


def build_snapshot(batch_dir: Path) -> dict[str, Any]:
    summary = pd.read_csv(batch_dir / "batch_summary.csv", dtype=str, keep_default_na=False)
    done = summary.loc[summary["status"].eq("success"), "image_name"].tolist()
    if not done:
        raise ValueError(f"No successful sections in {batch_dir}")
    return {
        "schema_version": SNAPSHOT_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "sections": [section_snapshot(batch_dir / image_id, image_id) for image_id in sorted(done)],
    }


@dataclass(frozen=True)
class SectionCheck:
    image_id: str
    level: str  # exact | tolerant | missing
    passed: bool
    detail: str


def _proportions(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    return {name: value / total for name, value in counts.items()} if total else {}


def compare_snapshots(
    reference: dict[str, Any],
    current: dict[str, Any],
    *,
    count_tolerance: float = DEFAULT_COUNT_TOLERANCE,
    composition_tolerance: float = DEFAULT_COMPOSITION_TOLERANCE,
) -> list[SectionCheck]:
    if reference.get("schema_version") != SNAPSHOT_SCHEMA:
        raise ValueError(f"Reference snapshot must use schema_version {SNAPSHOT_SCHEMA}")
    now = {section["image_id"]: section for section in current["sections"]}
    checks: list[SectionCheck] = []
    for expected in reference["sections"]:
        image_id = expected["image_id"]
        actual = now.get(image_id)
        if actual is None:
            checks.append(SectionCheck(image_id, "missing", False, "section not in this run"))
            continue
        if actual["model_sha256"] != expected["model_sha256"]:
            checks.append(SectionCheck(image_id, "exact", False, "a different model was used"))
            continue
        if actual["mask_sha256"] == expected["mask_sha256"]:
            same = actual["calls_sha256"] == expected["calls_sha256"]
            detail = (
                "identical mask and identical fiber calls"
                if same
                else "identical mask but fiber calls differ"
            )
            checks.append(SectionCheck(image_id, "exact", same, detail))
            continue
        count_difference = abs(actual["n_fibers"] - expected["n_fibers"]) / max(
            expected["n_fibers"], 1
        )
        reference_share = _proportions(expected["class_counts"])
        current_share = _proportions(actual["class_counts"])
        composition_difference = max(
            abs(reference_share.get(name, 0.0) - current_share.get(name, 0.0))
            for name in set(reference_share) | set(current_share)
        )
        passed = (
            count_difference <= count_tolerance and composition_difference <= composition_tolerance
        )
        checks.append(
            SectionCheck(
                image_id,
                "tolerant",
                passed,
                f"mask differs; fiber count {100 * count_difference:.2f}% different "
                f"(limit {100 * count_tolerance:.0f}%), largest class-share difference "
                f"{100 * composition_difference:.2f} points "
                f"(limit {100 * composition_tolerance:.0f})",
            )
        )
    return checks


def freeze(batch_dir: Path, snapshot_path: Path) -> dict[str, Any]:
    if snapshot_path.exists():
        raise ValueError(f"{snapshot_path} already exists; a frozen reference is not overwritten")
    snapshot = build_snapshot(batch_dir)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    return snapshot


def check(
    batch_dir: Path,
    snapshot_path: Path,
    *,
    count_tolerance: float = DEFAULT_COUNT_TOLERANCE,
    composition_tolerance: float = DEFAULT_COMPOSITION_TOLERANCE,
) -> list[SectionCheck]:
    reference = json.loads(snapshot_path.read_text(encoding="utf-8"))
    return compare_snapshots(
        reference,
        build_snapshot(batch_dir),
        count_tolerance=count_tolerance,
        composition_tolerance=composition_tolerance,
    )
