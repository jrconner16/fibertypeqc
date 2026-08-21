"""Versioned index of retained per-image result artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

RESULT_BUNDLE_SCHEMA_VERSION = "fibertypeqc.result_bundle.v1"
RETAIN_MODES = frozenset(("full", "tables", "summary"))

ARTIFACT_CONTRACTS: dict[str, dict[str, Any]] = {
    "fiber_labels": {
        "kind": "label_image",
        "media_type": "image/tiff",
        "cardinality": "one_label_image_per_image",
        "join_keys": ["label"],
        "domains": ["fiber_geometry"],
    },
    "fiber_table": {
        "kind": "table",
        "media_type": "text/csv",
        "cardinality": "one_row_per_fiber",
        "join_keys": ["label"],
        "domains": ["fiber_geometry", "fiber_identity"],
    },
    "feature_diagnostics": {
        "kind": "table",
        "media_type": "text/csv",
        "cardinality": "one_row_per_fiber",
        "join_keys": ["label"],
        "domains": ["fiber_identity"],
    },
    "fiber_identity_predictions": {
        "kind": "table",
        "media_type": "text/csv",
        "cardinality": "one_row_per_fiber",
        "join_keys": ["label"],
        "domains": ["fiber_identity"],
    },
    "image_summary": {
        "kind": "table",
        "media_type": "text/csv",
        "cardinality": "one_row_per_image",
        "join_keys": [],
        "domains": ["summary"],
    },
    "preflight_qc": {
        "kind": "qc_report",
        "media_type": "application/json",
        "cardinality": "one_report_per_image",
        "join_keys": [],
        "domains": ["quality_control"],
    },
    "postrun_qc": {
        "kind": "qc_report",
        "media_type": "application/json",
        "cardinality": "one_report_per_image",
        "join_keys": [],
        "domains": ["quality_control"],
    },
    "run_provenance": {
        "kind": "provenance",
        "media_type": "application/json",
        "cardinality": "one_manifest_per_image",
        "join_keys": [],
        "domains": ["provenance"],
    },
    "nuclei_labels": {
        "kind": "label_image",
        "media_type": "image/tiff",
        "cardinality": "one_label_image_per_image",
        "join_keys": ["nucleus_id"],
        "domains": ["nuclei"],
    },
    "nuclei_table": {
        "kind": "table",
        "media_type": "text/csv",
        "cardinality": "one_row_per_nucleus",
        "join_keys": ["nucleus_id", "assigned_fiber_id"],
        "domains": ["nuclei", "association"],
    },
    "nucleus_fiber_associations": {
        "kind": "table",
        "media_type": "text/csv",
        "cardinality": "one_row_per_assigned_nucleus",
        "join_keys": ["nucleus_id", "fiber_id"],
        "domains": ["association"],
    },
    "fiber_nuclei_summary": {
        "kind": "table",
        "media_type": "text/csv",
        "cardinality": "one_row_per_fiber",
        "join_keys": ["fiber_id"],
        "domains": ["nuclei", "association"],
    },
    "nuclear_provenance": {
        "kind": "provenance",
        "media_type": "application/json",
        "cardinality": "one_manifest_per_image",
        "join_keys": [],
        "domains": ["provenance", "nuclei"],
    },
}


def build_result_bundle(
    *,
    output_dir: Path,
    image_id: str,
    retain_mode: str,
    artifact_paths: Mapping[str, Path | None],
    additional_domains: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Build a portable index containing only artifacts retained on disk."""
    if not image_id.strip():
        raise ValueError("Result bundle image_id must not be empty.")
    if retain_mode not in RETAIN_MODES:
        raise ValueError(f"Unsupported retain mode: {retain_mode}")
    root = output_dir.resolve()
    extra_domains = additional_domains or {}
    unknown = sorted(set(artifact_paths) - set(ARTIFACT_CONTRACTS))
    if unknown:
        raise ValueError(f"Unsupported result-bundle artifacts: {', '.join(unknown)}")

    artifacts: dict[str, dict[str, Any]] = {}
    for name, raw_path in artifact_paths.items():
        if raw_path is None or not raw_path.is_file():
            continue
        resolved = raw_path.resolve()
        try:
            relative_path = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Result artifact must be inside output_dir: {raw_path}") from exc
        contract = ARTIFACT_CONTRACTS[name]
        domains = list(dict.fromkeys([*contract["domains"], *extra_domains.get(name, ())]))
        artifacts[name] = {
            "path": relative_path,
            "kind": contract["kind"],
            "media_type": contract["media_type"],
            "cardinality": contract["cardinality"],
            "join_keys": list(contract["join_keys"]),
            "domains": domains,
        }

    return {
        "schema_version": RESULT_BUNDLE_SCHEMA_VERSION,
        "image_id": image_id,
        "retain_mode": retain_mode,
        "artifacts": artifacts,
    }


def write_result_bundle(path: Path, bundle: Mapping[str, Any]) -> None:
    """Write a result bundle using stable, portable JSON formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(bundle), indent=2, sort_keys=True) + "\n")
