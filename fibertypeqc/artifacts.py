"""Small, versioned provenance records for pipeline runs."""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from fibertypeqc import __version__

RUN_MANIFEST_SCHEMA_VERSION = 1
LEGACY_OUTPUT_SCHEMA_VERSION = "legacy_fibers.v1"


def fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def build_run_manifest(
    *,
    input_path: Path,
    image_shape: tuple[int, ...],
    pixel_size_um: tuple[float | None, float | None],
    panel_fingerprint: str,
    panel_channels: Mapping[str, int | None],
    segmentation: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    classifier_path: str | None,
    model_manifest_path: Path | None,
) -> dict[str, Any]:
    segmentation_fingerprint_input = {
        **dict(segmentation),
        **dict(preprocessing),
        "panel": panel_fingerprint,
    }
    classification_fingerprint_input = {
        "classifier_path": classifier_path,
        "model_manifest_path": str(model_manifest_path) if model_manifest_path else None,
        "panel": panel_fingerprint,
    }
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "application_version": __version__,
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "dependency_versions": {
            "cellpose": package_version("cellpose"),
            "torch": package_version("torch"),
        },
        "output_schema_version": LEGACY_OUTPUT_SCHEMA_VERSION,
        "source_image": str(input_path),
        "image_shape": list(image_shape),
        "image_channel_count": image_shape[0],
        "pixel_size_um": {"x": pixel_size_um[0], "y": pixel_size_um[1]},
        "panel": {"channels": dict(panel_channels), "fingerprint": panel_fingerprint},
        "segmentation": dict(segmentation),
        "preprocessing": dict(preprocessing),
        "classifier_path": classifier_path,
        "model_manifest_path": str(model_manifest_path) if model_manifest_path else None,
        "stage_fingerprints": {
            "fiber_segmentation": fingerprint(segmentation_fingerprint_input),
            "classification": fingerprint(classification_fingerprint_input),
        },
    }


def write_run_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
