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

RUN_MANIFEST_SCHEMA_VERSION = 2
REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDED_DISTRIBUTIONS = (
    "cellpose",
    "torch",
    "numpy",
    "scipy",
    "scikit-image",
    "scikit-learn",
    "pandas",
    "tifffile",
    "czifile",
)
LEGACY_OUTPUT_SCHEMA_VERSION = "legacy_fibers.v1"


def decide_artifact_reuse(
    *,
    panel_changed: bool = False,
    fiber_segmentation_changed: bool = False,
    nuclear_segmentation_changed: bool = False,
    marker_features_changed: bool = False,
    classifier_changed: bool = False,
    nucleus_association_changed: bool = False,
) -> dict[str, bool]:
    """Return which cached stages can be reused under the documented matrix."""
    reuse_fiber_labels = not (panel_changed or fiber_segmentation_changed)
    reuse_nuclei_labels = not (panel_changed or nuclear_segmentation_changed)
    recompute_features_or_links = any(
        (
            panel_changed,
            fiber_segmentation_changed,
            nuclear_segmentation_changed,
            marker_features_changed,
            classifier_changed,
            nucleus_association_changed,
        )
    )
    return {
        "reuse_fiber_labels": reuse_fiber_labels,
        "reuse_nuclei_labels": reuse_nuclei_labels,
        "recompute_features_or_links": recompute_features_or_links,
    }


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Stream a file's SHA-256 so large microscopy inputs are not read into memory twice."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path | str | None, base: Path | None = None) -> str:
    """Return a path safe to record in shareable outputs.

    Relative paths are kept. Absolute paths become relative to ``base`` (or the working
    directory) when they are inside it; otherwise only the file name is kept, so user-specific
    directories never reach result files. Inputs are identified by digest, not location.
    """
    if path is None or str(path) == "":
        return ""
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    root = (base or Path.cwd()).resolve()
    try:
        return candidate.resolve().relative_to(root).as_posix()
    except ValueError:
        return candidate.name


def fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, cwd=REPO_ROOT
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
    input_sha256: str,
    image_shape: tuple[int, ...],
    pixel_size_um: tuple[float | None, float | None],
    panel_fingerprint: str,
    panel_channels: Mapping[str, int | None],
    segmentation: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    classifier_path: str | None,
    model_manifest_path: Path | None,
    classifier_sha256: str | None = None,
) -> dict[str, Any]:
    dependency_versions = {name: package_version(name) for name in RECORDED_DISTRIBUTIONS}
    # Segmentation is reusable only for the same image content, labels source, device, and
    # Cellpose version as well as the same parameters and panel.
    segmentation_fingerprint_input = {
        **dict(segmentation),
        **dict(preprocessing),
        "panel": panel_fingerprint,
        "input_sha256": input_sha256,
        "cellpose_version": dependency_versions["cellpose"],
    }
    classification_fingerprint_input = {
        "classifier_sha256": classifier_sha256,
        "model_manifest_path": portable_path(model_manifest_path),
        "panel": panel_fingerprint,
    }
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "application_version": __version__,
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "dependency_versions": dependency_versions,
        "output_schema_version": LEGACY_OUTPUT_SCHEMA_VERSION,
        "source_image": portable_path(input_path),
        "source_image_sha256": input_sha256,
        "image_shape": list(image_shape),
        "image_channel_count": image_shape[0],
        "pixel_size_um": {"x": pixel_size_um[0], "y": pixel_size_um[1]},
        "panel": {"channels": dict(panel_channels), "fingerprint": panel_fingerprint},
        "segmentation": dict(segmentation),
        "preprocessing": dict(preprocessing),
        "classifier_path": portable_path(classifier_path) or None,
        "classifier_sha256": classifier_sha256,
        "model_manifest_path": portable_path(model_manifest_path) or None,
        "stage_fingerprints": {
            "fiber_segmentation": fingerprint(segmentation_fingerprint_input),
            "classification": fingerprint(classification_fingerprint_input),
        },
    }


def write_run_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def can_reuse_fiber_labels(previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """Whether the current fiber segmentation fingerprint matches a prior run."""
    return previous.get("schema_version") == RUN_MANIFEST_SCHEMA_VERSION and previous.get(
        "stage_fingerprints", {}
    ).get("fiber_segmentation") == current.get("stage_fingerprints", {}).get("fiber_segmentation")


def load_run_manifest(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None
