"""Versioned model-sidecar validation for future multi-panel classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fibertypeqc.panels import OBSERVED_CHANNELS, Panel

MODEL_MANIFEST_VERSION = 1
LEGACY_FROZEN_ALPHA_REQUIRED_MARKERS = frozenset(("laminin", "type_iia", "type_iib"))


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    task: str
    feature_schema_version: str
    required_markers: frozenset[str]
    outputs: tuple[str, ...]
    source_path: Path


def load_model_manifest(path: Path) -> ModelManifest:
    try:
        raw: Any = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"Model manifest file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in model manifest {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Model manifest {path} must be a mapping.")
    if raw.get("manifest_version") != MODEL_MANIFEST_VERSION:
        raise ValueError(
            f"Model manifest {path} must set manifest_version: {MODEL_MANIFEST_VERSION}."
        )
    required = ("model_id", "task", "feature_schema_version", "required_markers", "outputs")
    missing = [name for name in required if not raw.get(name)]
    if missing:
        raise ValueError(f"Model manifest {path} is missing required fields: {', '.join(missing)}.")
    markers = raw["required_markers"]
    if not isinstance(markers, list) or not all(isinstance(marker, str) for marker in markers):
        raise ValueError(f"Model manifest {path} required_markers must be a list of names.")
    unknown = set(markers) - set(OBSERVED_CHANNELS)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Model manifest {path} has unsupported markers: {names}.")
    outputs = raw["outputs"]
    if not isinstance(outputs, list) or not all(isinstance(value, str) for value in outputs):
        raise ValueError(f"Model manifest {path} outputs must be a list of names.")
    return ModelManifest(
        model_id=str(raw["model_id"]),
        task=str(raw["task"]),
        feature_schema_version=str(raw["feature_schema_version"]),
        required_markers=frozenset(markers),
        outputs=tuple(outputs),
        source_path=path,
    )


def validate_model_compatibility(panel: Panel, manifest: ModelManifest | None) -> None:
    """Fail before inference when a selected model lacks required observations."""
    required = manifest.required_markers if manifest else LEGACY_FROZEN_ALPHA_REQUIRED_MARKERS
    missing = sorted(name for name in required if panel.channels.get(name) is None)
    if missing:
        model_name = manifest.model_id if manifest else "legacy_frozen_alpha"
        raise ValueError(f"Model '{model_name}' requires observed channels: {', '.join(missing)}.")
