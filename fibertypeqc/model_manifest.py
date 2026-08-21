"""Versioned model-sidecar validation for future multi-panel classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from fibertypeqc.feature_schema import LEGACY_FROZEN_ALPHA_FEATURE_SCHEMA, MULTIPANEL_FEATURE_SCHEMA
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
    artifact: str | None = None
    artifact_sha256: str | None = None
    intended_use: str | None = None


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
    artifact = raw.get("artifact")
    if artifact is not None and not isinstance(artifact, str):
        raise ValueError(f"Model manifest {path} artifact must be a path string.")
    artifact_sha256 = raw.get("artifact_sha256")
    if artifact_sha256 is not None:
        if not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64:
            raise ValueError(f"Model manifest {path} artifact_sha256 must be a SHA-256 digest.")
        try:
            int(artifact_sha256, 16)
        except ValueError as exc:
            raise ValueError(
                f"Model manifest {path} artifact_sha256 must be a SHA-256 digest."
            ) from exc
    intended_use = raw.get("intended_use")
    if intended_use is not None and not isinstance(intended_use, str):
        raise ValueError(f"Model manifest {path} intended_use must be a string.")
    return ModelManifest(
        model_id=str(raw["model_id"]),
        task=str(raw["task"]),
        feature_schema_version=str(raw["feature_schema_version"]),
        required_markers=frozenset(markers),
        outputs=tuple(outputs),
        source_path=path,
        artifact=artifact,
        artifact_sha256=artifact_sha256,
        intended_use=intended_use,
    )


def validate_model_artifact(path: Path, manifest: ModelManifest) -> None:
    """Verify a selected model artifact against an optional manifest digest."""
    if not path.is_file():
        raise ValueError(f"Model artifact not found: {path}")
    if manifest.artifact is not None:
        declared = Path(manifest.artifact)
        declared_name = declared.name
        if path.name != declared_name:
            raise ValueError(
                f"Model manifest {manifest.source_path} declares artifact '{declared_name}', "
                f"but --classifier-path selected '{path.name}'."
            )
    if manifest.artifact_sha256 is None:
        return
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != manifest.artifact_sha256.lower():
        raise ValueError(
            f"Model artifact digest mismatch for {path}: expected "
            f"{manifest.artifact_sha256.lower()}, got {digest}."
        )


def validate_model_compatibility(
    panel: Panel,
    manifest: ModelManifest | None,
    *,
    available_feature_schema: str = LEGACY_FROZEN_ALPHA_FEATURE_SCHEMA,
    require_legacy_model: bool = False,
) -> None:
    """Fail before inference when a selected model lacks required observations."""
    if manifest is None:
        if not require_legacy_model:
            return
        required = LEGACY_FROZEN_ALPHA_REQUIRED_MARKERS
    else:
        required = manifest.required_markers
    missing = sorted(name for name in required if panel.channels.get(name) is None)
    if missing:
        model_name = manifest.model_id if manifest else "legacy_frozen_alpha"
        raise ValueError(f"Model '{model_name}' requires observed channels: {', '.join(missing)}.")
    supported = {available_feature_schema, MULTIPANEL_FEATURE_SCHEMA}
    if manifest is not None and manifest.feature_schema_version not in supported:
        raise ValueError(
            f"Model '{manifest.model_id}' requires feature schema "
            f"'{manifest.feature_schema_version}', but this pipeline provides "
            f"'{available_feature_schema}'."
        )
