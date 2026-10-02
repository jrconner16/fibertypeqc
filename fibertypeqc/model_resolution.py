"""Resolve a registered model ID to its manifest and artifact.

Tracked artifacts live in the repository. Artifacts trained on private or unpublished data are not
committed; they are looked up by file name in the directory named by ``FIBERTYPEQC_MODEL_ROOT``
and verified against the registry digest before use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fibertypeqc.artifacts import file_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_REGISTRY_PATH = Path("manifests/model_registry.v1.yaml")
MODEL_ROOT_ENV = "FIBERTYPEQC_MODEL_ROOT"


@dataclass(frozen=True)
class ResolvedModel:
    model_id: str
    artifact_path: Path
    manifest_path: Path


def _registry_entries(repo_root: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load((repo_root / MODEL_REGISTRY_PATH).read_text(encoding="utf-8"))
    return list(raw.get("models", []))


def registered_model_ids(repo_root: Path = REPO_ROOT) -> list[str]:
    return [str(entry["model_id"]) for entry in _registry_entries(repo_root)]


def resolve_model(
    model_id: str,
    *,
    repo_root: Path = REPO_ROOT,
    model_root: Path | None = None,
) -> ResolvedModel:
    """Return the artifact and manifest paths for a registered model ID."""
    entries = {str(entry["model_id"]): entry for entry in _registry_entries(repo_root)}
    entry = entries.get(model_id)
    if entry is None:
        known = ", ".join(sorted(entries))
        raise ValueError(
            f"Unknown model '{model_id}'. --model takes a registered model ID or a path to a "
            f"registered model file. Registered models: {known}."
        )
    artifact = entry.get("artifact")
    manifest = entry.get("manifest")
    if not artifact or not manifest:
        raise ValueError(f"Model '{model_id}' has no runnable artifact and manifest registered.")
    manifest_path = repo_root / str(manifest)
    if entry.get("artifact_location", "tracked") == "private":
        root = model_root
        if root is None:
            configured = os.environ.get(MODEL_ROOT_ENV)
            if not configured:
                raise ValueError(
                    f"Model '{model_id}' is distributed privately. Set {MODEL_ROOT_ENV} to the "
                    f"directory containing {artifact}."
                )
            root = Path(configured).expanduser()
        artifact_path = root / str(artifact)
        if not artifact_path.is_file():
            raise ValueError(
                f"Model '{model_id}' artifact {artifact} was not found in {MODEL_ROOT_ENV}."
            )
    else:
        artifact_path = repo_root / str(artifact)
    return ResolvedModel(
        model_id=model_id, artifact_path=artifact_path, manifest_path=manifest_path
    )


def default_model_id(task: str = "fiber_identity", repo_root: Path = REPO_ROOT) -> str | None:
    """Return the registry's default model for a task, if one is declared."""
    for entry in _registry_entries(repo_root):
        if entry.get("default") and entry.get("task") == task:
            return str(entry["model_id"])
    return None


def _looks_like_path(value: str) -> bool:
    return (
        os.sep in value
        or value.endswith((".joblib", ".pkl", ".pickle"))
        or Path(value).expanduser().exists()
    )


def resolve_model_argument(
    value: str,
    *,
    repo_root: Path = REPO_ROOT,
    model_root: Path | None = None,
) -> ResolvedModel:
    """Resolve ``--model``: a registered model ID or a path to a registered model file.

    A file is identified by its SHA-256 against the registry, so it is verified and uses the
    registered manifest (including pinned feature extraction) regardless of its file name.
    """
    if not _looks_like_path(value):
        return resolve_model(value, repo_root=repo_root, model_root=model_root)
    path = Path(value).expanduser()
    if path.is_dir():
        raise ValueError(
            f"--model got a directory ({value}). Pass a registered model ID or the model file "
            f"itself, or put the directory in {MODEL_ROOT_ENV}."
        )
    if not path.is_file():
        raise ValueError(f"--model file not found: {value}")
    digest = file_sha256(path)
    for entry in _registry_entries(repo_root):
        if str(entry.get("artifact_sha256", "")).lower() == digest and entry.get("manifest"):
            return ResolvedModel(
                model_id=str(entry["model_id"]),
                artifact_path=path.resolve(),
                manifest_path=repo_root / str(entry["manifest"]),
            )
    raise ValueError(
        f"--model file {path.name} (sha256 {digest[:12]}…) is not a registered model. "
        f"Registered models: {', '.join(sorted(registered_model_ids(repo_root)))}."
    )
