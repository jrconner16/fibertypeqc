"""Validated YAML project manifests and review output path separation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.review.schemas import PROJECT_SCHEMA_VERSION, Domain, parse_enum


def _required_text(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _resolve(base: Path, value: str, context: str) -> Path:
    if not value.strip():
        raise ValueError(f"{context} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class ProjectImage:
    image_id: str
    mouse_id: str
    section_id: str
    raw_image_path: Path
    prediction_directory: Path
    condition: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class Project:
    schema_version: str
    project_id: str
    project_name: str
    panel_manifest: Path
    model_version: str
    manifest_path: Path
    images: tuple[ProjectImage, ...]
    qc_version: str = ""

    @property
    def root(self) -> Path:
        return self.manifest_path.parent

    @property
    def review_directory(self) -> Path:
        return self.root / "review"

    @property
    def review_state_path(self) -> Path:
        return self.review_directory / "review_state.json"

    @property
    def review_events_path(self) -> Path:
        return self.review_directory / "review_events.csv"

    def image(self, image_id: str) -> ProjectImage:
        for image in self.images:
            if image.image_id == image_id:
                return image
        raise KeyError(f"Unknown project image_id: {image_id}")

    def reviewed_mask_path(self, image_id: str, domain: Domain | str) -> Path:
        image = self.image(image_id)
        parsed_domain = parse_enum(Domain, domain, "reviewed mask domain")
        if parsed_domain is Domain.FIBER_TYPING:
            raise ValueError("fiber_typing has no reviewed mask; store reviewed types in CSV")
        directory_name = (
            "reviewed_fiber_labels"
            if parsed_domain is Domain.FIBER_SEGMENTATION
            else "reviewed_nuclei_labels"
        )
        output_key = (
            "fiber_labels" if parsed_domain is Domain.FIBER_SEGMENTATION else "nuclei_labels"
        )
        predicted = image.outputs.get(output_key)
        filename = predicted.name if predicted else f"{image_id}_{output_key}.tif"
        reviewed = (self.review_directory / directory_name / filename).resolve()
        for candidate in self.images:
            if _is_within(reviewed, candidate.prediction_directory):
                raise ValueError(
                    f"Reviewed mask path must be outside prediction directories: {reviewed}"
                )
        return reviewed


def _parse_image(
    raw: Any,
    *,
    index: int,
    manifest_directory: Path,
    validate_paths: bool,
) -> ProjectImage:
    context = f"images[{index}]"
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a mapping")
    image_id = _required_text(raw, "image_id", context)
    mouse_id = _required_text(raw, "mouse_id", context)
    section_id = _required_text(raw, "section_id", context)
    raw_image_path = _resolve(
        manifest_directory,
        _required_text(raw, "raw_image_path", context),
        f"{context}.raw_image_path",
    )
    prediction_directory = _resolve(
        manifest_directory,
        _required_text(raw, "prediction_directory", context),
        f"{context}.prediction_directory",
    )
    condition = raw.get("condition", {})
    if not isinstance(condition, dict):
        raise ValueError(f"{context}.condition must be a mapping")
    raw_outputs = raw.get("outputs", {})
    if not isinstance(raw_outputs, dict):
        raise ValueError(f"{context}.outputs must be a mapping")
    outputs: dict[str, Path] = {}
    for key, value in raw_outputs.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"{context}.outputs must map string names to string paths")
        outputs[key] = _resolve(
            prediction_directory, value, f"{context}.outputs.{key}"
        )

    if validate_paths:
        if not raw_image_path.is_file():
            raise FileNotFoundError(f"{context}.raw_image_path does not exist: {raw_image_path}")
        if not prediction_directory.is_dir():
            raise FileNotFoundError(
                f"{context}.prediction_directory does not exist: {prediction_directory}"
            )
        for key, path in outputs.items():
            if not path.is_file():
                raise FileNotFoundError(f"{context}.outputs.{key} does not exist: {path}")

    return ProjectImage(
        image_id=image_id,
        mouse_id=mouse_id,
        section_id=section_id,
        raw_image_path=raw_image_path,
        prediction_directory=prediction_directory,
        condition=dict(condition),
        outputs=outputs,
    )


def load_project(path: Path | str, *, validate_paths: bool = True) -> Project:
    """Load and validate a project manifest.

    Relative paths resolve against the manifest directory. Identity metadata is
    always explicit; this loader never infers mouse or section IDs from filenames.
    """
    manifest_path = Path(path).expanduser().resolve()
    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Project manifest does not exist: {manifest_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Project manifest is not valid YAML: {manifest_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Project manifest must contain a mapping: {manifest_path}")

    schema_version = _required_text(data, "schema_version", "project")
    if schema_version != PROJECT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported project schema_version {schema_version!r}; "
            f"expected {PROJECT_SCHEMA_VERSION!r}"
        )
    project_id = _required_text(data, "project_id", "project")
    project_name = _required_text(data, "project_name", "project")
    model_version = _required_text(data, "model_version", "project")
    panel_manifest = _resolve(
        manifest_path.parent,
        _required_text(data, "panel_manifest", "project"),
        "project.panel_manifest",
    )
    raw_images = data.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        raise ValueError("project.images must be a non-empty list")
    images = tuple(
        _parse_image(
            item,
            index=index,
            manifest_directory=manifest_path.parent,
            validate_paths=validate_paths,
        )
        for index, item in enumerate(raw_images)
    )
    image_ids = [image.image_id for image in images]
    if len(image_ids) != len(set(image_ids)):
        duplicates = sorted({image_id for image_id in image_ids if image_ids.count(image_id) > 1})
        raise ValueError(f"project image_id values must be unique; duplicates: {duplicates}")
    if validate_paths and not panel_manifest.is_file():
        raise FileNotFoundError(f"project.panel_manifest does not exist: {panel_manifest}")

    review_directory = (manifest_path.parent / "review").resolve()
    for image in images:
        if _is_within(review_directory, image.prediction_directory):
            raise ValueError(
                "Project review directory must be outside every prediction directory; "
                f"review={review_directory}, prediction={image.prediction_directory}"
            )
        for key, output_path in image.outputs.items():
            if _is_within(output_path, review_directory):
                raise ValueError(
                    f"Predicted output {key!r} must be outside the review directory: "
                    f"{output_path}"
                )

    qc_version = data.get("qc_version", "")
    if not isinstance(qc_version, str):
        raise ValueError("project.qc_version must be a string when provided")
    return Project(
        schema_version=schema_version,
        project_id=project_id,
        project_name=project_name,
        panel_manifest=panel_manifest,
        model_version=model_version,
        manifest_path=manifest_path,
        images=images,
        qc_version=qc_version,
    )
