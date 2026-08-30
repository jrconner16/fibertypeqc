"""Validation for versioned model and dataset-evidence ledgers.

These ledgers deliberately describe evidence without loading private images or labels.  They make
the identity, scope, and status of a model explicit before a comparison script is allowed to give
it a broader name such as "candidate".
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

MODEL_REGISTRY_SCHEMA_VERSION = 1
DATASET_SPLIT_LEDGER_SCHEMA_VERSION = 1
DATASET_EVIDENCE_INVENTORY_SCHEMA_VERSION = 1

MODEL_STATUSES = frozenset({"released", "standing_candidate", "exploratory", "retired"})
MODEL_TASKS = frozenset({"fiber_identity", "emhc_status", "review_risk", "segmentation"})
LABEL_AUTHORITY_TIERS = frozenset(
    {"manual_gold", "reviewed_myosight", "myosight_derived", "model_prediction", "mixed"}
)
GROUP_ROLES = frozenset({"development", "evaluation", "reserve", "excluded"})


def _load_yaml_mapping(path: Path, *, description: str) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{description} file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {description} {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{description} {path} must be a mapping.")
    return raw


def _require_string(entry: dict[str, Any], field: str, *, description: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} requires a non-empty '{field}'.")
    return value


def _require_sha256(value: Any, *, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{description} must be a SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{description} must be a SHA-256 digest.") from exc
    return value.lower()


def validate_model_registry(path: Path, *, repo_root: Path | None = None) -> None:
    """Validate model identifiers, declared scope, and tracked artifact digests."""
    raw = _load_yaml_mapping(path, description="model registry")
    if raw.get("schema_version") != MODEL_REGISTRY_SCHEMA_VERSION:
        raise ValueError(f"Model registry {path} must set schema_version: 1.")
    entries = raw.get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Model registry {path} must contain a non-empty models list.")

    seen_ids: set[str] = set()
    root = repo_root or path.parents[1]
    for index, entry in enumerate(entries, start=1):
        description = f"Model registry entry {index}"
        if not isinstance(entry, dict):
            raise ValueError(f"{description} must be a mapping.")
        model_id = _require_string(entry, "model_id", description=description)
        if model_id in seen_ids:
            raise ValueError(f"Model registry has duplicate model_id: {model_id}.")
        seen_ids.add(model_id)
        task = _require_string(entry, "task", description=description)
        if task not in MODEL_TASKS:
            raise ValueError(f"{description} has unsupported task: {task}.")
        status = _require_string(entry, "status", description=description)
        if status not in MODEL_STATUSES:
            raise ValueError(f"{description} has unsupported status: {status}.")
        required_markers = entry.get("required_markers")
        if not isinstance(required_markers, list) or not all(
            isinstance(marker, str) and marker for marker in required_markers
        ):
            raise ValueError(f"{description} required_markers must be a list of marker names.")
        for field in ("feature_schema", "development_groups", "evaluation_groups", "limitations"):
            _require_string(entry, field, description=description)

        artifact = entry.get("artifact")
        digest = entry.get("artifact_sha256")
        if artifact is None:
            if digest is not None:
                raise ValueError(f"{description} cannot declare a digest without an artifact.")
            continue
        if not isinstance(artifact, str) or not artifact:
            raise ValueError(f"{description} artifact must be a repository-relative path or null.")
        _require_sha256(
            digest,
            description=f"{description} artifact_sha256",
        )
        artifact_path = root / artifact
        if not artifact_path.is_file():
            raise ValueError(f"{description} artifact is not tracked locally: {artifact}.")
        actual_digest = sha256(artifact_path.read_bytes()).hexdigest()
        if actual_digest != digest.lower():
            raise ValueError(
                f"{description} artifact digest mismatch for {artifact}: "
                f"expected {digest.lower()}, got {actual_digest}."
            )


def validate_dataset_split_ledger(path: Path) -> None:
    """Validate the append-only cohort/group split contract without reading private data."""
    raw = _load_yaml_mapping(path, description="dataset/split ledger")
    if raw.get("schema_version") != DATASET_SPLIT_LEDGER_SCHEMA_VERSION:
        raise ValueError(f"Dataset/split ledger {path} must set schema_version: 1.")
    if raw.get("amendment_policy") != "append_only_new_version":
        raise ValueError("Dataset/split ledger must use amendment_policy: append_only_new_version.")
    cohorts = raw.get("cohorts")
    assignments = raw.get("group_assignments")
    if not isinstance(cohorts, list) or not cohorts:
        raise ValueError("Dataset/split ledger must contain a non-empty cohorts list.")
    if not isinstance(assignments, list):
        raise ValueError("Dataset/split ledger group_assignments must be a list.")

    cohort_ids: set[str] = set()
    for index, cohort in enumerate(cohorts, start=1):
        description = f"Dataset ledger cohort {index}"
        if not isinstance(cohort, dict):
            raise ValueError(f"{description} must be a mapping.")
        cohort_id = _require_string(cohort, "cohort_id", description=description)
        if cohort_id in cohort_ids:
            raise ValueError(f"Dataset/split ledger has duplicate cohort_id: {cohort_id}.")
        cohort_ids.add(cohort_id)
        _require_string(cohort, "data_availability", description=description)
        authority = _require_string(cohort, "label_authority", description=description)
        if authority not in LABEL_AUTHORITY_TIERS:
            raise ValueError(f"{description} has unsupported label_authority: {authority}.")

    assigned_groups: dict[tuple[str, str], str] = {}
    for index, assignment in enumerate(assignments, start=1):
        description = f"Dataset ledger group assignment {index}"
        if not isinstance(assignment, dict):
            raise ValueError(f"{description} must be a mapping.")
        cohort_id = _require_string(assignment, "cohort_id", description=description)
        if cohort_id not in cohort_ids:
            raise ValueError(f"{description} references unknown cohort_id: {cohort_id}.")
        group_id = _require_string(assignment, "group_id", description=description)
        role = _require_string(assignment, "role", description=description)
        if role not in GROUP_ROLES:
            raise ValueError(f"{description} has unsupported role: {role}.")
        key = (cohort_id, group_id)
        if key in assigned_groups:
            raise ValueError(
                f"Dataset/split ledger assigns {cohort_id}/{group_id} more than once "
                f"({assigned_groups[key]} and {role})."
            )
        assigned_groups[key] = role

    contracts = raw.get("split_contracts", [])
    if not isinstance(contracts, list):
        raise ValueError("Dataset/split ledger split_contracts must be a list.")
    seen_contract_ids: set[str] = set()
    for index, contract in enumerate(contracts, start=1):
        description = f"Dataset ledger split contract {index}"
        if not isinstance(contract, dict):
            raise ValueError(f"{description} must be a mapping.")
        contract_id = _require_string(contract, "contract_id", description=description)
        if contract_id in seen_contract_ids:
            raise ValueError(f"Dataset/split ledger has duplicate contract_id: {contract_id}.")
        seen_contract_ids.add(contract_id)
        cohort_id = _require_string(contract, "cohort_id", description=description)
        if cohort_id not in cohort_ids:
            raise ValueError(f"{description} references unknown cohort_id: {cohort_id}.")
        _require_string(contract, "scope", description=description)
        _require_string(contract, "unit_of_independence", description=description)
        group_prefix = _require_string(
            contract,
            "assignment_group_prefix",
            description=description,
        )

        role_counts = contract.get("role_counts")
        if not isinstance(role_counts, dict) or not role_counts:
            raise ValueError(f"{description} role_counts must be a non-empty mapping.")
        expected_counts: dict[str, int] = {}
        for role, count in role_counts.items():
            if role not in GROUP_ROLES:
                raise ValueError(f"{description} has unsupported role count: {role}.")
            if type(count) is not int or count < 0:
                raise ValueError(f"{description} role count for {role} must be a non-negative int.")
            if count:
                expected_counts[role] = count

        artifact_digests = contract.get("private_artifact_sha256")
        if not isinstance(artifact_digests, dict) or not artifact_digests:
            raise ValueError(f"{description} private_artifact_sha256 must be a non-empty mapping.")
        for artifact_name, digest in artifact_digests.items():
            if not isinstance(artifact_name, str) or not artifact_name:
                raise ValueError(f"{description} has an invalid private artifact name.")
            _require_sha256(
                digest,
                description=f"{description} digest for {artifact_name}",
            )

        actual_counts: dict[str, int] = {}
        for (assigned_cohort, group_id), role in assigned_groups.items():
            if assigned_cohort == cohort_id and group_id.startswith(group_prefix):
                actual_counts[role] = actual_counts.get(role, 0) + 1
        if actual_counts != expected_counts:
            raise ValueError(
                f"{description} role_counts do not match assignments with prefix {group_prefix!r}: "
                f"expected {expected_counts}, got {actual_counts}."
            )


def validate_dataset_evidence_inventory(path: Path) -> None:
    """Validate a mouse-grouped dataset inventory without loading private artifacts."""
    raw = _load_yaml_mapping(path, description="dataset evidence inventory")
    if raw.get("schema_version") != DATASET_EVIDENCE_INVENTORY_SCHEMA_VERSION:
        raise ValueError(f"Dataset evidence inventory {path} must set schema_version: 1.")
    _require_string(raw, "cohort_id", description="Dataset evidence inventory")
    if raw.get("unit_of_independence") != "mouse":
        raise ValueError("Dataset evidence inventory must use mouse-level groups.")
    authority = _require_string(raw, "label_authority", description="Dataset evidence inventory")
    if authority not in LABEL_AUTHORITY_TIERS:
        raise ValueError(
            f"Dataset evidence inventory has unsupported label_authority: {authority}."
        )
    requirements = raw.get("artifact_requirements")
    if not isinstance(requirements, list) or not all(
        isinstance(requirement, str) and requirement for requirement in requirements
    ):
        raise ValueError(
            "Dataset evidence inventory artifact_requirements must be a list of names."
        )
    groups = raw.get("complete_mouse_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError(
            "Dataset evidence inventory must contain a non-empty complete_mouse_groups list."
        )

    seen_group_ids: set[str] = set()
    for index, group in enumerate(groups, start=1):
        description = f"Dataset evidence inventory group {index}"
        if not isinstance(group, dict):
            raise ValueError(f"{description} must be a mapping.")
        group_id = _require_string(group, "group_id", description=description)
        if group_id in seen_group_ids:
            raise ValueError(f"Dataset evidence inventory has duplicate group_id: {group_id}.")
        seen_group_ids.add(group_id)
        _require_string(group, "age", description=description)
        _require_string(group, "genotype", description=description)
