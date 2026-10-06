from pathlib import Path

import pytest
import yaml

from fibertypeqc.evidence_registry import (
    validate_dataset_evidence_inventory,
    validate_dataset_split_ledger,
    validate_model_registry,
)


def test_tracked_model_registry_validates():
    root = Path(__file__).resolve().parents[1]
    validate_model_registry(root / "manifests/model_registry.v1.yaml", repo_root=root)


def test_dataset_ledger_rejects_group_crossing_split_roles(tmp_path):
    path = tmp_path / "ledger.yaml"
    path.write_text(
        "schema_version: 1\n"
        "ledger_id: test\n"
        "amendment_policy: append_only_new_version\n"
        "cohorts:\n"
        "  - cohort_id: ta\n"
        "    data_availability: private\n"
        "    label_authority: manual_gold\n"
        "group_assignments:\n"
        "  - cohort_id: ta\n"
        "    group_id: mouse_1\n"
        "    role: development\n"
        "  - cohort_id: ta\n"
        "    group_id: mouse_1\n"
        "    role: evaluation\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="more than once"):
        validate_dataset_split_ledger(path)


def test_synthetic_dataset_ledger_example_validates():
    root = Path(__file__).resolve().parents[1]
    validate_dataset_split_ledger(root / "examples/reference/dataset_split_ledger.example.yaml")


def test_synthetic_dataset_ledger_has_no_private_identifiers():
    root = Path(__file__).resolve().parents[1]
    path = root / "examples/reference/dataset_split_ledger.example.yaml"
    raw = yaml.safe_load(path.read_text())
    serialized = path.read_text().lower()

    assert raw["ledger_id"].startswith("synthetic_example")
    assert {row["cohort_id"] for row in raw["cohorts"]} == {"example_ta"}
    assert "cohort_b" not in serialized
    assert "jag" not in serialized


def test_dataset_ledger_rejects_split_contract_count_mismatch(tmp_path):
    path = tmp_path / "ledger.yaml"
    path.write_text(
        "schema_version: 1\n"
        "ledger_id: test\n"
        "amendment_policy: append_only_new_version\n"
        "cohorts:\n"
        "  - cohort_id: ta\n"
        "    data_availability: private\n"
        "    label_authority: manual_gold\n"
        "split_contracts:\n"
        "  - contract_id: ta_split.v1\n"
        "    cohort_id: ta\n"
        "    scope: pilot\n"
        "    unit_of_independence: mouse\n"
        "    assignment_group_prefix: mouse_\n"
        "    role_counts:\n"
        "      development: 2\n"
        "    private_artifact_sha256:\n"
        f'      group_split_csv: "{"0" * 64}"\n'
        "group_assignments:\n"
        "  - cohort_id: ta\n"
        "    group_id: mouse_1\n"
        "    role: development\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="role_counts do not match assignments"):
        validate_dataset_split_ledger(path)


def test_synthetic_dataset_evidence_inventory_validates():
    root = Path(__file__).resolve().parents[1]
    validate_dataset_evidence_inventory(
        root / "examples/reference/dataset_evidence_inventory.example.yaml"
    )


def test_dataset_ledger_accepts_mixed_cohort_label_authority(tmp_path):
    path = tmp_path / "ledger.yaml"
    path.write_text(
        "schema_version: 1\n"
        "ledger_id: test\n"
        "amendment_policy: append_only_new_version\n"
        "cohorts:\n"
        "  - cohort_id: ta\n"
        "    data_availability: private\n"
        "    label_authority: mixed\n"
        "group_assignments: []\n",
        encoding="utf-8",
    )

    validate_dataset_split_ledger(path)


def _registry_entry(**overrides):
    entry = {
        "model_id": "private_model_v1",
        "task": "fiber_identity",
        "status": "released",
        "required_markers": ["laminin", "type_i"],
        "feature_schema": "multiplanel_features.v1",
        "artifact": "private_model_v1.joblib",
        "artifact_location": "private",
        "artifact_sha256": "a" * 64,
        "development_groups": "private",
        "evaluation_groups": "private",
        "limitations": "synthetic test entry",
    }
    entry.update(overrides)
    return entry


def test_registry_accepts_private_artifact_with_digest_only(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "models": [_registry_entry()]}))

    validate_model_registry(path, repo_root=tmp_path)


def test_registry_rejects_private_artifact_paths_and_missing_digests(tmp_path):
    path = tmp_path / "registry.yaml"
    for entry, message in (
        (_registry_entry(artifact="private/dir/model.joblib"), "bare file name"),
        (_registry_entry(artifact_sha256=None), "SHA-256"),
        (_registry_entry(artifact_location="cloud"), "artifact_location"),
    ):
        path.write_text(yaml.safe_dump({"schema_version": 1, "models": [entry]}))
        with pytest.raises(ValueError, match=message):
            validate_model_registry(path, repo_root=tmp_path)
