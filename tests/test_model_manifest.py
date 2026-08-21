import pytest

from fibertypeqc.model_manifest import (
    load_model_manifest,
    validate_model_artifact,
    validate_model_compatibility,
)
from fibertypeqc.panels import Panel


def test_model_manifest_requires_observed_markers(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text(
        "manifest_version: 1\n"
        "model_id: type_i_demo\n"
        "task: fiber_identity\n"
        "feature_schema_version: multiplanel_features.v1\n"
        "required_markers: [laminin, type_i]\n"
        "outputs: [type_i, uncertain]\n"
    )
    manifest = load_model_manifest(path)
    with pytest.raises(ValueError, match="type_i"):
        validate_model_compatibility(Panel({"laminin": 0, "type_i": None}), manifest)


def test_model_manifest_accepts_semantic_feature_schema(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text(
        "manifest_version: 1\n"
        "model_id: future_semantic_model\n"
        "task: fiber_identity\n"
        "feature_schema_version: multiplanel_features.v1\n"
        "required_markers: [laminin, type_iia, type_iib]\n"
        "outputs: [type_iia, uncertain]\n"
    )
    manifest = load_model_manifest(path)
    panel = Panel({"laminin": 0, "type_iia": 1, "type_iib": 2})
    validate_model_compatibility(panel, manifest)


def test_diagnostics_only_panel_does_not_require_legacy_model_markers():
    validate_model_compatibility(Panel({"laminin": 0, "type_i": 1}), None)


def test_legacy_model_still_requires_legacy_markers():
    with pytest.raises(ValueError, match="type_iib"):
        validate_model_compatibility(
            Panel({"laminin": 0, "type_iia": 1}), None, require_legacy_model=True
        )


def test_model_manifest_verifies_artifact_digest(tmp_path):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"frozen model")
    path = tmp_path / "model.yaml"
    path.write_text(
        "manifest_version: 1\n"
        "model_id: frozen_demo\n"
        "task: fiber_identity\n"
        "feature_schema_version: legacy_type1_type2.v1\n"
        "required_markers: [laminin, type_iia, type_iib]\n"
        "outputs: [iia, iib, iix]\n"
        "artifact: model.joblib\n"
        "artifact_sha256: e53d989a57dfed763fd28993b27f15ef304bc2ed1d384f2fff91afbe889c5fab\n"
        "intended_use: deterministic reference fixture\n"
    )

    manifest = load_model_manifest(path)
    validate_model_artifact(artifact, manifest)

    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_model_artifact(artifact, manifest)
