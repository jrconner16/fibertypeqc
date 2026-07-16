import pytest

from fibertypeqc.model_manifest import load_model_manifest, validate_model_compatibility
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


def test_model_manifest_rejects_feature_schema_not_available_in_legacy_pipeline(tmp_path):
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
    with pytest.raises(ValueError, match="feature schema"):
        validate_model_compatibility(panel, manifest)
