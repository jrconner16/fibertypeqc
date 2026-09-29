from pathlib import Path

import pytest
import yaml

from fibertypeqc.model_manifest import load_model_manifest
from fibertypeqc.model_resolution import default_model_id, resolve_model

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_quad_is_the_registered_default_fiber_model():
    assert default_model_id("fiber_identity") == "quad_four_class_rf_v1"


def test_tracked_model_resolves_inside_repository():
    resolved = resolve_model("synthetic_four_class_reference_v1")

    assert resolved.artifact_path.is_file()
    assert load_model_manifest(resolved.manifest_path).model_id == resolved.model_id


def test_private_model_requires_model_root(monkeypatch, tmp_path):
    monkeypatch.delenv("FIBERTYPEQC_MODEL_ROOT", raising=False)
    with pytest.raises(ValueError, match="FIBERTYPEQC_MODEL_ROOT"):
        resolve_model("quad_four_class_rf_v1")

    with pytest.raises(ValueError, match="was not found"):
        resolve_model("quad_four_class_rf_v1", model_root=tmp_path)

    (tmp_path / "quad_four_class_rf_v1.joblib").write_bytes(b"placeholder")
    resolved = resolve_model("quad_four_class_rf_v1", model_root=tmp_path)
    assert resolved.artifact_path == tmp_path / "quad_four_class_rf_v1.joblib"


def test_unknown_model_lists_registered_ids():
    with pytest.raises(ValueError, match="Registered models:.*quad_four_class_rf_v1"):
        resolve_model("missing_model")


def test_quad_manifest_pins_every_feature_setting():
    manifest = load_model_manifest(REPO_ROOT / "manifests/models/quad_four_class_rf_v1.yaml")

    assert manifest.feature_extraction["typing_preprocess"] == "global_subtract"
    assert manifest.feature_extraction["typing_erode_px"] == 2
    assert len(manifest.features) == 19


def test_manifest_rejects_partial_feature_pins(tmp_path):
    raw = yaml.safe_load((REPO_ROOT / "manifests/models/quad_four_class_rf_v1.yaml").read_text())
    del raw["feature_extraction"]["typing_smooth_sigma"]
    path = tmp_path / "partial.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="missing: \\['typing_smooth_sigma'\\]"):
        load_model_manifest(path)
