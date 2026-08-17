from pathlib import Path

import pytest

from src.run_quad_semantic_array_task import _load_manifest, build_pipeline_command


def test_load_manifest_accepts_only_unique_raw_czi_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,input_relpath\nquad_a,cohort/quad_a.czi\n")

    assert _load_manifest(manifest)[0].input_relpath == Path("cohort/quad_a.czi")

    manifest.write_text("image_id,input_relpath\nquad_a,cohort/quad_a.tif\n")
    with pytest.raises(ValueError, match="raw .czi"):
        _load_manifest(manifest)

    manifest.write_text("image_id,input_relpath\nquad_a,a.czi\nquad_a,b.czi\n")
    with pytest.raises(ValueError, match="duplicate image_id"):
        _load_manifest(manifest)


def test_build_pipeline_command_uses_semantic_model_contract(tmp_path: Path) -> None:
    command = build_pipeline_command(
        input_path=tmp_path / "raw_section.tif",
        output_dir=tmp_path / "output",
        panel_config=tmp_path / "panel.yaml",
        classifier_path=tmp_path / "model.joblib",
        model_manifest=tmp_path / "model.yaml",
        fiber_downsample_factor=2,
        fiber_diameter=30,
    )

    assert command[command.index("--panel-config") + 1] == str(tmp_path / "panel.yaml")
    assert command[command.index("--classifier-path") + 1] == str(tmp_path / "model.joblib")
    assert command[command.index("--model-manifest") + 1] == str(tmp_path / "model.yaml")
    assert command[command.index("--requested-domain") + 1] == "fiber_identity"
    assert "--export-diagnostics" in command
