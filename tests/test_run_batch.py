from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.run_batch import (
    V0_PARAMS,
    BatchChannelOverrides,
    _load_input_manifest,
    _pipeline_timing_lines,
    build_batch_command,
    run_single_image,
)


def test_pipeline_timing_lines_excludes_summary_payload():
    stdout = "\n".join(
        [
            "[1/7] prepare output + load image ...",
            "Cellpose device: mps",
            "summary: {'many': 'fields'}",
            "[3/7] done: segment fibers with Cellpose (12.3s)",
            "total runtime: 20.0s",
        ]
    )
    assert _pipeline_timing_lines(stdout) == [
        "[1/7] prepare output + load image ...",
        "Cellpose device: mps",
        "[3/7] done: segment fibers with Cellpose (12.3s)",
        "total runtime: 20.0s",
    ]


def test_build_batch_command_uses_frozen_v0_flags_by_default(tmp_path):
    input_file = tmp_path / "image.czi"
    output_dir = tmp_path / "out"

    cmd = build_batch_command(
        input_file,
        output_dir,
        channel_overrides=BatchChannelOverrides(),
    )

    assert "--type1-channel" in cmd
    assert "--type2-channel" in cmd
    assert "--membrane-channel" in cmd
    assert "--iib-channel" not in cmd
    assert "--iia-channel" not in cmd
    assert "--channel-config" not in cmd
    assert str(V0_PARAMS["type1_channel"]) in cmd
    assert str(V0_PARAMS["type2_channel"]) in cmd
    assert str(V0_PARAMS["membrane_channel"]) in cmd


def test_build_batch_command_uses_panel_aware_flags_when_configured(tmp_path):
    input_file = tmp_path / "image.czi"
    output_dir = tmp_path / "out"
    config_path = tmp_path / "panel.yml"
    config_path.write_text("channels:\n  membrane: 2\n  markers:\n    iia: 1\n    iib: 0\n")

    cmd = build_batch_command(
        input_file,
        output_dir,
        channel_overrides=BatchChannelOverrides(
            channel_config=config_path,
            membrane_channel=4,
            iia_channel=5,
            iib_channel=6,
        ),
        downsample_factor=3,
    )

    assert "--channel-config" in cmd
    assert str(config_path.resolve()) in cmd
    assert "--iia-channel" in cmd
    assert "--iib-channel" in cmd
    assert "--membrane-channel" in cmd
    assert "--type1-channel" not in cmd
    assert "--type2-channel" not in cmd
    assert cmd[cmd.index("--downsample-factor") + 1] == "3"


def test_build_batch_command_can_export_diagnostics(tmp_path):
    input_file = tmp_path / "image.czi"
    output_dir = tmp_path / "out"

    cmd = build_batch_command(
        input_file,
        output_dir,
        channel_overrides=BatchChannelOverrides(),
        export_diagnostics=True,
    )

    assert "--export-diagnostics" in cmd


def test_build_batch_command_can_override_classifier_path(tmp_path):
    input_file = tmp_path / "image.czi"
    output_dir = tmp_path / "out"
    classifier_path = tmp_path / "candidate.joblib"

    cmd = build_batch_command(
        input_file,
        output_dir,
        channel_overrides=BatchChannelOverrides(),
        classifier_path=classifier_path,
    )

    assert "--classifier-path" in cmd
    assert cmd[cmd.index("--classifier-path") + 1] == str(classifier_path.resolve())


def test_build_batch_command_can_set_retain_mode(tmp_path):
    input_file = tmp_path / "image.czi"
    output_dir = tmp_path / "out"

    cmd = build_batch_command(
        input_file,
        output_dir,
        channel_overrides=BatchChannelOverrides(),
        retain_mode="tables",
    )

    assert "--retain-mode" in cmd
    assert cmd[cmd.index("--retain-mode") + 1] == "tables"


def test_load_input_manifest_requires_image_id_and_input_path(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,input_path\nimg1,/tmp/image1.czi\n", encoding="utf-8")

    rows = _load_input_manifest(manifest)

    assert rows == [("img1", Path("/tmp/image1.czi"))]


def test_load_input_manifest_resolves_portable_relative_paths(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,input_relpath\nimg1,cohort/image1.czi\n", encoding="utf-8")

    rows = _load_input_manifest(manifest, input_root=Path("/data"))

    assert rows == [("img1", Path("/data/cohort/image1.czi"))]


def test_load_input_manifest_requires_root_for_relative_paths(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,input_relpath\nimg1,cohort/image1.czi\n", encoding="utf-8")

    with pytest.raises(ValueError, match="--input-root"):
        _load_input_manifest(manifest)


def test_run_single_image_uses_manifest_image_name_for_outputs(tmp_path, monkeypatch):
    input_file = tmp_path / "raw name.czi"
    input_file.write_text("", encoding="utf-8")
    output_dir = tmp_path / "batch"

    def fake_run(cmd, capture_output, text, timeout, check, cwd):
        image_output_dir = output_dir / "manifest_image"
        source_stem = "raw_name"
        pd.DataFrame({"label": [1, 2]}).to_csv(
            image_output_dir / f"{source_stem}_fibers.csv",
            index=False,
        )
        pd.DataFrame({"summary": [1]}).to_csv(
            image_output_dir / f"{source_stem}_summary.csv",
            index=False,
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("src.run_batch.subprocess.run", fake_run)

    result = run_single_image(
        input_file,
        output_dir,
        channel_overrides=BatchChannelOverrides(),
        image_name="manifest_image",
    )

    assert result["image_name"] == "manifest_image"
    assert result["fiber_count"] == 2
    assert result["summary_path"].endswith("manifest_image/manifest_image_summary.csv")
    assert (output_dir / "manifest_image" / "manifest_image_fibers.csv").exists()
    assert (output_dir / "manifest_image" / "manifest_image_summary.csv").exists()
