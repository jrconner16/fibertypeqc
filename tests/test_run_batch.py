from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.run_batch import (
    V0_PARAMS,
    BatchChannelOverrides,
    _error_tail,
    _load_input_manifest,
    _pipeline_timing_lines,
    build_batch_command,
    run_single_image,
)
from src.run_pipeline import build_parser


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


def test_build_batch_command_forwards_crop_controls(tmp_path):
    input_file = tmp_path / "image.czi"
    output_dir = tmp_path / "out"

    cmd = build_batch_command(
        input_file,
        output_dir,
        channel_overrides=BatchChannelOverrides(),
        crop_auto=False,
        crop_ds=4,
        crop_pad=12000,
        crop_min_size=1000,
    )

    assert "--no-crop-auto" in cmd
    assert cmd[cmd.index("--crop-ds") + 1] == "4"
    assert cmd[cmd.index("--crop-pad") + 1] == "12000"
    assert cmd[cmd.index("--crop-min-size") + 1] == "1000"


def test_run_pipeline_parser_allows_disabling_auto_crop():
    args = build_parser().parse_args(
        ["--input", "image.czi", "--output-dir", "out", "--no-crop-auto"]
    )

    assert args.crop_auto is False


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


def test_run_single_image_names_outputs_from_manifest_image_id(tmp_path, monkeypatch):
    input_file = tmp_path / "raw name.czi"
    input_file.write_text("", encoding="utf-8")
    output_dir = tmp_path / "batch"
    seen_cmds = []

    def fake_run(cmd, capture_output, text, timeout, check, cwd):
        seen_cmds.append(cmd)
        image_id = cmd[cmd.index("--image-id") + 1]
        image_output_dir = output_dir / "manifest_image"
        pd.DataFrame({"label": [1, 2]}).to_csv(
            image_output_dir / f"{image_id}_fibers.csv",
            index=False,
        )
        pd.DataFrame({"summary": [1]}).to_csv(
            image_output_dir / f"{image_id}_summary.csv",
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

    assert seen_cmds[0][seen_cmds[0].index("--image-id") + 1] == "manifest_image"
    assert result["image_name"] == "manifest_image"
    assert result["fiber_count"] == 2
    assert result["summary_path"].endswith("manifest_image/manifest_image_summary.csv")
    assert not list((output_dir / "manifest_image").glob("raw*"))


def test_build_batch_command_verifies_frozen_model_manifest_by_default(tmp_path):
    cmd = build_batch_command(
        tmp_path / "image.czi", tmp_path / "out", channel_overrides=BatchChannelOverrides()
    )

    manifest = Path(cmd[cmd.index("--model-manifest") + 1])
    assert manifest.name == "rebaseline_tile_v2_p75p90_iib_iia_iix.yaml"


def test_build_batch_command_only_verifies_custom_classifier_with_its_manifest(tmp_path):
    custom = tmp_path / "custom.joblib"
    without_manifest = build_batch_command(
        tmp_path / "image.czi",
        tmp_path / "out",
        channel_overrides=BatchChannelOverrides(),
        classifier_path=custom,
    )
    with_manifest = build_batch_command(
        tmp_path / "image.czi",
        tmp_path / "out",
        channel_overrides=BatchChannelOverrides(),
        classifier_path=custom,
        model_manifest=tmp_path / "custom.yaml",
    )

    assert "--model-manifest" not in without_manifest
    assert with_manifest[with_manifest.index("--model-manifest") + 1].endswith("custom.yaml")


def test_error_tail_keeps_the_end_of_child_tracebacks():
    stderr = "\n".join(["noise"] * 50 + ["Traceback (most recent call last):", "ValueError: bad"])

    tail = _error_tail(stderr)

    assert tail.endswith("ValueError: bad")
    assert len(tail.splitlines()) == 8


def test_pinned_model_does_not_receive_legacy_typing_flags(tmp_path):
    overrides = BatchChannelOverrides(channel_config=tmp_path / "panel.yaml")
    cmd = build_batch_command(
        tmp_path / "image.czi", tmp_path / "out", overrides, model_id="quad_four_class_rf_v1"
    )

    assert cmd[cmd.index("--model") + 1] == "quad_four_class_rf_v1"
    legacy_flags = ("--typing-preprocess", "--typing-tile-size", "--classifier-path")
    for flag in (*legacy_flags, "--type1-channel"):
        assert flag not in cmd


def test_legacy_model_id_keeps_frozen_v0_typing_flags(tmp_path):
    cmd = build_batch_command(
        tmp_path / "image.czi",
        tmp_path / "out",
        BatchChannelOverrides(),
        model_id="rebaseline_tile_v2_p75p90_iib_iia_iix",
    )

    assert cmd[cmd.index("--typing-preprocess") + 1] == V0_PARAMS["typing_preprocess"]
    assert cmd[cmd.index("--type1-channel") + 1] == str(V0_PARAMS["type1_channel"])


def test_batch_defaults_to_registry_model_and_requires_panel(tmp_path, monkeypatch, capsys):
    from src import run_batch

    monkeypatch.setattr(
        "sys.argv", ["run_batch", "--input-dir", str(tmp_path), "--output-dir", str(tmp_path)]
    )
    with pytest.raises(SystemExit):
        run_batch.main()

    assert "model 'quad_four_class_rf_v1' needs --panel-config" in capsys.readouterr().err
