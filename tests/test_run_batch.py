from __future__ import annotations

from src.run_batch import V0_PARAMS, BatchChannelOverrides, build_batch_command


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
