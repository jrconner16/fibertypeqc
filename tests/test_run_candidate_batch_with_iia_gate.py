from __future__ import annotations

from src.run_batch import BatchChannelOverrides
from src.run_candidate_batch_with_iia_gate import _build_pipeline_command


def test_build_pipeline_command_uses_frozen_v0_channel_flags_by_default(tmp_path):
    input_file = tmp_path / "image.czi"
    output_dir = tmp_path / "out"
    classifier_path = tmp_path / "candidate.joblib"

    cmd = _build_pipeline_command(
        input_file=input_file,
        output_dir=output_dir,
        classifier_path=classifier_path,
        channel_overrides=BatchChannelOverrides(),
        downsample_factor=None,
        export_diagnostics=False,
        retain_mode="full",
    )

    assert "--type1-channel" in cmd
    assert "--type2-channel" in cmd
    assert "--membrane-channel" in cmd
    assert "--iia-channel" not in cmd
    assert "--iib-channel" not in cmd
    assert "--channel-config" not in cmd


def test_build_pipeline_command_uses_panel_aware_flags_when_configured(tmp_path):
    input_file = tmp_path / "image.czi"
    output_dir = tmp_path / "out"
    classifier_path = tmp_path / "candidate.joblib"
    config_path = tmp_path / "panel.yml"
    config_path.write_text("channels:\n  membrane: 2\n  markers:\n    iia: 1\n    iib: 3\n")

    cmd = _build_pipeline_command(
        input_file=input_file,
        output_dir=output_dir,
        classifier_path=classifier_path,
        channel_overrides=BatchChannelOverrides(
            channel_config=config_path,
            membrane_channel=2,
            iia_channel=1,
            iib_channel=3,
        ),
        downsample_factor=3,
        export_diagnostics=True,
        retain_mode="tables",
    )

    assert "--channel-config" in cmd
    assert str(config_path.resolve()) in cmd
    assert "--iia-channel" in cmd
    assert "--iib-channel" in cmd
    assert "--membrane-channel" in cmd
    assert "--type1-channel" not in cmd
    assert "--type2-channel" not in cmd
    assert cmd[cmd.index("--downsample-factor") + 1] == "3"
    assert "--export-diagnostics" in cmd
