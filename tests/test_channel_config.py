from __future__ import annotations

from pathlib import Path

import pytest

from fibertypeqc.config import ChannelConfig, load_channel_config, resolve_channel_config


def write_yaml(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_load_channel_config_reads_required_channels(tmp_path):
    path = write_yaml(
        tmp_path / "channels.yml",
        "channels:\n  type1: 3\n  type2: 1\n  membrane: 2\n",
    )

    cfg = load_channel_config(path)

    assert cfg.membrane_channel == 2
    assert cfg.iib_channel == 3
    assert cfg.iia_channel == 1
    assert cfg.type1_channel == 3
    assert cfg.type2_channel == 1


def test_load_channel_config_rejects_missing_channels_block(tmp_path):
    path = write_yaml(tmp_path / "channels.yml", "type1: 0\n")

    with pytest.raises(ValueError, match="missing required top-level key 'channels'"):
        load_channel_config(path)


def test_load_channel_config_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError, match="Channel config file not found"):
        load_channel_config(tmp_path / "missing.yml")


def test_load_channel_config_rejects_missing_required_entries(tmp_path):
    path = write_yaml(tmp_path / "channels.yml", "channels:\n  type1: 0\n  membrane: 2\n")

    with pytest.raises(ValueError, match="missing required channel entries: type2"):
        load_channel_config(path)


def test_load_channel_config_rejects_unknown_keys(tmp_path):
    path = write_yaml(
        tmp_path / "channels.yml",
        "channels:\n  type1: 0\n  type2: 1\n  membrane: 2\n  bad: 3\n",
    )

    with pytest.raises(ValueError, match="unknown channel keys: bad"):
        load_channel_config(path)


def test_load_channel_config_rejects_non_integer_channels(tmp_path):
    path = write_yaml(
        tmp_path / "channels.yml",
        "channels:\n  type1: zero\n  type2: 1\n  membrane: 2\n",
    )

    with pytest.raises(ValueError, match="must be an integer"):
        load_channel_config(path)


def test_load_channel_config_rejects_duplicate_channel_indices(tmp_path):
    path = write_yaml(
        tmp_path / "channels.yml",
        "channels:\n  type1: 1\n  type2: 1\n  membrane: 2\n",
    )

    with pytest.raises(ValueError, match="reuses channel indices"):
        load_channel_config(path)


def test_resolve_channel_config_uses_defaults_without_yaml():
    cfg, warnings = resolve_channel_config(
        channel_config_path=None,
        i_channel=None,
        iia_channel=None,
        iib_channel=None,
        iix_channel=None,
        dapi_channel=None,
        type1_channel=None,
        type2_channel=None,
        membrane_channel=None,
    )

    assert cfg == ChannelConfig()
    assert warnings == []


def test_resolve_channel_config_uses_yaml_when_cli_not_set(tmp_path):
    path = write_yaml(
        tmp_path / "channels.yml",
        "channels:\n  type1: 4\n  type2: 5\n  membrane: 6\n",
    )

    cfg, warnings = resolve_channel_config(
        channel_config_path=path,
        i_channel=None,
        iia_channel=None,
        iib_channel=None,
        iix_channel=None,
        dapi_channel=None,
        type1_channel=None,
        type2_channel=None,
        membrane_channel=None,
    )

    assert cfg.iib_channel == 4
    assert cfg.iia_channel == 5
    assert cfg.membrane_channel == 6
    assert warnings == []


def test_resolve_channel_config_cli_overrides_yaml_with_warning(tmp_path):
    path = write_yaml(
        tmp_path / "channels.yml",
        "channels:\n  type1: 4\n  type2: 5\n  membrane: 6\n",
    )

    cfg, warnings = resolve_channel_config(
        channel_config_path=path,
        i_channel=None,
        iia_channel=None,
        iib_channel=None,
        iix_channel=None,
        dapi_channel=None,
        type1_channel=0,
        type2_channel=None,
        membrane_channel=2,
    )

    assert cfg.iib_channel == 0
    assert cfg.iia_channel == 5
    assert cfg.membrane_channel == 2
    assert warnings == [
        f"--membrane-channel=2 overrides {path}:membrane_channel=6",
        f"--type1-channel=0 overrides {path}:iib_channel=4",
        "--type1-channel is a legacy alias for --iib-channel",
    ]


def test_resolve_channel_config_modern_flags_override_yaml(tmp_path):
    path = write_yaml(
        tmp_path / "panel.yml",
        "channels:\n  membrane: 2\n  markers:\n    iia: 1\n    iib: 0\n    i: null\n",
    )

    cfg, warnings = resolve_channel_config(
        channel_config_path=path,
        i_channel=3,
        iia_channel=5,
        iib_channel=4,
        iix_channel=None,
        dapi_channel=6,
        type1_channel=None,
        type2_channel=None,
        membrane_channel=7,
    )

    assert cfg.i_channel == 3
    assert cfg.iia_channel == 5
    assert cfg.iib_channel == 4
    assert cfg.dapi_channel == 6
    assert cfg.membrane_channel == 7
    assert warnings == [
        f"--i-channel=3 overrides {path}:i_channel=None",
        f"--iia-channel=5 overrides {path}:iia_channel=1",
        f"--iib-channel=4 overrides {path}:iib_channel=0",
        f"--dapi-channel=6 overrides {path}:dapi_channel=None",
        f"--membrane-channel=7 overrides {path}:membrane_channel=2",
    ]


def test_resolve_channel_config_modern_flags_beat_legacy_aliases(tmp_path):
    path = write_yaml(
        tmp_path / "panel.yml",
        "channels:\n  membrane: 2\n  markers:\n    iia: 1\n    iib: 0\n",
    )

    cfg, warnings = resolve_channel_config(
        channel_config_path=path,
        i_channel=None,
        iia_channel=8,
        iib_channel=7,
        iix_channel=None,
        dapi_channel=None,
        type1_channel=4,
        type2_channel=5,
        membrane_channel=None,
    )

    assert cfg.iib_channel == 7
    assert cfg.iia_channel == 8
    assert warnings == [
        f"--iia-channel=8 overrides {path}:iia_channel=1",
        f"--iib-channel=7 overrides {path}:iib_channel=0",
        "--type1-channel=4 ignored because --iib-channel=7 was also provided",
        "--type2-channel=5 ignored because --iia-channel=8 was also provided",
    ]


def test_load_channel_config_reads_nested_panel_schema(tmp_path):
    path = write_yaml(
        tmp_path / "panel.yml",
        "\n".join(
            [
                "channels:",
                "  membrane: 3",
                "  dapi: null",
                "  markers:",
                "    i: 0",
                "    iia: 1",
                "    iib: null",
                "    iix: 2",
                "classification:",
                "  residual_inference:",
                "    enabled: false",
                "    target_class: iix",
                "    requires_negative_markers: [i, iia]",
            ]
        )
        + "\n",
    )

    cfg = load_channel_config(path)

    assert cfg.membrane_channel == 3
    assert cfg.dapi_channel is None
    assert cfg.i_channel == 0
    assert cfg.iia_channel == 1
    assert cfg.iib_channel is None
    assert cfg.iix_channel == 2
    assert cfg.available_markers == ("i", "iia", "iix")
    assert cfg.residual_inference_enabled is False
    assert cfg.residual_target_class == "iix"
    assert cfg.residual_requires_negative_markers == ("i", "iia")


def test_load_channel_config_rejects_duplicate_indices_in_nested_schema(tmp_path):
    path = write_yaml(
        tmp_path / "panel.yml",
        "channels:\n  membrane: 2\n  markers:\n    iia: 1\n    iib: 1\n",
    )

    with pytest.raises(ValueError, match="reuses channel indices across configured channels"):
        load_channel_config(path)


def test_load_channel_config_rejects_bad_residual_target(tmp_path):
    path = write_yaml(
        tmp_path / "panel.yml",
        "\n".join(
            [
                "channels:",
                "  membrane: 2",
                "  markers:",
                "    iia: 1",
                "    iib: 0",
                "classification:",
                "  residual_inference:",
                "    enabled: true",
                "    target_class: bad",
            ]
        )
        + "\n",
    )

    with pytest.raises(ValueError, match="Invalid marker name"):
        load_channel_config(path)
