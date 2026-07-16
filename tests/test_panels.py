import pytest

from fibertypeqc.panels import Panel


def test_panel_validates_image_bounds_and_fingerprint():
    panel = Panel(
        {
            "laminin": 2,
            "type_iia": 1,
            "type_iib": 0,
            "dapi": None,
            "type_i": None,
            "type_iix": None,
            "emhc": None,
        }
    )
    panel.validate(image_channel_count=3)
    assert len(panel.fingerprint) == 64


def test_panel_rejects_more_than_four_active_channels():
    panel = Panel({"laminin": 0, "dapi": 1, "type_i": 2, "type_iia": 3, "type_iib": 4})
    with pytest.raises(ValueError, match="at most four"):
        panel.validate(image_channel_count=5)


def test_panel_rejects_duplicate_indices_after_cli_resolution():
    panel = Panel({"laminin": 0, "type_iia": 1, "type_iib": 1})
    with pytest.raises(ValueError, match="unique"):
        panel.validate(image_channel_count=2)
