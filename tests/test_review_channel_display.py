from __future__ import annotations

import numpy as np

from src.review.channel_display import channel_displays, stain_composite


def test_channel_display_uses_panel_mapping_and_preserves_unmapped_channels(tmp_path) -> None:
    panel = tmp_path / "panel.yaml"
    panel.write_text(
        "channels:\n  membrane: 2\n  markers:\n    iib: 3\n    iia: 1\n    i: 0\n",
        encoding="utf-8",
    )

    displays = channel_displays(panel, 5)

    assert [(display.role, display.channel) for display in displays] == [
        ("Type I", 0),
        ("IIa", 1),
        ("IIb", 3),
        ("Membrane", 2),
        ("Channel 4", 4),
    ]


def test_stain_composite_is_rgb_and_display_bounded(tmp_path) -> None:
    panel = tmp_path / "panel.yaml"
    panel.write_text("channels:\n  membrane: 2\n", encoding="utf-8")
    image = np.zeros((3, 4, 5), dtype=np.float32)
    image[0, 1, 1] = 10
    image[1, 1, 1] = 10

    composite = stain_composite(image, channel_displays(panel, 3))

    assert composite.shape == (4, 5, 3)
    assert composite.min() >= 0
    assert composite.max() <= 1
