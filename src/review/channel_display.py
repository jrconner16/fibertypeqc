"""Panel-aware, display-only channel names and colors for project review."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fibertypeqc.config import DEFAULT_CHANNEL_CONFIG, ChannelConfig, load_channel_config
from src.typing_display import normalize_for_display


@dataclass(frozen=True)
class ChannelDisplay:
    role: str
    channel: int
    colormap: str
    rgb: tuple[float, float, float]


_ROLE_DISPLAY = {
    "Type I": ("blue", (0.2, 0.4, 1.0)),
    "IIa": ("green", (0.1, 0.9, 0.2)),
    "IIb": ("magenta", (1.0, 0.0, 0.8)),
    "IIx": ("orange", (1.0, 0.55, 0.0)),
    "eMHC": ("yellow", (1.0, 0.9, 0.0)),
    "DAPI": ("cyan", (0.0, 0.9, 1.0)),
    "Membrane": ("gray", (0.8, 0.8, 0.8)),
}


def _channel_config(panel_manifest: Path) -> ChannelConfig:
    try:
        return load_channel_config(panel_manifest)
    except (OSError, ValueError):
        return DEFAULT_CHANNEL_CONFIG


def channel_displays(panel_manifest: Path, channel_count: int) -> tuple[ChannelDisplay, ...]:
    """Return non-duplicated semantic mappings followed by unnamed channels.

    A malformed or placeholder panel manifest falls back to the legacy default
    mapping for display only; it never changes analysis configuration.
    """
    config = _channel_config(panel_manifest)
    roles = (
        ("Type I", config.i_channel),
        ("IIa", config.iia_channel),
        ("IIb", config.iib_channel),
        ("IIx", config.iix_channel),
        ("eMHC", config.emhc_channel),
        ("DAPI", config.dapi_channel),
        ("Membrane", config.membrane_channel),
    )
    displays: list[ChannelDisplay] = []
    used: set[int] = set()
    for role, channel in roles:
        if channel is None or channel < 0 or channel >= channel_count or channel in used:
            continue
        colormap, rgb = _ROLE_DISPLAY[role]
        displays.append(ChannelDisplay(role, channel, colormap, rgb))
        used.add(channel)
    for channel in range(channel_count):
        if channel not in used:
            displays.append(ChannelDisplay(f"Channel {channel}", channel, "gray", (0.8, 0.8, 0.8)))
    return tuple(displays)


def stain_composite(image: np.ndarray, displays: tuple[ChannelDisplay, ...]) -> np.ndarray:
    """Build a display-only additive RGB composite from mapped raw channels."""
    if image.ndim != 3:
        raise ValueError(f"Expected CHW raw image, got shape {image.shape}")
    rgb = np.zeros((*image.shape[1:], 3), dtype=np.float32)
    for display in displays:
        signal = normalize_for_display(image[display.channel])
        rgb += signal[..., np.newaxis] * np.asarray(display.rgb, dtype=np.float32)
    return np.clip(rgb, 0.0, 1.0)
