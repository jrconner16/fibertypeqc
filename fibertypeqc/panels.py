"""Semantic panel validation shared by pipeline entry points.

This module deliberately describes observed channels only.  It does not select a
classifier or infer an unobserved biological state.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

from fibertypeqc.config import ChannelConfig

PANEL_SCHEMA_VERSION = 1
OBSERVED_CHANNELS = ("laminin", "dapi", "type_i", "type_iia", "type_iib", "type_iix", "emhc")


@dataclass(frozen=True)
class Panel:
    """A semantic mapping from observed channel names to image indices."""

    channels: Mapping[str, int | None]

    @classmethod
    def from_channel_config(cls, config: ChannelConfig) -> Panel:
        return cls(
            {
                "laminin": config.membrane_channel,
                "dapi": config.dapi_channel,
                "type_i": config.i_channel,
                "type_iia": config.iia_channel,
                "type_iib": config.iib_channel,
                "type_iix": config.iix_channel,
                "emhc": None,
            }
        )

    @property
    def active_channels(self) -> dict[str, int]:
        return {name: index for name, index in self.channels.items() if index is not None}

    @property
    def fingerprint(self) -> str:
        payload = {name: self.channels.get(name) for name in OBSERVED_CHANNELS}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()

    def validate(self, *, image_channel_count: int) -> None:
        unknown = set(self.channels) - set(OBSERVED_CHANNELS)
        if unknown:
            raise ValueError(f"Unsupported panel channels: {', '.join(sorted(unknown))}.")
        if self.channels.get("laminin") is None:
            raise ValueError("A laminin channel is required for fiber segmentation.")
        active = self.active_channels
        if len(active) > 4:
            raise ValueError("A panel supports at most four active observed channels.")
        indices = list(active.values())
        if len(indices) != len(set(indices)):
            raise ValueError("Panel channel indices must be unique.")
        for name, index in active.items():
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError(f"Invalid {name} channel index {index!r}.")
            if index >= image_channel_count:
                raise ValueError(
                    f"Invalid {name} channel {index} for image with {image_channel_count} channels."
                )
