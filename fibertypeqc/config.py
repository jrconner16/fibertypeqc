"""Panel-aware channel-config loading and legacy resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

MARKER_NAMES = ("i", "iia", "iib", "iix")
PANEL_MARKER_NAMES = (*MARKER_NAMES, "emhc")


@dataclass(frozen=True)
class ChannelConfig:
    membrane_channel: int = 2
    dapi_channel: int | None = None
    i_channel: int | None = None
    iia_channel: int = 1
    iib_channel: int = 0
    iix_channel: int | None = None
    emhc_channel: int | None = None
    residual_inference_enabled: bool = True
    residual_target_class: str | None = "iix"
    residual_requires_negative_markers: tuple[str, ...] = ("iia", "iib")

    @property
    def type1_channel(self) -> int:
        """Legacy alias for the default IIb marker channel."""
        return self.iib_channel

    @property
    def type2_channel(self) -> int:
        """Legacy alias for the default IIa marker channel."""
        return self.iia_channel

    @property
    def available_markers(self) -> tuple[str, ...]:
        markers: list[str] = []
        for marker in MARKER_NAMES:
            if getattr(self, f"{marker}_channel") is not None:
                markers.append(marker)
        return tuple(markers)

DEFAULT_CHANNEL_CONFIG = ChannelConfig()

_LEGACY_CHANNEL_KEY_ALIASES = {
    "type1": "iib_channel",
    "type1_channel": "iib_channel",
    "type2": "iia_channel",
    "type2_channel": "iia_channel",
    "membrane": "membrane_channel",
    "membrane_channel": "membrane_channel",
    "dapi": "dapi_channel",
    "dapi_channel": "dapi_channel",
    "i": "i_channel",
    "i_channel": "i_channel",
    "iia": "iia_channel",
    "iia_channel": "iia_channel",
    "iib": "iib_channel",
    "iib_channel": "iib_channel",
    "iix": "iix_channel",
    "iix_channel": "iix_channel",
    "emhc": "emhc_channel",
    "emhc_channel": "emhc_channel",
}

_CANONICAL_PANEL_CHANNEL_ALIASES = {
    "laminin": "membrane_channel",
    "dapi": "dapi_channel",
    "type_i": "i_channel",
    "type_iia": "iia_channel",
    "type_iib": "iib_channel",
    "type_iix": "iix_channel",
    "emhc": "emhc_channel",
}


def _validate_channel_value(key: str, value: object, *, allow_none: bool) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        expected = "integer or null" if allow_none else "integer"
        raise ValueError(f"Channel '{key}' must be an {expected}, got {value!r}.")
    if value < 0:
        raise ValueError(f"Channel '{key}' must be >= 0, got {value}.")
    return value


def _validate_marker_name(marker: object) -> str:
    name = str(marker).strip().lower()
    if name not in MARKER_NAMES:
        allowed = ", ".join(MARKER_NAMES)
        raise ValueError(f"Invalid marker name {marker!r}. Allowed markers: {allowed}.")
    return name


def _validate_residual_config(path: Path, raw: object) -> tuple[bool, str | None, tuple[str, ...]]:
    if raw is None:
        return True, "iix", ("iia", "iib")
    if not isinstance(raw, dict):
        raise ValueError(
            f"Channel config {path} has invalid 'classification.residual_inference' value. "
            "Expected a mapping."
        )

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(
            f"Channel config {path} has invalid residual inference 'enabled' value {enabled!r}. "
            "Expected true or false."
        )

    target = raw.get("target_class", "iix")
    if target is not None:
        target = _validate_marker_name(target)

    requires = raw.get("requires_negative_markers", ("iia", "iib"))
    if not isinstance(requires, (list, tuple)):
        raise ValueError(
            f"Channel config {path} has invalid residual inference 'requires_negative_markers'. "
            "Expected a list."
        )
    normalized_requires = tuple(_validate_marker_name(marker) for marker in requires)

    if enabled and target is None:
        raise ValueError(
            f"Channel config {path} enables residual inference but does not define 'target_class'."
        )

    return enabled, target, normalized_requires


def _parse_flat_channels(path: Path, channels: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    unknown_keys = sorted(set(channels) - set(_LEGACY_CHANNEL_KEY_ALIASES))
    if unknown_keys:
        allowed = "membrane, dapi, i, iia, iib, iix, type1, type2"
        raise ValueError(
            f"Channel config {path} has unknown channel keys: {', '.join(unknown_keys)}. "
            f"Allowed keys: {allowed}."
        )

    for raw_key, raw_value in channels.items():
        canonical_key = _LEGACY_CHANNEL_KEY_ALIASES[raw_key]
        if canonical_key in normalized:
            raise ValueError(
                f"Channel config {path} defines '{raw_key}' more than once via aliases."
            )
        allow_none = canonical_key != "membrane_channel"
        normalized[canonical_key] = _validate_channel_value(
            raw_key,
            raw_value,
            allow_none=allow_none,
        )

    missing = [
        key for key in ("iib_channel", "iia_channel", "membrane_channel") if key not in normalized
    ]
    if missing:
        pretty = ", ".join(
            key.replace("_channel", "").replace("iib", "type1").replace("iia", "type2")
            for key in missing
        )
        raise ValueError(
            f"Legacy flat channel config {path} is missing required channel entries: {pretty}."
        )

    return normalized


def _parse_nested_channels(path: Path, channels: dict[str, object]) -> dict[str, object]:
    allowed_top = {"membrane", "membrane_channel", "dapi", "dapi_channel", "markers"}
    unknown_keys = sorted(set(channels) - allowed_top)
    if unknown_keys:
        raise ValueError(
            f"Channel config {path} has unknown top-level channel keys: {', '.join(unknown_keys)}. "
            "Allowed keys: membrane, dapi, markers."
        )

    normalized: dict[str, object] = {}
    membrane_key = "membrane_channel" if "membrane_channel" in channels else "membrane"
    if membrane_key not in channels:
        raise ValueError(f"Channel config {path} is missing required channel entry: membrane.")
    normalized["membrane_channel"] = _validate_channel_value(
        "membrane",
        channels[membrane_key],
        allow_none=False,
    )

    if "dapi" in channels or "dapi_channel" in channels:
        dapi_key = "dapi_channel" if "dapi_channel" in channels else "dapi"
        normalized["dapi_channel"] = _validate_channel_value(
            "dapi",
            channels[dapi_key],
            allow_none=True,
        )

    markers = channels.get("markers", {})
    if not isinstance(markers, dict):
        raise ValueError(
            f"Channel config {path} has invalid 'channels.markers' value. Expected a mapping."
        )
    unknown_markers = sorted(set(markers) - set(PANEL_MARKER_NAMES))
    if unknown_markers:
        allowed = ", ".join(PANEL_MARKER_NAMES)
        raise ValueError(
            f"Channel config {path} has unknown marker keys: {', '.join(unknown_markers)}. "
            f"Allowed marker keys: {allowed}."
        )
    for marker_name, value in markers.items():
        normalized[f"{marker_name}_channel"] = _validate_channel_value(
            marker_name,
            value,
            allow_none=True,
        )

    return normalized


def _parse_canonical_panel_channels(path: Path, channels: dict[str, object]) -> dict[str, object]:
    unknown_keys = sorted(set(channels) - set(_CANONICAL_PANEL_CHANNEL_ALIASES))
    if unknown_keys:
        allowed = ", ".join(_CANONICAL_PANEL_CHANNEL_ALIASES)
        raise ValueError(
            f"Panel config {path} has unknown channel keys: {', '.join(unknown_keys)}. "
            f"Allowed keys: {allowed}."
        )
    if "laminin" not in channels:
        raise ValueError(f"Panel config {path} is missing required channel entry: laminin.")
    return {
        target: _validate_channel_value(name, value, allow_none=(name != "laminin"))
        for name, value in channels.items()
        for target in (_CANONICAL_PANEL_CHANNEL_ALIASES[name],)
    }


def load_channel_config(path: Path) -> ChannelConfig:
    """Load panel-aware channel mapping from a YAML config file."""
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"Channel config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in channel config {path}: {exc}") from exc

    if raw is None:
        raise ValueError(f"Channel config {path} is empty. Expected a 'channels' mapping.")
    if not isinstance(raw, dict):
        raise ValueError(f"Channel config {path} must be a YAML mapping at the top level.")

    channels = raw.get("channels")
    if channels is None:
        raise ValueError(f"Channel config {path} is missing required top-level key 'channels'.")
    if not isinstance(channels, dict):
        raise ValueError(f"Channel config {path} has invalid 'channels' value. Expected a mapping.")

    if "laminin" in channels:
        normalized = _parse_canonical_panel_channels(path, channels)
    elif "markers" in channels:
        normalized = _parse_nested_channels(path, channels)
    else:
        normalized = _parse_flat_channels(path, channels)

    if "membrane_channel" not in normalized:
        raise ValueError(f"Channel config {path} is missing required channel entry: membrane.")

    classification = raw.get("classification", {})
    if "classification" in raw and not isinstance(classification, dict):
        raise ValueError(
            f"Channel config {path} has invalid 'classification' value. Expected a mapping."
        )
    residual_inference = classification.get("residual_inference")
    enabled, target, requires = _validate_residual_config(path, residual_inference)
    normalized["residual_inference_enabled"] = enabled
    normalized["residual_target_class"] = target
    normalized["residual_requires_negative_markers"] = requires

    used_values = [
        value for key, value in normalized.items() if key.endswith("_channel") and value is not None
    ]
    if len(set(used_values)) != len(used_values):
        raise ValueError(
            f"Channel config {path} reuses channel indices across configured channels."
        )

    return ChannelConfig(
        membrane_channel=int(normalized["membrane_channel"]),
        dapi_channel=normalized.get("dapi_channel"),
        i_channel=normalized.get("i_channel"),
        iia_channel=normalized.get("iia_channel", DEFAULT_CHANNEL_CONFIG.iia_channel),
        iib_channel=normalized.get("iib_channel", DEFAULT_CHANNEL_CONFIG.iib_channel),
        iix_channel=normalized.get("iix_channel"),
        emhc_channel=normalized.get("emhc_channel"),
        residual_inference_enabled=bool(normalized["residual_inference_enabled"]),
        residual_target_class=normalized["residual_target_class"],
        residual_requires_negative_markers=tuple(normalized["residual_requires_negative_markers"]),
    )


def resolve_channel_config(
    *,
    channel_config_path: Path | None,
    i_channel: int | None = None,
    iia_channel: int | None = None,
    iib_channel: int | None = None,
    iix_channel: int | None = None,
    emhc_channel: int | None = None,
    dapi_channel: int | None = None,
    type1_channel: int | None,
    type2_channel: int | None,
    membrane_channel: int | None,
) -> tuple[ChannelConfig, list[str]]:
    """Resolve defaults, YAML config, explicit marker flags, and legacy CLI aliases."""
    resolved = DEFAULT_CHANNEL_CONFIG
    warnings: list[str] = []

    if channel_config_path is not None:
        resolved = load_channel_config(channel_config_path)

    modern_overrides = {
        "i_channel": i_channel,
        "iia_channel": iia_channel,
        "iib_channel": iib_channel,
        "iix_channel": iix_channel,
        "emhc_channel": emhc_channel,
        "dapi_channel": dapi_channel,
        "membrane_channel": membrane_channel,
    }
    legacy_overrides = {
        "iib_channel": type1_channel,
        "iia_channel": type2_channel,
    }
    resolved_values = {
        "membrane_channel": resolved.membrane_channel,
        "dapi_channel": resolved.dapi_channel,
        "i_channel": resolved.i_channel,
        "iia_channel": resolved.iia_channel,
        "iib_channel": resolved.iib_channel,
        "iix_channel": resolved.iix_channel,
        "emhc_channel": resolved.emhc_channel,
        "residual_inference_enabled": resolved.residual_inference_enabled,
        "residual_target_class": resolved.residual_target_class,
        "residual_requires_negative_markers": resolved.residual_requires_negative_markers,
    }

    modern_flag_names = {
        "i_channel": "i-channel",
        "iia_channel": "iia-channel",
        "iib_channel": "iib-channel",
        "iix_channel": "iix-channel",
        "emhc_channel": "emhc-channel",
        "dapi_channel": "dapi-channel",
        "membrane_channel": "membrane-channel",
    }
    for key, cli_value in modern_overrides.items():
        if cli_value is None:
            continue
        if channel_config_path is not None and cli_value != resolved_values[key]:
            warnings.append(
                f"--{modern_flag_names[key]}={cli_value} overrides "
                f"{channel_config_path}:{key}={resolved_values[key]}"
            )
        resolved_values[key] = cli_value

    legacy_flag_names = {
        "iib_channel": "type1-channel",
        "iia_channel": "type2-channel",
    }
    replacement_flag_names = {
        "iib_channel": "iib-channel",
        "iia_channel": "iia-channel",
    }
    for key, cli_value in legacy_overrides.items():
        if cli_value is None:
            continue
        if modern_overrides[key] is not None:
            warnings.append(
                f"--{legacy_flag_names[key]}={cli_value} ignored because "
                f"--{replacement_flag_names[key]}={modern_overrides[key]} was also provided"
            )
            continue
        if channel_config_path is not None and cli_value != resolved_values[key]:
            warnings.append(
                f"--{legacy_flag_names[key]}={cli_value} overrides "
                f"{channel_config_path}:{key}={resolved_values[key]}"
            )
        warnings.append(
            f"--{legacy_flag_names[key]} is a legacy alias for --{replacement_flag_names[key]}"
        )
        resolved_values[key] = cli_value

    return ChannelConfig(**resolved_values), warnings
