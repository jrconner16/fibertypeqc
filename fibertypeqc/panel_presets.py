"""Named panel presets and helpers for writing a panel file."""

from __future__ import annotations

from pathlib import Path

import yaml

PRESET_DIRECTORY = Path(__file__).resolve().parents[1] / "manifests" / "panels"
MARKERS = ("laminin", "type_i", "type_iia", "type_iib", "type_iix", "dapi", "emhc")
MARKER_LABELS = {
    "laminin": "laminin (fiber outlines)",
    "type_i": "Type I",
    "type_iia": "Type IIa",
    "type_iib": "Type IIb",
    "type_iix": "Type IIx (direct stain)",
    "dapi": "DAPI (nuclei)",
    "emhc": "eMHC",
}
RESIDUAL_MARKER_NAMES = {"type_i": "i", "type_iia": "iia", "type_iib": "iib"}


def list_presets() -> dict[str, tuple[Path, str]]:
    """Preset name -> (file, one-line description from its leading comment)."""
    presets: dict[str, tuple[Path, str]] = {}
    for path in sorted(PRESET_DIRECTORY.glob("*.yaml")):
        comment = [
            line.lstrip("# ").strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("#")
        ]
        presets[path.stem] = (path, " ".join(comment))
    return presets


def preset_path(name: str) -> Path | None:
    entry = list_presets().get(name)
    return entry[0] if entry else None


def panel_from_channels(channels: dict[str, int | None]) -> dict:
    """Build a panel mapping from marker -> channel index (None when not stained).

    When no direct IIx stain is present, IIx is inferred from the absence of whichever of
    Type I, IIa, and IIb are stained, provided at least IIa and IIb are present.
    """
    unknown = sorted(set(channels) - set(MARKERS))
    if unknown:
        raise ValueError(f"Unknown markers: {', '.join(unknown)}")
    if channels.get("laminin") is None:
        raise ValueError("A laminin channel is required for segmentation.")
    used = [index for index in channels.values() if index is not None]
    if len(used) != len(set(used)):
        raise ValueError("Two markers were given the same channel.")
    full = {marker: channels.get(marker) for marker in MARKERS}
    stained = [name for marker, name in RESIDUAL_MARKER_NAMES.items() if full[marker] is not None]
    infer_iix = full["type_iix"] is None and {"iia", "iib"} <= set(stained)
    return {
        "channels": {
            key: full[key]
            for key in ("laminin", "dapi", "type_i", "type_iia", "type_iib", "type_iix", "emhc")
        },
        "classification": {
            "residual_inference": {
                "enabled": infer_iix,
                "target_class": "iix" if infer_iix else None,
                "requires_negative_markers": stained if infer_iix else [],
            }
        },
    }


def write_panel(path: Path, panel: dict, comment: str = "") -> None:
    header = f"# {comment}\n" if comment else ""
    path.write_text(header + yaml.safe_dump(panel, sort_keys=False), encoding="utf-8")
