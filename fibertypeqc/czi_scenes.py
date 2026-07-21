"""Scene discovery and extraction for tiled, multi-scene Zeiss CZI files."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

try:
    import czifile
except Exception:  # pragma: no cover
    czifile = None


@dataclass(frozen=True)
class CziScene:
    index: int
    name: str
    y_start: int
    y_stop: int
    x_start: int
    x_stop: int


def _dimension_values(entry: object) -> dict[str, tuple[int, int]]:
    return {item.dimension: (int(item.start), int(item.size)) for item in entry.dimension_entries}


def discover_czi_scenes(path: Path) -> tuple[CziScene, ...]:
    """Discover true Zeiss scenes from native-resolution CZI tile metadata."""
    if czifile is None:
        raise ImportError("czifile is required for CZI scene extraction")
    with czifile.CziFile(str(path)) as czi:
        if "S" not in czi.axes:
            return ()
        names: dict[int, str] = {}
        root = ET.fromstring(czi.metadata())
        for scene in root.findall(".//Scene"):
            index = scene.attrib.get("Index")
            if index is not None:
                names[int(index)] = scene.attrib.get("Name", "")

        bounds: dict[int, list[int]] = {}
        for entry in czi.filtered_subblock_directory:
            values = _dimension_values(entry)
            if not {"S", "Y", "X"}.issubset(values):
                continue
            scene_index = values["S"][0]
            y_start, y_size = values["Y"]
            x_start, x_size = values["X"]
            current = bounds.setdefault(
                scene_index, [y_start, y_start + y_size, x_start, x_start + x_size]
            )
            current[0] = min(current[0], y_start)
            current[1] = max(current[1], y_start + y_size)
            current[2] = min(current[2], x_start)
            current[3] = max(current[3], x_start + x_size)
    return tuple(CziScene(index, names.get(index, ""), *bounds[index]) for index in sorted(bounds))


def extract_czi_scene(path: Path, scene: CziScene) -> np.ndarray:
    """Assemble one native-resolution scene as a scalar CHW array."""
    if czifile is None:
        raise ImportError("czifile is required for CZI scene extraction")
    with czifile.CziFile(str(path)) as czi:
        channel_axis = czi.axes.index("C")
        channel_count = int(czi.shape[channel_axis])
        image = np.zeros(
            (channel_count, scene.y_stop - scene.y_start, scene.x_stop - scene.x_start),
            dtype=czi.dtype,
        )
        for entry in czi.filtered_subblock_directory:
            values = _dimension_values(entry)
            if values.get("S", (-1, 0))[0] != scene.index or not {"C", "Y", "X"}.issubset(values):
                continue
            channel = values["C"][0]
            y_start, y_size = values["Y"]
            x_start, x_size = values["X"]
            tile = np.squeeze(entry.data_segment().data())
            image[
                channel,
                y_start - scene.y_start : y_start - scene.y_start + y_size,
                x_start - scene.x_start : x_start - scene.x_start + x_size,
            ] = tile
    return image


def export_czi_scenes(path: Path, output_dir: Path) -> list[Path]:
    """Export each native CZI scene as a scalar TIFF with deterministic section names."""
    scenes = discover_czi_scenes(path)
    if not scenes:
        raise ValueError("CZI input has no scene dimension to split.")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for order, scene in enumerate(scenes, start=1):
        output = output_dir / f"{path.stem}_section-{order:02d}.tif"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing scene export: {output}")
        tifffile.imwrite(output, extract_czi_scene(path, scene), metadata={"axes": "CYX"})
        outputs.append(output)
    return outputs
