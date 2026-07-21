from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

try:
    import czifile
except Exception:  # pragma: no cover
    czifile = None


def _has_tiff_signature(path: Path) -> bool:
    """Return whether a file is TIFF-formatted, regardless of its suffix."""
    with path.open("rb") as handle:
        signature = handle.read(4)
    return signature in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_multichannel_image(path: Path) -> np.ndarray:
    """Load CZI/TIFF into CHW array."""
    suffix = path.suffix.lower()
    # Some ImageJ exports retain a source-image `.czi` suffix even though their
    # bytes are TIFF. Detect their actual container without renaming raw data.
    if suffix in {".tif", ".tiff"} or _has_tiff_signature(path):
        arr = np.asarray(tifffile.imread(path))
    elif suffix == ".czi":
        if czifile is None:
            raise ImportError("czifile is required for .czi input")
        with czifile.CziFile(str(path)) as czi:
            arr = np.asarray(czi.asarray())
        arr = np.squeeze(arr)
    else:
        raise ValueError(f"Unsupported input type: {path.suffix}")

    while arr.ndim > 3:
        arr = arr[0]

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 3:
        # Infer channel axis for microscopy-ish arrays.
        if arr.shape[0] <= 8:
            pass
        elif arr.shape[-1] <= 8:
            arr = np.moveaxis(arr, -1, 0)
        elif arr.shape[1] <= 8:
            arr = np.moveaxis(arr, 1, 0)
        else:
            raise ValueError(f"Could not infer channel axis for shape {arr.shape}")
    else:
        raise ValueError(f"Expected 2D/3D image, got shape {arr.shape}")

    return arr


def extract_pixel_size_um(path: Path) -> tuple[float | None, float | None]:
    """Return physical pixel size as (x_um, y_um) when available."""
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"} or _has_tiff_signature(path):
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            x_res = page.tags.get("XResolution")
            y_res = page.tags.get("YResolution")
            unit = page.tags.get("ResolutionUnit")
            if x_res is None or y_res is None or unit is None:
                return None, None
            x_pixels_per_unit = float(x_res.value[0]) / float(x_res.value[1])
            y_pixels_per_unit = float(y_res.value[0]) / float(y_res.value[1])
            imagej_unit = str((tif.imagej_metadata or {}).get("unit", "")).strip().lower()
            if imagej_unit in {"micron", "micrometer", "micrometers", "um", "µm"}:
                return 1.0 / x_pixels_per_unit, 1.0 / y_pixels_per_unit
            unit_name = str(unit.value).upper()
            if unit_name not in {"CENTIMETER", "INCH"}:
                return None, None
            unit_um = 10_000.0 if unit_name == "CENTIMETER" else 25_400.0
            return unit_um / x_pixels_per_unit, unit_um / y_pixels_per_unit

    if suffix == ".czi":
        if czifile is None:
            return None, None
        with czifile.CziFile(str(path)) as czi:
            metadata = czi.metadata()
        root = ET.fromstring(metadata)
        values: dict[str, float] = {}
        for distance in root.findall(".//Scaling/Items/Distance"):
            axis = str(distance.attrib.get("Id", "")).upper()
            value = distance.findtext("Value")
            if axis not in {"X", "Y"} or value is None:
                continue
            # Zeiss CZI stores these distances in meters even when the display unit is um.
            values[axis] = float(value) * 1_000_000.0
        return values.get("X"), values.get("Y")

    return None, None


def save_labels(path: Path, labels: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, labels.astype(np.int32))
    return path


def save_dataframe(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def label_summary(labels: np.ndarray) -> dict[str, float | int]:
    areas = np.bincount(labels.ravel())[1:]
    if len(areas) == 0:
        return {
            "n_labels": 0,
            "area_median": 0.0,
            "area_min": 0.0,
            "area_max": 0.0,
        }
    return {
        "n_labels": int((areas > 0).sum()),
        "area_median": float(np.median(areas)),
        "area_min": float(np.min(areas)),
        "area_max": float(np.max(areas)),
    }


def parse_channels(channels: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(channels, str):
        return tuple(int(x.strip()) for x in channels.split(",") if x.strip())
    return tuple(int(c) for c in channels)
