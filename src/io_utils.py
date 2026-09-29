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


MAX_CHANNELS = 8
SCENE_SPLIT_HINT = (
    "Split it first with `run_batch --split-czi-scenes` or "
    "`python -m src.split_czi_scenes --input FILE --output-dir DIR`, then run each section."
)


def _to_chw(arr: np.ndarray, axes: str, path: Path, *, czi: bool) -> np.ndarray:
    """Reduce a microscopy array with known axes to channel-first ``(C, Y, X)``.

    Singleton axes are dropped. Anything that cannot be mapped unambiguously (several scenes,
    Z/T stacks, mosaic tiles, or two channel-like axes) raises instead of silently selecting
    a plane.
    """
    axes = axes.upper()
    if len(axes) != arr.ndim:
        raise ValueError(f"{path.name}: axes {axes!r} do not match array shape {arr.shape}")
    keep = [i for i, size in enumerate(arr.shape) if size > 1 or axes[i] in "YX"]
    arr = arr.reshape(tuple(arr.shape[i] for i in keep))
    axes = "".join(axes[i] for i in keep)

    if czi and "S" in axes:
        n_scenes = arr.shape[axes.index("S")]
        raise ValueError(f"{path.name} contains {n_scenes} scenes. {SCENE_SPLIT_HINT}")
    # CZI stores RGB samples as '0'; TIFF stores them as 'S'.
    channel_like = "C0" if czi else "CS"
    channel_axes = [i for i, name in enumerate(axes) if name in channel_like]
    unknown_axes = [i for i, name in enumerate(axes) if name not in channel_like + "YX"]
    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"{path.name}: expected Y and X axes, found {axes!r}")
    if len(channel_axes) > 1:
        raise ValueError(f"{path.name}: more than one channel-like axis in {axes!r}")
    if unknown_axes and not channel_axes and len(unknown_axes) == 1:
        # An unlabeled third axis (for example plain TIFF 'Q'/'I') is the channel axis only when
        # it is small enough to be one.
        candidate = unknown_axes[0]
        if arr.shape[candidate] > MAX_CHANNELS:
            raise ValueError(
                f"{path.name}: cannot tell whether axis {axes[candidate]!r} of size "
                f"{arr.shape[candidate]} is channels or a stack; save the image with channel "
                "metadata (e.g. ImageJ axes 'CYX')."
            )
        channel_axes, unknown_axes = [candidate], []
    if unknown_axes:
        described = ", ".join(f"{axes[i]}={arr.shape[i]}" for i in unknown_axes)
        raise ValueError(
            f"{path.name} has non-channel dimensions ({described}). Project or split the stack "
            "into single-plane multichannel images before running."
        )

    order = channel_axes + [axes.index("Y"), axes.index("X")]
    arr = np.transpose(arr, order)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    return arr


def load_multichannel_image(path: Path) -> np.ndarray:
    """Load CZI/TIFF into a ``(C, Y, X)`` array using the file's axis metadata."""
    suffix = path.suffix.lower()
    # Some ImageJ exports retain a source-image `.czi` suffix even though their
    # bytes are TIFF. Detect their actual container without renaming raw data.
    if suffix in {".tif", ".tiff"} or _has_tiff_signature(path):
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            arr = np.asarray(series.asarray())
            axes = series.axes
        return _to_chw(arr, axes, path, czi=False)
    if suffix == ".czi":
        if czifile is None:
            raise ImportError("czifile is required for .czi input")
        with czifile.CziFile(str(path)) as czi:
            # Multi-threaded assembly writes overlapping mosaic tiles in nondeterministic order,
            # so repeated reads of the same file can differ. One worker assembles tiles in
            # subblock-directory order, matching the scene splitter.
            arr = np.asarray(czi.asarray(max_workers=1))
            axes = czi.axes
        return _to_chw(arr, axes, path, czi=True)
    raise ValueError(f"Unsupported input type: {path.suffix}")


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
