from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk, remove_small_objects


@dataclass
class PreprocessConfig:
    crop_auto: bool = True
    crop_ds: int = 8
    crop_pad: int = 128
    crop_min_size: int = 2000
    crop_min_fraction: float = 0.20
    downsample_factor: int = 6

    # Membrane-only adaptive preprocessing.
    bg_sigma: float = 30.0
    smooth_sigma: float = 1.0
    p_low: float = 1.0
    p_high: float = 99.8
    noise_floor: float = 0.05


@dataclass
class PreprocessResult:
    membrane_full: np.ndarray
    membrane_crop: np.ndarray
    membrane_model_input: np.ndarray
    crop_slices: tuple[slice, slice]


def contrast_stretch(channel: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    ch = np.asarray(channel, dtype=np.float32)
    lo, hi = np.percentile(ch, [low_pct, high_pct])
    if hi <= lo:
        return np.zeros_like(ch, dtype=np.float32)
    return np.clip((ch - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def auto_tissue_slices(
    image_2d: np.ndarray,
    ds: int = 8,
    pad: int = 128,
    min_size: int = 2000,
) -> tuple[slice, slice]:
    h, w = image_2d.shape
    low = image_2d[::ds, ::ds].astype(np.float32)

    nz = low[low > 0]
    if nz.size == 0:
        return (slice(0, h), slice(0, w))

    try:
        thr = float(threshold_otsu(nz))
    except Exception:
        thr = float(np.percentile(nz, 75))

    mask = low > thr
    mask = closing(mask, footprint=disk(3))
    mask = remove_small_objects(mask, max_size=max(0, int(min_size) - 1))

    ys, xs = np.where(mask)
    if ys.size == 0:
        fallback_thr = float(np.percentile(nz, 90))
        ys, xs = np.where(low > fallback_thr)
        if ys.size == 0:
            return (slice(0, h), slice(0, w))

    y0 = max(0, int(ys.min() * ds) - pad)
    y1 = min(h, int((ys.max() + 1) * ds) + pad)
    x0 = max(0, int(xs.min() * ds) - pad)
    x1 = min(w, int((xs.max() + 1) * ds) + pad)
    return (slice(y0, y1), slice(x0, x1))


def preprocess_membrane_channel(membrane: np.ndarray, cfg: PreprocessConfig) -> PreprocessResult:
    full_slices = (slice(0, membrane.shape[0]), slice(0, membrane.shape[1]))

    if cfg.crop_auto:
        crop_slices = auto_tissue_slices(
            membrane,
            ds=cfg.crop_ds,
            pad=cfg.crop_pad,
            min_size=cfg.crop_min_size,
        )
        ys, xs = crop_slices
        crop_h = max(1, int(ys.stop - ys.start))
        crop_w = max(1, int(xs.stop - xs.start))
        frac = (crop_h * crop_w) / float(membrane.shape[0] * membrane.shape[1])
        touches_edge = (
            ys.start <= 0
            or xs.start <= 0
            or ys.stop >= membrane.shape[0]
            or xs.stop >= membrane.shape[1]
        )
        if touches_edge and frac < cfg.crop_min_fraction:
            crop_slices = full_slices
    else:
        crop_slices = full_slices

    membrane_crop = membrane[crop_slices].astype(np.float32)

    bg = gaussian_filter(membrane_crop, sigma=cfg.bg_sigma)
    x = np.clip(membrane_crop - bg, 0, None)
    x = gaussian_filter(x, sigma=cfg.smooth_sigma)
    x = contrast_stretch(x, cfg.p_low, cfg.p_high)
    if cfg.noise_floor > 0:
        x[x < cfg.noise_floor] = 0.0

    if cfg.downsample_factor > 1:
        x_model = x[:: cfg.downsample_factor, :: cfg.downsample_factor]
    else:
        x_model = x

    return PreprocessResult(
        membrane_full=membrane.astype(np.float32),
        membrane_crop=membrane_crop,
        membrane_model_input=x_model.astype(np.float32),
        crop_slices=crop_slices,
    )


def upsample_labels_nearest(
    labels: np.ndarray, target_shape: tuple[int, int], factor: int
) -> np.ndarray:
    if factor <= 1:
        return labels.astype(np.int32)
    up = np.repeat(np.repeat(labels, factor, axis=0), factor, axis=1)
    return up[: target_shape[0], : target_shape[1]].astype(np.int32)


def paste_crop_labels(
    crop_labels: np.ndarray,
    full_shape: tuple[int, int],
    crop_slices: tuple[slice, slice],
) -> np.ndarray:
    full = np.zeros(full_shape, dtype=np.int32)
    full[crop_slices] = crop_labels.astype(np.int32)
    return full
