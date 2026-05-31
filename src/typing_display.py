from __future__ import annotations

import numpy as np
import pandas as pd

from src.quantify_classify import _preprocess_typing_channel


def threshold_from_table(table: pd.DataFrame, column: str) -> float | None:
    if column not in table.columns:
        return None
    values = pd.to_numeric(table[column], errors="coerce").dropna()
    if values.empty:
        return None
    threshold = float(values.median())
    if threshold <= 0:
        return None
    return threshold


def normalize_for_display(
    channel: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.7,
) -> np.ndarray:
    x = channel.astype(np.float32, copy=False)
    lo, hi = np.percentile(x, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def optional_channel(image: np.ndarray, channel: int | None) -> np.ndarray | None:
    if image.ndim == 2 or channel is None:
        return None
    if channel < 0 or channel >= image.shape[0]:
        return None
    return image[channel]


def typing_signal_for_display(
    channel: np.ndarray,
    threshold: float | None,
    preprocess: str,
    bg_quantile: float,
    tile_size: int,
    bg_sigma: float,
    smooth_sigma: float,
    signal_scale: float,
    threshold_floor: float,
    mask: np.ndarray | None,
) -> np.ndarray:
    x = _preprocess_typing_channel(
        channel,
        mode=preprocess,
        bg_quantile=float(bg_quantile),
        tile_size=int(tile_size),
        bg_sigma=float(bg_sigma),
        smooth_sigma=float(smooth_sigma),
    )
    if threshold is None or threshold <= 0:
        out = normalize_for_display(x)
        if mask is not None:
            out = out * mask
        return out

    floor = float(threshold) * float(threshold_floor)
    ceiling = float(threshold) * float(signal_scale)
    denom = max(ceiling - floor, 1e-6)
    out = np.clip((x - floor) / denom, 0, 1).astype(np.float32)
    if mask is not None:
        out = out * mask
    return out


def typing_composite(
    image: np.ndarray,
    type1_channel: int | None,
    type2_channel: int | None,
    membrane_channel: int | None,
    type1_threshold: float | None,
    type2_threshold: float | None,
    preprocess: str,
    bg_quantile: float,
    tile_size: int,
    bg_sigma: float,
    smooth_sigma: float,
    signal_scale: float,
    threshold_floor: float,
    signal_mask: np.ndarray | None,
) -> np.ndarray:
    base = image if image.ndim == 2 else image[0]
    h, w = base.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    membrane = optional_channel(image, membrane_channel)
    if membrane is not None:
        m = normalize_for_display(membrane)
        rgb += 0.35 * np.stack([m, m, m], axis=-1)

    type1 = optional_channel(image, type1_channel)
    if type1 is not None:
        t1 = typing_signal_for_display(
            type1,
            threshold=type1_threshold,
            preprocess=preprocess,
            bg_quantile=bg_quantile,
            tile_size=tile_size,
            bg_sigma=bg_sigma,
            smooth_sigma=smooth_sigma,
            signal_scale=signal_scale,
            threshold_floor=threshold_floor,
            mask=signal_mask,
        )
        rgb[..., 0] += t1
        rgb[..., 2] += 0.85 * t1

    type2 = optional_channel(image, type2_channel)
    if type2 is not None:
        t2 = typing_signal_for_display(
            type2,
            threshold=type2_threshold,
            preprocess=preprocess,
            bg_quantile=bg_quantile,
            tile_size=tile_size,
            bg_sigma=bg_sigma,
            smooth_sigma=smooth_sigma,
            signal_scale=signal_scale,
            threshold_floor=threshold_floor,
            mask=signal_mask,
        )
        rgb[..., 1] += t2

    return np.clip(rgb, 0, 1)
