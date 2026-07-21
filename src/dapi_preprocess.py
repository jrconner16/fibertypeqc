"""Small, explicit preprocessing options for DAPI nuclear segmentation."""

from __future__ import annotations

import numpy as np


def tile_background_subtract(
    image: np.ndarray,
    *,
    tile_size: int = 512,
    background_quantile: float = 0.02,
) -> np.ndarray:
    """Subtract a per-tile low background estimate while preserving local contrast."""
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"DAPI preprocessing expects a 2D image, got {values.shape}")
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    if not 0.0 <= background_quantile < 1.0:
        raise ValueError("background_quantile must be in [0, 1)")
    output = np.empty_like(values, dtype=np.float32)
    height, width = values.shape
    for y0 in range(0, height, tile_size):
        for x0 in range(0, width, tile_size):
            y1 = min(y0 + tile_size, height)
            x1 = min(x0 + tile_size, width)
            tile = values[y0:y1, x0:x1]
            background = float(np.quantile(tile, background_quantile))
            output[y0:y1, x0:x1] = np.maximum(tile - background, 0.0)
    return output


def tile_percentile_normalize(
    image: np.ndarray,
    *,
    tile_size: int = 512,
    low_percentile: float = 1.0,
    high_percentile: float = 99.8,
) -> np.ndarray:
    """Normalize each tile independently to improve weak local contrast."""
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"DAPI preprocessing expects a 2D image, got {values.shape}")
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    output = np.empty_like(values, dtype=np.float32)
    height, width = values.shape
    for y0 in range(0, height, tile_size):
        for x0 in range(0, width, tile_size):
            y1 = min(y0 + tile_size, height)
            x1 = min(x0 + tile_size, width)
            tile = values[y0:y1, x0:x1]
            low = float(np.percentile(tile, low_percentile))
            high = float(np.percentile(tile, high_percentile))
            output[y0:y1, x0:x1] = np.clip((tile - low) / max(high - low, 1e-6), 0.0, 1.0)
    return output
