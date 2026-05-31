from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, find_objects


def erode_labels(labels: np.ndarray, erode_px: int) -> np.ndarray:
    """Erode each labeled object independently, preserving tiny objects as a fallback."""
    labels = np.asarray(labels)
    if erode_px <= 0:
        return labels.astype(np.int32, copy=True)

    out = np.zeros_like(labels, dtype=np.int32)
    object_slices = find_objects(labels)
    for label_index, slc in enumerate(object_slices, start=1):
        if slc is None:
            continue
        local = labels[slc] == label_index
        if not np.any(local):
            continue
        eroded = binary_erosion(local, iterations=int(erode_px), border_value=0)
        if not np.any(eroded):
            eroded = local
        out_local = out[slc]
        out_local[eroded] = label_index
    return out


def eroded_label_mask(labels: np.ndarray, erode_px: int) -> np.ndarray:
    return erode_labels(labels, erode_px) > 0
