from __future__ import annotations

import numpy as np
import pytest

from src.review_project_napari import (
    downsample_review_data,
    selected_fiber_outline,
    selected_fiber_outline_rgba,
)


def test_display_downsample_preserves_label_ids_and_alignment() -> None:
    raw = np.arange(2 * 4 * 6, dtype=np.float32).reshape(2, 4, 6)
    labels = np.arange(4 * 6, dtype=np.int32).reshape(4, 6)

    display_raw, display_labels = downsample_review_data(raw, labels, 2)

    assert display_raw.shape == (2, 2, 3)
    assert np.array_equal(display_raw, raw[:, ::2, ::2])
    assert np.array_equal(display_labels, labels[::2, ::2])


def test_display_downsample_rejects_invalid_factor_or_alignment() -> None:
    raw = np.zeros((2, 4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="at least 1"):
        downsample_review_data(raw, None, 0)
    with pytest.raises(ValueError, match="does not match"):
        downsample_review_data(raw, np.zeros((3, 3), dtype=np.int32), 2)


def test_selected_fiber_is_outline_only() -> None:
    labels = np.zeros((7, 7), dtype=np.int32)
    labels[2:5, 2:5] = 4

    outline = selected_fiber_outline(labels, 4)

    assert outline[3, 3] == 0
    assert outline.sum() > 0

    rgba = selected_fiber_outline_rgba(labels, 4)
    assert rgba.shape == (7, 7, 4)
    assert rgba[3, 3, 3] == 0
    assert rgba[..., 1].max() == 1
    assert rgba[..., 2].max() == 1
