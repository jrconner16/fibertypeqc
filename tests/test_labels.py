from __future__ import annotations

import numpy as np

from src.label_masks import erode_labels, eroded_label_mask


def test_erode_labels_preserves_label_ids_and_separates_objects():
    labels = np.zeros((7, 10), dtype=np.int32)
    labels[1:6, 1:5] = 1
    labels[1:6, 5:9] = 2

    eroded = erode_labels(labels, 1)

    assert set(np.unique(eroded)) == {0, 1, 2}
    assert np.all(eroded[:, 4:6] == 0)
    assert np.sum(eroded == 1) < np.sum(labels == 1)
    assert np.sum(eroded == 2) < np.sum(labels == 2)


def test_erode_labels_keeps_tiny_object_when_erosion_would_remove_it():
    labels = np.zeros((3, 3), dtype=np.int32)
    labels[1, 1] = 1

    eroded = erode_labels(labels, 2)

    assert np.array_equal(eroded, labels)


def test_eroded_label_mask_returns_boolean_foreground():
    labels = np.array([[0, 1], [2, 0]], dtype=np.int32)

    mask = eroded_label_mask(labels, 0)

    assert mask.dtype == bool
    assert mask.tolist() == [[False, True], [True, False]]
