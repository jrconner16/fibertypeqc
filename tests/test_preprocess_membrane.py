import numpy as np

from src.preprocess_membrane import upsample_labels_nearest


def test_upsample_labels_nearest_restores_cellpose_resized_masks():
    labels = np.array([[1, 2], [3, 4]], dtype=np.int32)

    restored = upsample_labels_nearest(labels, target_shape=(6, 6), factor=1)

    assert restored.shape == (6, 6)
    assert restored[0, 0] == 1
    assert restored[0, -1] == 2
    assert restored[-1, 0] == 3
    assert restored[-1, -1] == 4
