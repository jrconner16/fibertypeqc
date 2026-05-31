from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile

from src.io_utils import (
    extract_pixel_size_um,
    label_summary,
    load_multichannel_image,
    save_dataframe,
)


def test_load_multichannel_tiff_moves_channel_axis(tmp_path):
    path = tmp_path / "image.tif"
    image_hwc = np.zeros((9, 10, 3), dtype=np.uint16)
    image_hwc[..., 0] = 1
    image_hwc[..., 1] = 2
    image_hwc[..., 2] = 3
    tifffile.imwrite(path, image_hwc)

    image = load_multichannel_image(path)

    assert image.shape == (3, 9, 10)
    assert np.all(image[0] == 1)
    assert np.all(image[1] == 2)
    assert np.all(image[2] == 3)


def test_extract_pixel_size_um_from_imagej_tiff(tmp_path):
    path = tmp_path / "calibrated.tif"
    tifffile.imwrite(
        path,
        np.zeros((4, 4), dtype=np.uint8),
        imagej=True,
        resolution=(2.0, 4.0),
        metadata={"unit": "um"},
    )

    x_um, y_um = extract_pixel_size_um(path)

    assert x_um == 0.5
    assert y_um == 0.25


def test_save_dataframe_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "table.csv"

    save_dataframe(path, pd.DataFrame({"a": [1, 2]}))

    assert path.exists()
    assert pd.read_csv(path)["a"].tolist() == [1, 2]


def test_label_summary_counts_nonzero_labels():
    labels = np.array(
        [
            [0, 1, 1],
            [0, 2, 2],
            [3, 3, 3],
        ],
        dtype=np.int32,
    )

    summary = label_summary(labels)

    assert summary["n_labels"] == 3
    assert summary["area_min"] == 2
    assert summary["area_median"] == 2
    assert summary["area_max"] == 3
