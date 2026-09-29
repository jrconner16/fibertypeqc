from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
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


def test_tiff_payload_with_czi_suffix_uses_tiff_reader(tmp_path):
    path = tmp_path / "image_exported_as_czi.czi"
    image = np.zeros((2, 4, 5), dtype=np.uint16)
    image[1] = 7
    tifffile.imwrite(path, image, imagej=True, metadata={"axes": "CYX", "unit": "um"})

    loaded = load_multichannel_image(path)

    assert loaded.shape == image.shape
    assert np.array_equal(loaded, image)


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


def test_czi_axes_with_multiple_scenes_stop_with_split_instructions(tmp_path):
    from src.io_utils import _to_chw

    arr = np.zeros((1, 3, 4, 5, 6, 1), dtype=np.uint16)

    with pytest.raises(ValueError, match="contains 3 scenes.*split-czi-scenes"):
        _to_chw(arr, "BSCYX0", tmp_path / "slide.czi", czi=True)


def test_single_scene_czi_axes_reduce_to_channel_first(tmp_path):
    from src.io_utils import _to_chw

    arr = np.arange(1 * 1 * 4 * 5 * 6 * 1, dtype=np.uint16).reshape(1, 1, 4, 5, 6, 1)

    image = _to_chw(arr, "HSCYX0", tmp_path / "section.czi", czi=True)

    assert image.shape == (4, 5, 6)
    assert np.array_equal(image, arr.reshape(4, 5, 6))


def test_tiff_z_stack_is_rejected_instead_of_taking_first_plane(tmp_path):
    path = tmp_path / "stack.tif"
    tifffile.imwrite(
        path, np.zeros((3, 2, 16, 16), dtype=np.uint16), imagej=True, metadata={"axes": "ZCYX"}
    )

    with pytest.raises(ValueError, match="non-channel dimensions \\(Z=3\\)"):
        load_multichannel_image(path)


def test_unlabeled_large_third_axis_is_ambiguous(tmp_path):
    path = tmp_path / "unknown.tif"
    tifffile.imwrite(path, np.zeros((12, 16, 16), dtype=np.uint16))

    with pytest.raises(ValueError, match="cannot tell whether axis"):
        load_multichannel_image(path)
