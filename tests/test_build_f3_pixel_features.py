from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.build_f3_pixel_features import extract_image_f3, f3_columns, resolve_source_image


def test_extract_image_f3_is_finite_and_uses_requested_labels():
    labels = np.zeros((11, 14), dtype=np.int32)
    labels[1:6, 1:6] = 1
    labels[2:9, 8:13] = 2
    image = np.zeros((4, 11, 14), dtype=np.float32)
    image[1] = np.arange(154, dtype=np.float32).reshape(11, 14)
    image[3] = image[1, ::-1]

    result = extract_image_f3(
        labels=labels,
        image_chw=image,
        marker_channels={"iia": 1, "iib": 3},
        label_ids=np.array([2, 1]),
        typing_smooth_sigma=0.0,
    )

    assert result["label"].tolist() == [2, 1]
    assert set(f3_columns()).issubset(result.columns)
    assert np.isfinite(result.loc[:, f3_columns()].to_numpy()).all()


def test_extract_image_f3_rejects_missing_requested_label():
    labels = np.zeros((7, 7), dtype=np.int32)
    labels[1:6, 1:6] = 1
    image = np.ones((4, 7, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="absent"):
        extract_image_f3(
            labels=labels,
            image_chw=image,
            marker_channels={"iia": 1, "iib": 3},
            label_ids=np.array([2]),
        )


def test_resolve_source_image_uses_explicit_recorded_root():
    result = resolve_source_image(
        Path("/cluster/input/cohort/mouse/image.czi"),
        recorded_source_root=Path("/cluster/input/cohort"),
        source_image_root=Path("/local/private/cohort"),
    )

    assert result == Path("/local/private/cohort/mouse/image.czi")


def test_resolve_source_image_rejects_path_outside_recorded_root():
    with pytest.raises(ValueError, match="outside --recorded-source-root"):
        resolve_source_image(
            Path("/different/root/image.czi"),
            recorded_source_root=Path("/cluster/input/cohort"),
            source_image_root=Path("/local/private/cohort"),
        )
