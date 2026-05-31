from __future__ import annotations

import numpy as np
import pytest

from src.quantify_classify import QuantifyConfig, quantify_labels


def test_quantify_labels_smoke_with_synthetic_channels():
    labels = np.zeros((8, 12), dtype=np.int32)
    labels[1:6, 1:5] = 1
    labels[1:6, 7:11] = 2

    image = np.zeros((2, 8, 12), dtype=np.float32)
    image[0, labels == 1] = 10.0
    image[1, labels == 2] = 12.0

    cfg = QuantifyConfig(
        type1_channel=0,
        type2_channel=1,
        threshold_mode="quantile",
        quantile=0.5,
        typing_preprocess="raw",
        typing_erode_px=0,
        min_coverage=0.1,
        use_percentile_gate=False,
        classifier_path=None,
        pixel_size_x_um=0.5,
        pixel_size_y_um=0.25,
        csa_erode_px=(1,),
    )

    fibers = quantify_labels(labels, image, cfg).sort_values("label").reset_index(drop=True)

    assert fibers["label"].tolist() == [1, 2]
    assert fibers["fiber_type"].tolist() == ["type1", "type2"]
    assert fibers["area"].tolist() == [20, 20]
    assert fibers["area_um2"].tolist() == pytest.approx([2.5, 2.5])
    assert "area_erode_1px_um2" in fibers.columns
    assert fibers["typing_erode_px"].tolist() == [0, 0]


def test_quantify_labels_returns_expected_columns_for_empty_labels():
    labels = np.zeros((4, 4), dtype=np.int32)
    image = np.zeros((2, 4, 4), dtype=np.float32)

    fibers = quantify_labels(labels, image, QuantifyConfig(classifier_path=None))

    assert fibers.empty
    assert {"label", "fiber_type", "needs_review", "classifier_path"}.issubset(fibers.columns)
