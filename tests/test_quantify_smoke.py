from __future__ import annotations

import numpy as np
import pytest

from src.quantify_classify import (
    MarkerSpec,
    QCConfig,
    QuantifyConfig,
    class_stats_with_ci,
    qc_flags_from_fibers,
    quantify_labels,
)


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
    assert fibers["fiber_type_source"].tolist() == ["direct_marker", "direct_marker"]
    assert fibers["available_markers"].tolist() == ["iib|iia", "iib|iia"]
    assert fibers["area"].tolist() == [20, 20]
    assert fibers["area_um2"].tolist() == pytest.approx([2.5, 2.5])
    assert "area_erode_1px_um2" in fibers.columns
    assert fibers["typing_erode_px"].tolist() == [0, 0]


def test_quantify_labels_returns_expected_columns_for_empty_labels():
    labels = np.zeros((4, 4), dtype=np.int32)
    image = np.zeros((2, 4, 4), dtype=np.float32)

    fibers = quantify_labels(labels, image, QuantifyConfig(classifier_path=None))

    assert fibers.empty
    assert {
        "label",
        "fiber_type",
        "fiber_type_source",
        "available_markers",
        "needs_review",
        "classifier_path",
    }.issubset(fibers.columns)


def test_quantify_labels_collects_optional_marker_stats_without_changing_typing():
    labels = np.zeros((8, 12), dtype=np.int32)
    labels[1:6, 1:5] = 1
    labels[1:6, 7:11] = 2

    image = np.zeros((4, 8, 12), dtype=np.float32)
    image[0, labels == 1] = 10.0  # IIb / legacy type1
    image[1, labels == 2] = 12.0  # IIa / legacy type2
    image[2, labels == 1] = 3.0   # I marker present but unused for typing
    image[3, labels == 2] = 4.0   # IIx marker present but unused for typing

    fibers = quantify_labels(
        labels,
        image,
        QuantifyConfig(
            type1_channel=0,
            type2_channel=1,
            i_channel=2,
            iix_channel=3,
            threshold_mode="quantile",
            quantile=0.5,
            typing_preprocess="raw",
            typing_smooth_sigma=0.0,
            typing_erode_px=0,
            min_coverage=0.1,
            use_percentile_gate=False,
            classifier_path=None,
        ),
    ).sort_values("label").reset_index(drop=True)

    assert fibers["fiber_type"].tolist() == ["type1", "type2"]
    assert fibers["fiber_type_source"].tolist() == ["direct_marker", "direct_marker"]
    assert fibers["available_markers"].tolist() == ["iib|iia|i|iix", "iib|iia|i|iix"]
    assert fibers.attrs["available_markers"] == ("iib", "iia", "i", "iix")
    assert {"i", "iia", "iib", "iix"} == set(fibers.attrs["marker_stats"])
    assert fibers.attrs["marker_stats"]["i"]["mean"][0] == pytest.approx(3.0)
    assert fibers.attrs["marker_stats"]["iix"]["mean"][1] == pytest.approx(4.0)


def test_class_stats_can_canonicalize_legacy_labels():
    fibers = quantify_labels(
        np.pad(np.array([[1, 1], [0, 2]], dtype=np.int32), 1),
        np.pad(
            np.array(
                [
                    [[10.0, 10.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 12.0]],
                ],
                dtype=np.float32,
            ),
            ((0, 0), (1, 1), (1, 1)),
        ),
        QuantifyConfig(
            type1_channel=0,
            type2_channel=1,
            threshold_mode="quantile",
            quantile=0.5,
            typing_preprocess="raw",
            typing_erode_px=0,
            min_coverage=0.1,
            use_percentile_gate=False,
            classifier_path=None,
        ),
    )

    stats = class_stats_with_ci(
        fibers,
        classes=("iib", "iia", "mixed", "iix"),
        bootstrap_reps=0,
        canonicalize_labels=True,
    )

    assert stats["prop_iib"] == pytest.approx(0.5)
    assert stats["prop_iia"] == pytest.approx(0.5)
    assert stats["prop_iix"] == pytest.approx(0.0)


def test_qc_flags_can_resolve_marker_columns_from_specs():
    fibers = quantify_labels(
        np.pad(np.array([[1, 1], [0, 2]], dtype=np.int32), 1),
        np.pad(
            np.array(
                [
                    [[10.0, 10.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 12.0]],
                ],
                dtype=np.float32,
            ),
            ((0, 0), (1, 1), (1, 1)),
        ),
        QuantifyConfig(
            type1_channel=0,
            type2_channel=1,
            threshold_mode="quantile",
            quantile=0.5,
            typing_preprocess="raw",
            typing_erode_px=0,
            min_coverage=0.1,
            use_percentile_gate=False,
            classifier_path=None,
        ),
    )

    qc = qc_flags_from_fibers(
        fibers,
        QCConfig(
            min_labels=1,
            max_unknown_rate=1.0,
            median_area_min=0.0,
            median_area_max=100.0,
            max_type_corr=1.1,
        ),
        marker_specs=(
            MarkerSpec(marker_name="iib", legacy_prefix="type1", channel_index=0),
            MarkerSpec(marker_name="iia", legacy_prefix="type2", channel_index=1),
        ),
    )

    assert qc["qc_status"] == "pass"
