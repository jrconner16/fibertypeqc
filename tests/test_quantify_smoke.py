from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

import src.quantify_classify as quantify_classify
from src.quantify_classify import (
    FROZEN_ALPHA_BASELINE_FEATURES,
    MarkerSpec,
    QCConfig,
    QuantifyConfig,
    build_feature_diagnostics_table,
    build_feature_table,
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
    assert fibers["feret_max_px"].tolist() == pytest.approx([5.830952, 5.830952], rel=1e-5)
    assert fibers["feret_min_px"].tolist() == pytest.approx([4.0, 4.0], rel=1e-5)
    assert fibers["feret_max_um"].tolist() == pytest.approx([2.236068, 2.236068], rel=1e-5)
    assert fibers["feret_min_um"].tolist() == pytest.approx([1.25, 1.25], rel=1e-5)
    assert "area_erode_1px_um2" in fibers.columns
    assert fibers["typing_erode_px"].tolist() == [0, 0]


def test_quantify_labels_reuses_isotropic_feret_measurements(monkeypatch):
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[1:6, 1:5] = 1
    image = np.zeros((2, 8, 8), dtype=np.float32)
    calls = 0
    original = quantify_classify._feret_diameters_by_label

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(quantify_classify, "_feret_diameters_by_label", counted)
    fibers = quantify_labels(
        labels,
        image,
        QuantifyConfig(
            typing_preprocess="raw",
            typing_erode_px=0,
            pixel_size_x_um=0.5,
            pixel_size_y_um=0.5,
        ),
    )

    assert calls == 1
    assert fibers["feret_max_um"].iloc[0] == pytest.approx(fibers["feret_max_px"].iloc[0] * 0.5)


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
        "feret_max_px",
        "feret_min_px",
    }.issubset(fibers.columns)


def test_quantify_labels_collects_optional_marker_stats_without_changing_typing():
    labels = np.zeros((8, 12), dtype=np.int32)
    labels[1:6, 1:5] = 1
    labels[1:6, 7:11] = 2

    image = np.zeros((4, 8, 12), dtype=np.float32)
    image[0, labels == 1] = 10.0  # IIb / legacy type1
    image[1, labels == 2] = 12.0  # IIa / legacy type2
    image[2, labels == 1] = 3.0  # I marker present but unused for typing
    image[3, labels == 2] = 4.0  # IIx marker present but unused for typing

    fibers = (
        quantify_labels(
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
                collect_spatial_marker_features=True,
                spatial_feature_erode_px=1,
                min_coverage=0.1,
                use_percentile_gate=False,
                classifier_path=None,
            ),
        )
        .sort_values("label")
        .reset_index(drop=True)
    )

    assert fibers["fiber_type"].tolist() == ["type1", "type2"]
    assert fibers["fiber_type_source"].tolist() == ["direct_marker", "direct_marker"]
    assert fibers["available_markers"].tolist() == ["iib|iia|i|iix", "iib|iia|i|iix"]
    assert fibers.attrs["available_markers"] == ("iib", "iia", "i", "iix")
    assert {"i", "iia", "iib", "iix"} == set(fibers.attrs["marker_stats"])
    assert fibers.attrs["marker_stats"]["i"]["mean"][0] == pytest.approx(3.0)
    assert fibers.attrs["marker_stats"]["iix"]["mean"][1] == pytest.approx(4.0)

    diagnostics = build_feature_diagnostics_table(fibers, QuantifyConfig())
    assert diagnostics["feature_schema_version"].tolist() == [
        "multiplanel_features.v1",
        "multiplanel_features.v1",
    ]
    assert {
        "type_i.mean",
        "type_i.p90",
        "type_i.coverage_high",
        "type_i.snr_mean",
        "type_i.center_mean",
        "type_i.edge_mean",
        "type_iia.mean",
        "type_iib.mean",
    }.issubset(diagnostics.columns)


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


def test_build_feature_table_preserves_frozen_baseline_and_adds_experimental_columns():
    df = pd.DataFrame(
        {
            "area": [10, 20],
            "feret_max_px": [4.0, 5.0],
            "feret_min_px": [3.0, 4.0],
            "feret_max_um": [2.0, 2.5],
            "feret_min_um": [1.5, 2.0],
            "type1_mean": [1.0, 2.0],
            "type2_mean": [3.0, 4.0],
            "type1_p75": [1.5, 2.5],
            "type2_p75": [3.5, 4.5],
            "type1_p90": [1.8, 2.8],
            "type2_p90": [3.8, 4.8],
            "type1_pctl": [1.7, 2.7],
            "type2_pctl": [3.7, 4.7],
            "type1_coverage": [0.2, 0.3],
            "type2_coverage": [0.4, 0.5],
        }
    )
    feats = build_feature_table(
        df,
        marker_specs=(
            MarkerSpec(marker_name="iib", legacy_prefix="type1", channel_index=0),
            MarkerSpec(marker_name="iia", legacy_prefix="type2", channel_index=1),
        ),
        marker_stats_metadata={
            "iib": {
                "mean": np.array([1.0, 2.0], dtype=np.float32),
                "p75": np.array([1.5, 2.5], dtype=np.float32),
                "p90": np.array([1.8, 2.8], dtype=np.float32),
                "pctl": np.array([1.7, 2.7], dtype=np.float32),
                "coverage": np.array([0.2, 0.3], dtype=np.float32),
                "tissue_median": 0.5,
                "tissue_mad": 0.25,
            },
            "iia": {
                "mean": np.array([3.0, 4.0], dtype=np.float32),
                "p75": np.array([3.5, 4.5], dtype=np.float32),
                "p90": np.array([3.8, 4.8], dtype=np.float32),
                "pctl": np.array([3.7, 4.7], dtype=np.float32),
                "coverage": np.array([0.4, 0.5], dtype=np.float32),
                "tissue_median": 1.0,
                "tissue_mad": 0.5,
            },
            "i": {
                "mean": np.array([5.0, 6.0], dtype=np.float32),
                "p75": np.array([5.5, 6.5], dtype=np.float32),
                "p90": np.array([5.8, 6.8], dtype=np.float32),
                "pctl": np.array([5.7, 6.7], dtype=np.float32),
                "coverage": np.array([0.1, 0.2], dtype=np.float32),
                "tissue_median": 0.8,
                "tissue_mad": 0.4,
            },
        },
    )
    assert {"feret_max_px", "feret_min_px", "feret_max_um", "feret_min_um"}.issubset(feats.columns)

    assert set(FROZEN_ALPHA_BASELINE_FEATURES).issubset(feats.columns)
    assert {"type_cov_sum", "type1_snr_mean", "marker_i_mean", "marker_i_snr_mean"}.issubset(
        feats.columns
    )


@pytest.mark.integration
def test_frozen_alpha_model_feature_contract_matches_code_baseline():
    model = joblib.load("data/models/rebaseline_tile_v2_p75p90_iib_iia_iix.joblib")
    assert tuple(model.feature_names_in_) == FROZEN_ALPHA_BASELINE_FEATURES


def test_legacy_rule_path_snapshot_for_frozen_alpha_regression():
    labels = np.zeros((20, 30), dtype=np.int32)
    labels[2:10, 2:10] = 1
    labels[2:10, 12:20] = 2
    labels[2:10, 22:28] = 3

    image = np.zeros((2, 20, 30), dtype=np.float32)
    image[0, labels == 1] = 12.0
    image[1, labels == 2] = 11.0
    image[0, labels == 3] = 1.0
    image[1, labels == 3] = 1.0

    fibers = (
        quantify_labels(
            labels,
            image,
            QuantifyConfig(classifier_path=None),
        )
        .sort_values("label")
        .reset_index(drop=True)
    )

    observed = fibers[
        [
            "label",
            "fiber_type",
            "classification_method",
            "type1_mean",
            "type2_mean",
            "type1_p75",
            "type2_p75",
            "type1_coverage",
            "type2_coverage",
            "type1_threshold",
            "type2_threshold",
            "confidence",
            "needs_review",
        ]
    ].assign(
        type1_mean=lambda df: df["type1_mean"].round(4),
        type2_mean=lambda df: df["type2_mean"].round(4),
        type1_p75=lambda df: df["type1_p75"].round(4),
        type2_p75=lambda df: df["type2_p75"].round(4),
        type1_coverage=lambda df: df["type1_coverage"].round(4),
        type2_coverage=lambda df: df["type2_coverage"].round(4),
        type1_threshold=lambda df: df["type1_threshold"].round(4),
        type2_threshold=lambda df: df["type2_threshold"].round(4),
        confidence=lambda df: df["confidence"].round(4),
    )

    expected = pd.DataFrame(
        {
            "label": [1, 2, 3],
            "fiber_type": ["type1", "type2", "unknown"],
            "classification_method": ["rules:quantile+p85+cov6+mx25"] * 3,
            "type1_mean": [11.9947, 0.0, 0.9993],
            "type2_mean": [0.0, 10.9952, 0.9993],
            "type1_p75": [11.9960, 0.0, 0.9996],
            "type2_p75": [0.0, 10.9964, 0.9996],
            "type1_coverage": [0.75, 0.0, 0.0],
            "type2_coverage": [0.0, 0.75, 0.0],
            "type1_threshold": [3.1984, 3.1984, 3.1984],
            "type2_threshold": [2.9985, 2.9985, 2.9985],
            "confidence": [1.0, 1.0, 0.0062],
            "needs_review": [False, False, True],
        }
    )

    pd.testing.assert_frame_equal(observed, expected, check_dtype=False)
