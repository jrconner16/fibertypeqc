import numpy as np

from src.quantify_classify import (
    QCConfig,
    QuantifyConfig,
    build_feature_diagnostics_table,
    qc_flags_from_fibers,
    quantify_labels,
)


def test_quantify_labels_supports_type_i_iia_diagnostics_without_legacy_calls():
    labels = np.array([[1, 1], [2, 2]], dtype=np.int32)
    image = np.zeros((4, 2, 2), dtype=np.float32)
    image[0] = [[2, 3], [4, 5]]  # Type I
    image[1] = [[5, 4], [3, 2]]  # IIa
    fibers = quantify_labels(
        labels,
        image,
        QuantifyConfig(
            type1_channel=None,
            type2_channel=1,
            i_channel=0,
            typing_preprocess="raw",
            typing_erode_px=0,
        ),
    )

    assert set(fibers["fiber_type"]) == {"unknown"}
    assert set(fibers["classification_method"]) == {"diagnostics_only"}
    assert {"i", "iia"} == set(fibers["available_markers"].iloc[0].split("|"))
    diagnostics = build_feature_diagnostics_table(
        fibers,
        QuantifyConfig(type1_channel=None, type2_channel=1, i_channel=0),
    )
    assert {"type_i.mean", "type_iia.mean"}.issubset(diagnostics.columns)
    qc = qc_flags_from_fibers(fibers, QCConfig(min_labels=1, max_unknown_rate=1.0))
    assert np.isnan(qc["type_corr"])
    assert not qc["flag_high_type_corr"]
