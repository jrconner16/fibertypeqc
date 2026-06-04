from __future__ import annotations

import pandas as pd
import pytest

from src.build_candidate_feature_table import (
    assemble_candidate_feature_table,
    discover_diagnostics_files,
)


def test_assemble_candidate_feature_table_orders_metadata_and_baseline_features(tmp_path):
    image_a = tmp_path / "image_a_feature_diagnostics.csv"
    image_b = tmp_path / "nested" / "image_b_feature_diagnostics.csv"
    image_b.parent.mkdir(parents=True)

    df_a = pd.DataFrame(
        {
            "label": [1, 2],
            "fiber_type": ["iib", "iix"],
            "fiber_type_source": ["direct_marker", "residual_inference"],
            "needs_review": [False, True],
            "area": [100.0, 120.0],
            "type1_mean": [2.0, 0.1],
            "type2_mean": [0.5, 0.2],
            "type_cov_sum": [0.7, 0.3],
        }
    )
    df_b = pd.DataFrame(
        {
            "label": [3],
            "fiber_type": ["iia"],
            "fiber_type_source": ["direct_marker"],
            "needs_review": [False],
            "area": [80.0],
            "type1_mean": [0.3],
            "type2_mean": [1.7],
            "type_cov_sum": [0.9],
        }
    )
    df_a.to_csv(image_a, index=False)
    df_b.to_csv(image_b, index=False)

    manifest = pd.DataFrame(
        {
            "image_id": ["image_a", "image_b"],
            "split": ["train", "heldout"],
            "cohort": ["A", "B"],
        }
    )

    files = discover_diagnostics_files(tmp_path, "**/*_feature_diagnostics.csv")
    out = assemble_candidate_feature_table(files, manifest)

    assert list(out["image_id"]) == ["image_a", "image_a", "image_b"]
    assert {"split", "cohort", "type_cov_sum"}.issubset(out.columns)
    assert out.columns[0] == "image_id"
    assert out.columns[1] == "feature_diagnostics_path"
    assert out.columns[2] == "label"
    assert out.columns[3] == "fiber_type"
    assert out.columns[4] == "fiber_type_source"
    assert out.columns[5] == "needs_review"
    assert out.columns[6] == "area"


def test_assemble_candidate_feature_table_requires_input_files():
    with pytest.raises(ValueError, match="No \\*_feature_diagnostics\\.csv files were found"):
        assemble_candidate_feature_table([])
