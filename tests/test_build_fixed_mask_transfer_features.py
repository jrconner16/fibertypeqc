from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.build_fixed_mask_transfer_features import (
    REQUIRED_F1_FEATURES,
    build_transfer_features,
)


def _diagnostics(tmp_path: Path) -> dict[str, Path]:
    path = tmp_path / "image_a_feature_diagnostics.csv"
    table = pd.DataFrame({"label": [10, 20]})
    for index, column in enumerate(REQUIRED_F1_FEATURES, start=1):
        table[column] = [float(index), float(index + 1)]
    table.to_csv(path, index=False)
    return {"image_a": path}


def _all_rois() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": ["image_a", "image_a", "image_a"],
            "roi_index": [0, 1, 2],
            "label_id": [10, 20, 0],
            "myosight_label": ["iia", "iib", "iix"],
            "one_to_one_eligible": [True, True, False],
            "match_outcome": [
                "matched_one_to_one",
                "matched_one_to_one",
                "no_pipeline_label_at_centroid",
            ],
        }
    )


def test_build_transfer_features_joins_only_one_to_one_rows(tmp_path):
    out = build_transfer_features(
        _all_rois(),
        _diagnostics(tmp_path),
        expected_images=1,
    )

    assert list(out["pipeline_label_id"]) == [10, 20]
    assert list(out["target_label"]) == ["iia", "iib"]
    assert out["group_id"].eq(out["image_id"]).all()
    assert out["label_authority"].eq("myosight_derived").all()
    assert out["evidence_role"].eq("development_transfer").all()
    assert set(REQUIRED_F1_FEATURES).issubset(out.columns)


def test_build_transfer_features_rejects_duplicate_pipeline_claims(tmp_path):
    rois = _all_rois()
    rois.loc[1, "label_id"] = 10

    with pytest.raises(ValueError, match="duplicate pipeline-label"):
        build_transfer_features(rois, _diagnostics(tmp_path))


def test_build_transfer_features_requires_complete_current_schema(tmp_path):
    path = tmp_path / "image_a_feature_diagnostics.csv"
    pd.DataFrame({"label": [10, 20], "type_iia.mean": [1.0, 2.0]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing F1 features"):
        build_transfer_features(_all_rois(), {"image_a": path})


def test_build_transfer_features_accepts_casefolded_diagnostic_key(tmp_path):
    result = build_transfer_features(
        _all_rois(), {"IMAGE_A": next(iter(_diagnostics(tmp_path).values()))}
    )

    assert len(result) == 2
