from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.build_existing_ta_audit_f1_features import F1_COLUMNS
from src.build_notch3_protected_all_rois import build_all_rois
from src.build_notch3_protected_f1_features import build_features
from src.build_notch3_protected_match_manifest import build_manifest


def test_build_protected_manifest_resolves_only_protected_rows(tmp_path):
    raw_root = tmp_path / "raw"
    labels_root = tmp_path / "labels"
    results_root = tmp_path / "results"
    raw_root.mkdir()
    source = raw_root / "source.czi"
    source.write_bytes(b"czi")
    image_id = "image_a"
    image_labels = labels_root / image_id
    image_labels.mkdir(parents=True)
    labels = image_labels / "image_a_cellpose_labels.tif"
    labels.write_bytes(b"labels")
    image_results = results_root / image_id
    image_results.mkdir(parents=True)
    (image_results / "Results.txt").write_text("Label\nIIa\n", encoding="utf-8")
    (image_results / "ROISet.zip").write_bytes(b"roi")
    split = pd.DataFrame(
        {
            "image_id": [image_id, "development_image"],
            "group_id": ["mouse_a", "mouse_b"],
            "role": ["protected_holdout", "development_model_selection"],
            "raw_czi": ["source.czi", "unused.czi"],
        }
    )

    result = build_manifest(
        split,
        raw_root=raw_root,
        labels_root=labels_root,
        results_root=results_root,
    )

    assert result["image_id"].tolist() == [image_id]
    assert result.loc[0, "raw_czi"] == str(source)
    assert result.loc[0, "pipeline_labels"] == str(labels)


def test_build_all_rois_marks_only_unique_pipeline_assignments_eligible(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    pd.DataFrame({"Label": ["IIa", "IIx", "IIb"]}).to_csv(
        results_dir / "Results.txt", sep="\t", index=False
    )
    matching_root = tmp_path / "matching"
    image_root = matching_root / "image_a"
    image_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "roi_index": [0, 1, 2],
            "candidate_status": ["matched", "matched", "matched"],
            "label_id": [7, 8, 8],
        }
    ).to_csv(image_root / "image_a_match_candidates.csv", index=False)
    manifest = pd.DataFrame(
        {
            "image_id": ["image_a"],
            "group_id": ["mouse_a"],
            "myosight_results_dir": [str(results_dir)],
        }
    )

    result = build_all_rois(manifest, matching_root)

    assert result["myosight_label"].tolist() == ["iia", "iix", "iib"]
    assert result["one_to_one_eligible"].tolist() == [True, False, False]
    assert result.loc[0, "match_outcome"] == "matched_one_to_one"


def test_build_protected_features_joins_f1_and_rejects_string_false(tmp_path):
    diagnostics_root = tmp_path / "diagnostics"
    image_root = diagnostics_root / "image_a"
    image_root.mkdir(parents=True)
    diagnostic = {"label": [7, 8]}
    diagnostic.update({column: [1.0, 2.0] for column in F1_COLUMNS})
    pd.DataFrame(diagnostic).to_csv(image_root / "example_feature_diagnostics.csv", index=False)
    all_rois = pd.DataFrame(
        {
            "image_id": ["image_a", "image_a"],
            "group_id": ["mouse_a", "mouse_a"],
            "roi_index": [0, 1],
            "label_id": [7, 8],
            "myosight_label": ["iia", "iib"],
            "one_to_one_eligible": ["True", "False"],
            "match_outcome": ["matched_one_to_one", "matched_one_to_one"],
        }
    )

    result = build_features(all_rois, diagnostics_root)

    assert result["pipeline_label_id"].tolist() == [7]
    assert result["target_label"].tolist() == ["iia"]
    assert result["eligible_for_model_fitting"].tolist() == [False]
    assert set(F1_COLUMNS).issubset(result.columns)


def test_build_protected_features_rejects_invalid_boolean(tmp_path):
    table = pd.DataFrame(
        {
            "image_id": ["image_a"],
            "group_id": ["mouse_a"],
            "roi_index": [0],
            "label_id": [7],
            "myosight_label": ["iia"],
            "one_to_one_eligible": ["maybe"],
            "match_outcome": ["matched_one_to_one"],
        }
    )

    with pytest.raises(ValueError, match="invalid boolean"):
        build_features(table, Path(tmp_path))
