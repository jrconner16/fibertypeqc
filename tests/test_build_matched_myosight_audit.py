from __future__ import annotations

import pandas as pd

from src.build_matched_myosight_audit import (
    _discover_best_matched_files,
    _load_manifest,
    build_matched_myosight_audit,
)


def test_build_matched_myosight_audit_filters_disparity_pairs(tmp_path):
    feature_table = pd.DataFrame(
        [
            {
                "image_id": "img1",
                "_merge_image_id": "img1",
                "label": 10,
                "fiber_type": "iix",
                "split": "dev",
                "needs_review": True,
            },
            {
                "image_id": "img1",
                "_merge_image_id": "img1",
                "label": 11,
                "fiber_type": "iib",
                "split": "dev",
                "needs_review": False,
            },
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "image_id": "img1",
                "myosight_results": "myosight/Results.txt",
            }
        ]
    ).to_csv(manifest_path, index=False)
    myosight_dir = tmp_path / "myosight"
    myosight_dir.mkdir()
    pd.DataFrame(
        [
            {"Label": "IIa", "Area": 10.0},
            {"Label": "IIx", "Area": 12.0},
        ]
    ).to_csv(myosight_dir / "Results.txt", sep="\t", index=False)

    matched_dir = tmp_path / "matched"
    matched_dir.mkdir()
    pd.DataFrame(
        [
            {"roi_index": 0, "label_id": 10, "centroid_distance_px": 3.0, "area_ratio": 0.9},
            {"roi_index": 1, "label_id": 11, "centroid_distance_px": 2.0, "area_ratio": 1.1},
        ]
    ).to_csv(matched_dir / "img1_offset_sweep_matched_fibers.csv", index=False)

    manifest = _load_manifest(manifest_path)
    matched_files = _discover_best_matched_files(matched_dir, manifest)
    out = build_matched_myosight_audit(
        feature_table=feature_table,
        manifest=manifest,
        matched_files_by_image=matched_files,
        manifest_path=manifest_path,
    )
    assert len(out) == 2
    assert set(out["myosight_vs_pipeline_pair"]) == {"iia_vs_iix", "iix_vs_iib"}
    disparity = out[(out["disparity_bucket"] != "") & (~out["labels_agree"])]
    assert set(disparity["disparity_bucket"]) == {"iia_vs_iix", "iix_vs_iib"}


def test_discover_best_matched_files_prefers_better_area_ratio(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([{"image_id": "img1", "myosight_results": "myo/Results.txt"}]).to_csv(
        manifest_path, index=False
    )
    manifest = _load_manifest(manifest_path)
    matched_dir = tmp_path / "matched"
    matched_dir.mkdir()
    pd.DataFrame([{"roi_index": 0, "label_id": 1, "area_ratio": 1.0}]).to_csv(
        matched_dir / "img1_good_matched_fibers.csv", index=False
    )
    pd.DataFrame([{"roi_index": 0, "label_id": 1, "area_ratio": 10.0}]).to_csv(
        matched_dir / "img1_bad_matched_fibers.csv", index=False
    )

    selected = _discover_best_matched_files(matched_dir, manifest)
    assert selected["img1"].name == "img1_good_matched_fibers.csv"
