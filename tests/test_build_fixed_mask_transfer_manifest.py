from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.build_fixed_mask_transfer_manifest import build_fixed_mask_manifest


def _source_tree(tmp_path: Path) -> tuple[pd.DataFrame, Path, Path]:
    source_root = tmp_path / "private"
    image = source_root / "images" / "image_a.tif"
    results = source_root / "myosight" / "image_a"
    labels_root = source_root / "diagnostics"
    label_dir = labels_root / "image_a"
    image.parent.mkdir(parents=True)
    results.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image.write_bytes(b"image")
    (results / "Results.txt").write_text("Label\nIIb\n")
    (results / "ROISet.zip").write_bytes(b"roi")
    (label_dir / "image_a_cellpose_labels.tif").write_bytes(b"labels")
    source = pd.DataFrame(
        {
            "image_id": ["image_a"],
            "source_image": ["images/image_a.tif"],
            "myosight_summary_path": ["myosight/image_a/Results.txt"],
            "split": ["heldout"],
        }
    )
    return source, source_root, labels_root


def test_build_fixed_mask_manifest_resolves_assets_and_replaces_historical_role(tmp_path):
    source, source_root, labels_root = _source_tree(tmp_path)

    out = build_fixed_mask_manifest(
        source,
        path_root=source_root,
        labels_root=labels_root,
        expected_images=1,
    )

    assert list(out.columns) == [
        "image_id",
        "input_path",
        "labels_path",
        "myosight_results_dir",
        "evidence_role",
    ]
    assert out.loc[0, "evidence_role"] == "development_transfer"
    assert "split" not in out.columns
    assert Path(out.loc[0, "labels_path"]).name == "image_a_cellpose_labels.tif"


def test_build_fixed_mask_manifest_requires_exactly_one_cached_mask(tmp_path):
    source, source_root, labels_root = _source_tree(tmp_path)
    (labels_root / "image_a" / "duplicate_cellpose_labels.tif").write_bytes(b"labels")

    with pytest.raises(ValueError, match="exactly one"):
        build_fixed_mask_manifest(source, path_root=source_root, labels_root=labels_root)


def test_build_fixed_mask_manifest_checks_expected_image_count(tmp_path):
    source, source_root, labels_root = _source_tree(tmp_path)

    with pytest.raises(ValueError, match="Expected 2"):
        build_fixed_mask_manifest(
            source,
            path_root=source_root,
            labels_root=labels_root,
            expected_images=2,
        )
