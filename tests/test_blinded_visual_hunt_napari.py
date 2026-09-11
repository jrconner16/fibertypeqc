from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.blinded_visual_hunt_napari import _fiber_keys, _load_development_manifest, _review_path


def test_visual_hunt_uses_development_sections_only(tmp_path: Path) -> None:
    manifest = tmp_path / "canonical_section_manifest.csv"
    pd.DataFrame(
        [
            {
                "mouse_id": "dev_mouse",
                "image_id": "dev_section_01",
                "section_id": "section-01",
                "split": "development",
                "raw_image_path": "/raw/dev.czi",
                "fiber_labels_path": "/labels/dev.tif",
            },
            {
                "mouse_id": "test_mouse",
                "image_id": "test_section_01",
                "section_id": "section-01",
                "split": "final_test",
                "raw_image_path": "/raw/test.czi",
                "fiber_labels_path": "/labels/test.tif",
            },
        ]
    ).to_csv(manifest, index=False)
    loaded = _load_development_manifest(manifest)
    assert loaded.image_id.tolist() == ["dev_section_01"]
    assert _review_path(manifest, "Jamie Reviewer").name == "development_visual_hunt_decisions.csv"
    assert _review_path(manifest, "Jamie", "development_visual_iix_hunt").name == (
        "development_visual_iix_hunt_decisions.csv"
    )


def test_visual_hunt_loads_excluded_fiber_keys(tmp_path: Path) -> None:
    decisions = tmp_path / "decisions.csv"
    pd.DataFrame([{"image_id": "img", "fiber_id": 42}]).to_csv(decisions, index=False)
    assert _fiber_keys(decisions) == {("img", 42)}


def test_visual_hunt_rejects_missing_manifest_columns(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.csv"
    pd.DataFrame([{"image_id": "img"}]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        _load_development_manifest(manifest)
