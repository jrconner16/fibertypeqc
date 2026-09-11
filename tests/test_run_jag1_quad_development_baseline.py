from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.run_jag1_quad_development_baseline import (
    FEATURES,
    _load_manifest,
    _load_reviewed,
    _lomo_predictions,
)


def test_conservative_baseline_excludes_final_and_named_mouse(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "mouse_id": "keep",
                "cre_status": "neg",
                "image_id": "keep_01",
                "section_id": "section-01",
                "split": "development",
                "raw_image_path": "/raw",
                "fiber_labels_path": "/labels",
            },
            {
                "mouse_id": "351545_R",
                "cre_status": "neg",
                "image_id": "exclude_01",
                "section_id": "section-01",
                "split": "development",
                "raw_image_path": "/raw",
                "fiber_labels_path": "/labels",
            },
            {
                "mouse_id": "test",
                "cre_status": "pos",
                "image_id": "test_01",
                "section_id": "section-01",
                "split": "final_test",
                "raw_image_path": "/raw",
                "fiber_labels_path": "/labels",
            },
        ]
    ).to_csv(manifest_path, index=False)
    loaded = _load_manifest(manifest_path, "351545_R")
    assert loaded.image_id.tolist() == ["keep_01"]


def test_current_reviews_keep_allowed_labels_and_reject_duplicates(tmp_path: Path) -> None:
    decisions = tmp_path / "review.csv"
    pd.DataFrame(
        [
            {
                "mouse_id": "m",
                "image_id": "img",
                "section_id": "section-01",
                "fiber_id": 1,
                "sampling_stratum": "development_random",
                "label": "i",
            },
            {
                "mouse_id": "m",
                "image_id": "img",
                "section_id": "section-01",
                "fiber_id": 2,
                "sampling_stratum": "development_random",
                "label": "exclude",
            },
        ]
    ).to_csv(decisions, index=False)
    reviewed = _load_reviewed([decisions], {"img"})
    assert reviewed[["fiber_id", "label"]].to_dict("records") == [{"fiber_id": 1, "label": "i"}]


def test_lomo_reports_random_stratum_separately() -> None:
    rows = []
    for mouse, offset in (("m1", 0.0), ("m2", 1.0)):
        for number, label in enumerate(("i", "iia", "iib", "iix"), start=1):
            row = {
                "mouse_id": mouse,
                "label": label,
                "sampling_stratum": "development_random",
            }
            row.update({feature: offset + number for feature in FEATURES})
            rows.append(row)
    predictions, metrics = _lomo_predictions(pd.DataFrame(rows))
    assert len(predictions) == 8
    assert metrics["development_random"]["n"] == 8
