from __future__ import annotations

import pandas as pd

from src.review_audit_napari import (
    _load_or_create_review_table,
    _save_review_table,
    _scaled_display_data,
)


def test_load_or_create_review_table_restores_existing_audit_labels(tmp_path):
    audit = pd.DataFrame(
        [
            {"image_id": "img1", "label": 1, "fiber_type": "iix"},
            {"image_id": "img1", "label": 2, "fiber_type": "iia"},
        ]
    )
    review_output = tmp_path / "audit_reviewed_img1.csv"
    saved = pd.DataFrame(
        [
            {
                "image_id": "img1",
                "label": 2,
                "audit_corrected_type": "iib",
                "audit_is_uncertain": False,
                "audit_is_excluded": False,
                "audit_notes": "checked",
            }
        ]
    )
    saved.to_csv(review_output, index=False)

    out = _load_or_create_review_table(audit, review_output)

    row1 = out.loc[out["label"] == 1].iloc[0]
    assert row1["audit_corrected_type"] == ""
    row2 = out.loc[out["label"] == 2].iloc[0]
    assert row2["audit_corrected_type"] == "iib"
    assert row2["audit_notes"] == "checked"


def test_load_or_create_review_table_coerces_blank_review_columns_to_strings(tmp_path):
    audit = pd.DataFrame(
        [
            {
                "image_id": "img1",
                "label": 1,
                "fiber_type": "iix",
                "audit_corrected_type": float("nan"),
                "audit_is_uncertain": False,
                "audit_is_excluded": False,
                "audit_notes": float("nan"),
            }
        ]
    )

    out = _load_or_create_review_table(audit, tmp_path / "missing_review.csv")

    row = out.iloc[0]
    assert row["audit_corrected_type"] == ""
    assert isinstance(row["audit_corrected_type"], str)
    assert row["audit_notes"] == ""
    assert isinstance(row["audit_notes"], str)


def test_save_review_table_writes_csv(tmp_path):
    review_output = tmp_path / "audit_reviewed_img1.csv"
    table = pd.DataFrame(
        [{"image_id": "img1", "label": 1, "audit_corrected_type": "iix"}]
    )
    _save_review_table(table, review_output)
    loaded = pd.read_csv(review_output)
    assert loaded.loc[0, "audit_corrected_type"] == "iix"


def test_scaled_display_data_clips_to_unit_interval():
    scaled = _scaled_display_data([0.2, 0.7, 1.0], gain=0.5)
    assert list(scaled) == [0.1, 0.35, 0.5]

    clipped = _scaled_display_data([0.6, 0.8, 1.0], gain=2.0)
    assert list(clipped) == [1.0, 1.0, 1.0]
