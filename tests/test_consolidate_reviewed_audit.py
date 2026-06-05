from __future__ import annotations

import pandas as pd

from src.consolidate_reviewed_audit import consolidate_reviewed_audit


def test_consolidate_reviewed_audit_builds_final_labels(tmp_path):
    p1 = tmp_path / "combined_audit_sample_reviewed_img1.csv"
    pd.DataFrame(
        [
            {
                "image_id": "img1",
                "label": 1,
                "audit_corrected_type": "iib",
                "audit_is_uncertain": False,
                "audit_is_excluded": False,
            },
            {
                "image_id": "img1",
                "label": 2,
                "audit_corrected_type": "",
                "audit_is_uncertain": True,
                "audit_is_excluded": False,
            },
        ]
    ).to_csv(p1, index=False)
    p2 = tmp_path / "combined_audit_sample_reviewed_img2.csv"
    pd.DataFrame(
        [
            {
                "image_id": "img2",
                "label": 3,
                "audit_corrected_type": "",
                "audit_is_uncertain": False,
                "audit_is_excluded": True,
            },
            {
                "image_id": "img2",
                "label": 4,
                "audit_corrected_type": "",
                "audit_is_uncertain": False,
                "audit_is_excluded": False,
            },
        ]
    ).to_csv(p2, index=False)

    reviewed, benchmark = consolidate_reviewed_audit([p1, p2])

    assert len(reviewed) == 4
    assert len(benchmark) == 3
    assert set(benchmark["audit_final_label"]) == {"iib", "uncertain", "exclude"}
    assert not bool(reviewed.loc[reviewed["label"] == 4, "audit_reviewed"].iloc[0])


def test_consolidate_reviewed_audit_deduplicates_same_fiber_and_keeps_later_review(tmp_path):
    p1 = tmp_path / "combined_audit_sample_reviewed_img1.csv"
    pd.DataFrame(
        [
            {
                "image_id": "img1",
                "label": 1,
                "audit_corrected_type": "iib",
                "audit_is_uncertain": False,
                "audit_is_excluded": False,
            }
        ]
    ).to_csv(p1, index=False)
    p2 = tmp_path / "manual_labeling_round2_sample_reviewed_img1.csv"
    pd.DataFrame(
        [
            {
                "image_id": "img1",
                "label": 1,
                "audit_corrected_type": "iix",
                "audit_is_uncertain": False,
                "audit_is_excluded": False,
            }
        ]
    ).to_csv(p2, index=False)

    reviewed, benchmark = consolidate_reviewed_audit([p1, p2])

    assert len(reviewed) == 1
    assert len(benchmark) == 1
    assert benchmark.iloc[0]["audit_final_label"] == "iix"
