from __future__ import annotations

import pandas as pd

from src.sample_combined_audit_set import sample_combined_audit_set


def test_sample_combined_audit_set_respects_image_cap_and_bucket_sampling():
    df = pd.DataFrame(
        [
            {
                "image_id": "img1",
                "label": 1,
                "split": "dev",
                "audit_bucket_count": 2,
                "bucket_a": True,
                "bucket_b": False,
            },
            {
                "image_id": "img1",
                "label": 2,
                "split": "dev",
                "audit_bucket_count": 1,
                "bucket_a": True,
                "bucket_b": False,
            },
            {
                "image_id": "img1",
                "label": 3,
                "split": "heldout",
                "audit_bucket_count": 1,
                "bucket_a": False,
                "bucket_b": True,
            },
            {
                "image_id": "img2",
                "label": 1,
                "split": "train",
                "audit_bucket_count": 1,
                "bucket_a": True,
                "bucket_b": False,
            },
            {
                "image_id": "img2",
                "label": 2,
                "split": "dev",
                "audit_bucket_count": 1,
                "bucket_a": False,
                "bucket_b": True,
            },
        ]
    )

    out = sample_combined_audit_set(df, per_bucket=2, max_per_image=2, seed=0)

    assert len(out) <= 4
    assert out.groupby("image_id").size().max() <= 2
    assert set(out.columns) >= {"image_id", "label", "bucket_a", "bucket_b"}
