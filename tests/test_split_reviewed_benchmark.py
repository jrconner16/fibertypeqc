from __future__ import annotations

import pandas as pd

from src.split_reviewed_benchmark import split_reviewed_benchmark


def test_split_reviewed_benchmark_holds_out_per_class_when_possible():
    benchmark = pd.DataFrame(
        [
            {"image_id": "img1", "label": 1, "audit_final_label": "iib"},
            {"image_id": "img1", "label": 2, "audit_final_label": "iib"},
            {"image_id": "img2", "label": 1, "audit_final_label": "iix"},
            {"image_id": "img2", "label": 2, "audit_final_label": "iix"},
            {"image_id": "img3", "label": 1, "audit_final_label": "uncertain"},
        ]
    )

    out = split_reviewed_benchmark(benchmark, eval_frac=0.5, seed=0)

    assert "manual_supervision_split" in out.columns
    assert "manual_supervision_weight" in out.columns
    assert (
        out.loc[out["audit_final_label"] == "iib", "manual_supervision_split"]
        .eq("manual_eval_holdout")
        .sum()
        >= 1
    )
    assert (
        out.loc[out["audit_final_label"] == "iix", "manual_supervision_split"]
        .eq("manual_eval_holdout")
        .sum()
        >= 1
    )
    assert (
        out.loc[out["audit_final_label"] == "uncertain", "manual_supervision_split"]
        .eq("manual_train")
        .all()
    )


def test_split_reviewed_benchmark_preserves_round2_pool_intent():
    benchmark = pd.DataFrame(
        [
            {
                "image_id": "img1",
                "label": 1,
                "audit_final_label": "iib",
                "manual_round2_pool": "manual_eval_candidate",
            },
            {
                "image_id": "img2",
                "label": 2,
                "audit_final_label": "iix",
                "manual_round2_pool": "manual_train_candidate",
            },
            {
                "image_id": "img3",
                "label": 3,
                "audit_final_label": "iix",
                "manual_round2_pool": "",
            },
        ]
    )

    out = split_reviewed_benchmark(benchmark, eval_frac=0.5, seed=0)

    assert (
        out.loc[out["image_id"] == "img1", "manual_supervision_split"].iloc[0]
        == "manual_eval_holdout"
    )
    assert (
        out.loc[out["image_id"] == "img2", "manual_supervision_split"].iloc[0]
        == "manual_train"
    )
