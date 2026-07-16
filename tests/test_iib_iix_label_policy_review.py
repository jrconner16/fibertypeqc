from __future__ import annotations

import pandas as pd

from src.apply_iib_iix_label_policy_review import apply_policy_overrides
from src.prepare_iib_iix_label_policy_review import (
    annotate_conservative_snapshot,
    build_liberal_review_queue,
)


def test_build_liberal_review_queue_selects_ambiguous_iib_iix_rows() -> None:
    df = pd.DataFrame(
        [
            {
                "image_id": "img1",
                "label": 1,
                "audit_final_label": "iib",
                "manual_supervision_split": "manual_train",
                "supervision_stratum": "train_iib_iix_ambiguity",
                "supervision_pool": "supervision_enrichment",
            },
            {
                "image_id": "img1",
                "label": 2,
                "audit_final_label": "iix",
                "manual_supervision_split": "manual_eval_holdout",
                "supervision_stratum": "iix_iib_boundary",
                "supervision_pool": "boundary_enrichment",
                "is_suspicious_iix_iib_boundary": True,
            },
            {
                "image_id": "img2",
                "label": 3,
                "audit_final_label": "iib",
                "manual_supervision_split": "manual_train",
                "supervision_stratum": "train_iia_focus",
                "supervision_pool": "supervision_enrichment",
            },
            {
                "image_id": "img3",
                "label": 4,
                "audit_final_label": "iia",
                "manual_supervision_split": "manual_eval_holdout",
                "supervision_stratum": "eval_iib_iix_ambiguity",
                "supervision_pool": "benchmark_enrichment",
            },
        ]
    )

    queue = build_liberal_review_queue(df)

    assert list(queue["label"]) == [2, 1]
    assert set(queue["conservative_audit_final_label"]) == {"iib", "iix"}
    assert set(queue["label_policy_target"]) == {"liberal_iib_iix_v1"}
    assert queue["audit_corrected_type"].eq("").all()


def test_apply_policy_overrides_updates_only_reviewed_rows() -> None:
    conservative = pd.DataFrame(
        [
            {
                "image_id": "img1",
                "label": 1,
                "audit_final_label": "iix",
                "manual_supervision_split": "manual_train",
            },
            {
                "image_id": "img1",
                "label": 2,
                "audit_final_label": "iib",
                "manual_supervision_split": "manual_eval_holdout",
            },
        ]
    )
    conservative = annotate_conservative_snapshot(
        conservative, policy_name="conservative_iib_iix_v1"
    )
    reviewed = pd.DataFrame(
        [
            {
                "image_id": "img1",
                "label": 1,
                "liberal_audit_final_label": "iib",
                "review_file": "queue_reviewed_img1.csv",
                "audit_corrected_type": "iib",
                "audit_is_uncertain": False,
                "audit_is_excluded": False,
                "audit_notes": "liberal call",
            }
        ]
    )

    out = apply_policy_overrides(conservative, reviewed, policy_name="liberal_iib_iix_v1")

    changed = out.loc[out["label"].eq(1)].iloc[0]
    unchanged = out.loc[out["label"].eq(2)].iloc[0]
    assert changed["audit_final_label"] == "iib"
    assert changed["conservative_audit_final_label"] == "iix"
    assert bool(changed["label_policy_changed"])
    assert changed["label_policy"] == "liberal_iib_iix_v1"
    assert unchanged["audit_final_label"] == "iib"
    assert not bool(unchanged["label_policy_changed"])

