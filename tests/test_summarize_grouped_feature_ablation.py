from __future__ import annotations

import pandas as pd
import pytest

from src.summarize_grouped_feature_ablation import (
    build_decision_record,
    paired_bootstrap_summary,
)


def _metrics() -> pd.DataFrame:
    rows = []
    for group, f0, f1 in [
        ("g1", 0.50, 0.60),
        ("g2", 0.60, 0.62),
        ("g3", 0.70, 0.65),
        ("g4", 0.80, 0.82),
    ]:
        rows.extend(
            [
                {
                    "heldout_group_id": group,
                    "feature_set": "F0",
                    "fold_index": 0,
                    "macro_f1": f0,
                    "multiclass_brier_score": 1.0 - f0,
                },
                {
                    "heldout_group_id": group,
                    "feature_set": "F1",
                    "fold_index": 0,
                    "macro_f1": f1,
                    "multiclass_brier_score": 1.0 - f1,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_paired_bootstrap_summary_is_grouped_and_deterministic():
    first = paired_bootstrap_summary(
        _metrics(), comparator="F0", candidate="F1", repetitions=500, seed=11
    )
    second = paired_bootstrap_summary(
        _metrics(), comparator="F0", candidate="F1", repetitions=500, seed=11
    )

    pd.testing.assert_frame_equal(first, second)
    macro = first.set_index("metric").loc["macro_f1"]
    assert macro["paired_median_delta"] == pytest.approx(0.02)
    assert macro["favorable_groups"] == 3
    brier = first.set_index("metric").loc["multiclass_brier_score"]
    assert brier["direction_better"] == "lower"
    assert brier["favorable_groups"] == 3


def test_paired_bootstrap_rejects_different_group_sets():
    metrics = _metrics()
    metrics = metrics.loc[
        ~((metrics["feature_set"] == "F1") & (metrics["heldout_group_id"] == "g4"))
    ]

    with pytest.raises(ValueError, match="group sets differ"):
        paired_bootstrap_summary(metrics, comparator="F0", candidate="F1", repetitions=10, seed=1)


def test_decision_record_does_not_claim_advancement_without_transfer_evidence():
    summary = paired_bootstrap_summary(
        _metrics(), comparator="F0", candidate="F1", repetitions=100, seed=3
    )
    contract = {
        "contract_id": "synthetic_contract.v1",
        "evaluation": {"primary_metric": "per_mouse_macro_f1"},
    }

    record = build_decision_record(
        summary,
        contract,
        comparator="F0",
        candidate="F1",
        repetitions=100,
        seed=3,
        metrics_sha256="a" * 64,
        contract_sha256="b" * 64,
    )

    assert record["status"] == "baseline_characterization"
    assert record["candidate_advancement_assessed"] is False
    assert record["candidate_advances"] is None
    assert record["protected_evaluation_read"] is False
    assert record["primary_metric"]["name"] == "macro_f1"
