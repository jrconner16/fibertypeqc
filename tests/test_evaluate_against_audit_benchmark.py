from __future__ import annotations

import pandas as pd

from src.evaluate_against_audit_benchmark import evaluate_against_audit_benchmark


def test_evaluate_against_audit_benchmark_reports_pipeline_comparator():
    feature_table = pd.DataFrame(
        [
            {
                "image_id": "img1",
                "label": 1,
                "split": "train",
                "fiber_type": "iib",
                "area": 10.0,
                "type1_mean": 5.0,
                "type2_mean": 1.0,
                "type1_p75": 5.0,
                "type2_p75": 1.0,
                "type1_p90": 5.0,
                "type2_p90": 1.0,
                "type1_pctl": 5.0,
                "type2_pctl": 1.0,
                "type1_coverage": 0.9,
                "type2_coverage": 0.1,
                "type_ratio": 5.0,
                "type_diff": 4.0,
                "type_pctl_ratio": 5.0,
                "type_pctl_diff": 4.0,
                "type_p75_ratio": 5.0,
                "type_p75_diff": 4.0,
                "type_p90_ratio": 5.0,
                "type_p90_diff": 4.0,
                "type_cov_ratio": 9.0,
                "type_cov_diff": 0.8,
            },
            {
                "image_id": "img2",
                "label": 2,
                "split": "train",
                "fiber_type": "iix",
                "area": 8.0,
                "type1_mean": 0.5,
                "type2_mean": 0.4,
                "type1_p75": 0.5,
                "type2_p75": 0.4,
                "type1_p90": 0.5,
                "type2_p90": 0.4,
                "type1_pctl": 0.5,
                "type2_pctl": 0.4,
                "type1_coverage": 0.1,
                "type2_coverage": 0.1,
                "type_ratio": 1.25,
                "type_diff": 0.1,
                "type_pctl_ratio": 1.25,
                "type_pctl_diff": 0.1,
                "type_p75_ratio": 1.25,
                "type_p75_diff": 0.1,
                "type_p90_ratio": 1.25,
                "type_p90_diff": 0.1,
                "type_cov_ratio": 1.0,
                "type_cov_diff": 0.0,
            },
            {
                "image_id": "img3",
                "label": 3,
                "split": "dev",
                "fiber_type": "iix",
                "area": 8.5,
                "type1_mean": 0.6,
                "type2_mean": 0.5,
                "type1_p75": 0.6,
                "type2_p75": 0.5,
                "type1_p90": 0.6,
                "type2_p90": 0.5,
                "type1_pctl": 0.6,
                "type2_pctl": 0.5,
                "type1_coverage": 0.1,
                "type2_coverage": 0.1,
                "type_ratio": 1.2,
                "type_diff": 0.1,
                "type_pctl_ratio": 1.2,
                "type_pctl_diff": 0.1,
                "type_p75_ratio": 1.2,
                "type_p75_diff": 0.1,
                "type_p90_ratio": 1.2,
                "type_p90_diff": 0.1,
                "type_cov_ratio": 1.0,
                "type_cov_diff": 0.0,
            },
        ]
    )
    benchmark = pd.DataFrame(
        [
            {"image_id": "img1", "label": 1, "audit_final_label": "iib", "split": "train"},
            {"image_id": "img3", "label": 3, "audit_final_label": "iix", "split": "dev"},
            {"image_id": "img3", "label": 4, "audit_final_label": "uncertain", "split": "dev"},
        ]
    )

    metrics, predictions, report = evaluate_against_audit_benchmark(feature_table, benchmark)

    assert "pipeline_current" in set(metrics["comparator"])
    assert "candidate_pred_pipeline_current" in set(predictions.columns)
    assert "[scope=all]" in report
