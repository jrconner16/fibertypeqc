from __future__ import annotations

import pandas as pd

from src.analyze_iia_gate import analyze_iia_gate


def test_analyze_iia_gate_adds_gated_strategies():
    feature_table = pd.DataFrame(
        [
            {
                "image_id": "img1",
                "label": 1,
                "split": "dev",
                "fiber_type": "iia",
                "needs_review": False,
                "model_confidence": 0.9,
                "model_margin": 0.8,
                "type1_mean": 0.5,
                "type2_mean": 8.0,
                "type1_coverage": 0.1,
                "type2_coverage": 0.9,
                "type1_cov_x_snr": 0.1,
                "type1_snr_mean": 0.4,
                "type2_cov_x_snr": 5.0,
                "type2_snr_mean": 6.0,
            },
            {
                "image_id": "img1",
                "label": 2,
                "split": "dev",
                "fiber_type": "iia",
                "needs_review": False,
                "model_confidence": 0.9,
                "model_margin": 0.8,
                "type1_mean": 4.0,
                "type2_mean": 5.0,
                "type1_coverage": 0.7,
                "type2_coverage": 0.8,
                "type1_cov_x_snr": 3.0,
                "type1_snr_mean": 3.5,
                "type2_cov_x_snr": 4.0,
                "type2_snr_mean": 4.5,
            },
            {
                "image_id": "img1",
                "label": 3,
                "split": "dev",
                "fiber_type": "iib",
                "needs_review": False,
                "model_confidence": 0.9,
                "model_margin": 0.8,
                "type1_mean": 8.0,
                "type2_mean": 0.5,
                "type1_coverage": 0.9,
                "type2_coverage": 0.1,
                "type1_cov_x_snr": 5.0,
                "type1_snr_mean": 6.0,
                "type2_cov_x_snr": 0.1,
                "type2_snr_mean": 0.4,
            },
        ]
    )
    benchmark_split = pd.DataFrame(
        [
            {
                "image_id": "img1",
                "label": 1,
                "audit_final_label": "iia",
                "manual_supervision_split": "manual_eval_holdout",
            },
            {
                "image_id": "img1",
                "label": 2,
                "audit_final_label": "iix",
                "manual_supervision_split": "manual_eval_holdout",
            },
            {
                "image_id": "img1",
                "label": 3,
                "audit_final_label": "iib",
                "manual_supervision_split": "manual_eval_holdout",
            },
        ]
    )
    weighted_predictions = pd.DataFrame(
        [
            {
                "image_id": "img1",
                "label": 1,
                "candidate_name": "baseline_gb",
                "predicted_fiber_type": "iia",
            },
            {
                "image_id": "img1",
                "label": 2,
                "candidate_name": "baseline_gb",
                "predicted_fiber_type": "iia",
            },
            {
                "image_id": "img1",
                "label": 3,
                "candidate_name": "baseline_gb",
                "predicted_fiber_type": "iib",
            },
        ]
    )
    true_iia_reviewed = pd.DataFrame(
        [
            {
                "image_id": "hunt1",
                "label": 10,
                "type1_mean": 0.5,
                "type2_mean": 8.0,
                "type1_coverage": 0.1,
                "type2_coverage": 0.9,
                "type1_cov_x_snr": 0.1,
                "type1_snr_mean": 0.4,
                "type2_cov_x_snr": 5.0,
                "type2_snr_mean": 6.0,
                "audit_corrected_type": "iia",
            },
            {
                "image_id": "hunt2",
                "label": 11,
                "type1_mean": 0.6,
                "type2_mean": 7.5,
                "type1_coverage": 0.12,
                "type2_coverage": 0.88,
                "type1_cov_x_snr": 0.12,
                "type1_snr_mean": 0.5,
                "type2_cov_x_snr": 4.8,
                "type2_snr_mean": 5.8,
                "audit_corrected_type": "iia",
            },
        ]
    )

    metrics, predictions, report = analyze_iia_gate(
        feature_table,
        benchmark_split,
        weighted_predictions,
        true_iia_reviewed,
        candidate_name="baseline_gb",
        gate_quantiles=[0.05, 0.10],
    )

    assert {
        "pipeline_current",
        "baseline_gb",
    }.issubset(set(metrics["strategy"]))
    assert "pipeline_gated_iia_q0.05" in set(metrics["strategy"])
    assert "baseline_gb_gated_iia_q0.10" in set(metrics["strategy"])
    assert "baseline_gb_gated_iia_iib_redirect_q0.10" in set(metrics["strategy"])
    assert "gate_iia_ok" in predictions.columns
    assert "gate_iib_redirect_ok" in predictions.columns
    assert "pred_baseline_gb_gated_iia_iib_redirect" in predictions.columns
    assert "Best gated baseline_gb quantile" in report
