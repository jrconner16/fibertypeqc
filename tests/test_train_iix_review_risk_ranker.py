from __future__ import annotations

import pandas as pd

from src.train_iix_review_risk_ranker import _prepare_training_table


def test_prepare_training_table_merges_feature_columns():
    reviewed = pd.DataFrame(
        [
            {
                "image_id": "img1",
                "label": 1,
                "candidate_pred_gated": "iix",
                "candidate_model_confidence": 0.6,
                "candidate_model_margin": 0.1,
                "type1_signal_evidence": "clear",
                "type2_signal_evidence": "low",
                "input_kind": "direct_czi",
                "genotype": "mdx",
                "timepoint": "12mo",
                "needs_correction": True,
            }
        ]
    )
    feature_table = pd.DataFrame(
        [
            {
                "image_id": "img1",
                "label": 1,
                "type1_mean": 5.0,
                "type2_mean": 1.0,
                "type1_coverage": 0.1,
                "type2_coverage": 0.0,
                "type1_cov_x_snr": 0.5,
                "type1_snr_mean": 8.0,
                "type2_cov_x_snr": 0.0,
                "type2_snr_mean": 1.0,
            }
        ]
    )
    merged = _prepare_training_table(reviewed, feature_table)
    assert len(merged) == 1
    assert float(merged.loc[0, "type1_mean"]) == 5.0
    assert float(merged.loc[0, "type2_snr_mean"]) == 1.0
