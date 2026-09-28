from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.score_locked_candidate as scorer
from src.run_reviewed_cohort_baseline import FEATURES


class _Model:
    classes_ = np.asarray(["i", "iia", "iib", "iix"])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(["iia", "iib"])

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray([[0.1, 0.7, 0.1, 0.1], [0.1, 0.2, 0.6, 0.1]])


def test_score_section_emits_locked_predictions_and_respects_manifest_eligibility(
    monkeypatch,
) -> None:
    features = pd.DataFrame(
        {
            "mouse_id": ["synthetic_specimen_a", "synthetic_specimen_a"],
            "cre_status": ["cre_negative_mdx", "cre_negative_mdx"],
            "image_id": ["synthetic_specimen_a_section_01", "synthetic_specimen_a_section_01"],
            "section_id": ["section-01", "section-01"],
            "pipeline_fiber_id": [101, 102],
            **{feature: [1.0, 2.0] for feature in FEATURES},
        }
    )
    monkeypatch.setattr(scorer, "_section_features", lambda section: features)
    section = pd.Series(
        {
            "mouse_id": "synthetic_specimen_a",
            "cre_status": "cre_negative_mdx",
            "image_id": "synthetic_specimen_a_section_01",
            "section_id": "section-01",
            "analysis_eligible": "false",
        }
    )

    predictions, summary = scorer.score_section(section, _Model())

    assert predictions["fiber_id"].tolist() == [101, 102]
    assert predictions["locked_prediction"].tolist() == ["iia", "iib"]
    assert predictions["model_margin"].tolist() == pytest.approx([0.6, 0.4])
    assert summary["analysis_eligible"] is False
    assert summary["n_iia"] == 1
    assert summary["pct_iib"] == 0.5
