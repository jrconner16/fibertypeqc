from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fibertypeqc.metrics import add_probability_metrics, normalized_entropy, soft_type_composition


def test_normalized_entropy_is_low_for_certain_and_high_for_uniform():
    entropy = normalized_entropy(np.array([[1.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3]]))

    assert entropy[0] == pytest.approx(0.0)
    assert entropy[1] == pytest.approx(1.0)


def test_add_probability_metrics_adds_entropy_column():
    fibers = pd.DataFrame(
        {
            "prob_iib": [0.9, 0.2],
            "prob_iia": [0.05, 0.7],
            "prob_iix": [0.05, 0.1],
        }
    )

    out = add_probability_metrics(fibers)

    assert "model_entropy_norm" in out.columns
    assert out.loc[0, "model_entropy_norm"] < out.loc[1, "model_entropy_norm"]


def test_soft_type_composition_averages_probabilities_as_percentages():
    fibers = pd.DataFrame(
        {
            "prob_iib": [1.0, 0.5],
            "prob_iia": [0.0, 0.25],
            "prob_iix": [0.0, 0.25],
        }
    )

    composition = soft_type_composition(fibers)

    assert composition["iib"] == pytest.approx(75.0)
    assert composition["iia"] == pytest.approx(12.5)
    assert composition["iix"] == pytest.approx(12.5)


def test_probability_metrics_require_probability_columns():
    with pytest.raises(ValueError, match="Missing probability columns"):
        add_probability_metrics(pd.DataFrame({"prob_iib": [1.0]}))
