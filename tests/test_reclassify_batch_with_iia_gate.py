import numpy as np
import pandas as pd

from src.reclassify_batch_with_iia_gate import reclassify_and_gate


class _AlwaysIIaModel:
    feature_names_in_ = np.array(["type1_mean", "type2_mean"])
    classes_ = np.array(["iia", "iix"])

    def predict(self, features):
        return np.repeat("iia", len(features))

    def predict_proba(self, features):
        return np.tile([0.9, 0.1], (len(features), 1))


def test_reclassify_and_gate_replaces_model_calls_and_demotes_unsupported_iia(tmp_path):
    diagnostics = pd.DataFrame(
        {
            "label": [1, 2],
            "type1_mean": [0.0, 1.0],
            "type2_mean": [1.0, 1.0],
            "type1_coverage": [0.0, 0.0],
            "type2_coverage": [1.0, 1.0],
            "type1_snr_mean": [0.0, 0.0],
            "type2_snr_mean": [1.0, 1.0],
            "type2_cov_x_snr": [1.0, 1.0],
        }
    )
    fibers = pd.DataFrame({"label": [1, 2], "fiber_type": ["iib", "iib"]})
    thresholds = {
        "min_type2_snr_mean": 0.5,
        "min_type2_coverage": 0.5,
        "min_type2_cov_x_snr": 0.5,
        "max_type1_snr_mean": 0.5,
        "max_type1_coverage": 0.5,
    }

    updated_fibers, updated_diagnostics = reclassify_and_gate(
        fibers, diagnostics, _AlwaysIIaModel(), thresholds, tmp_path / "model.joblib"
    )

    assert updated_diagnostics["fiber_type_pre_iia_gate"].tolist() == ["iia", "iia"]
    assert updated_diagnostics["fiber_type"].tolist() == ["iia", "iix"]
    assert updated_fibers["fiber_type"].tolist() == ["iia", "iix"]
