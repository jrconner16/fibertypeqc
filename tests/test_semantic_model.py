import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier

from fibertypeqc.model_manifest import ModelManifest
from fibertypeqc.semantic_model import predict_semantic_candidate


def test_predict_semantic_candidate_uses_declared_features(tmp_path):
    model = DummyClassifier(strategy="constant", constant="i")
    model.fit([[0.0], [1.0]], ["i", "i"])
    path = tmp_path / "candidate.joblib"
    joblib.dump({"model": model, "features": ["type_i.mean"]}, path)
    manifest = ModelManifest(
        model_id="toy_type_i",
        task="fiber_identity",
        feature_schema_version="multiplanel_features.v1",
        required_markers=frozenset({"laminin", "type_i"}),
        outputs=("i",),
        source_path=tmp_path / "candidate.yaml",
    )

    predictions = predict_semantic_candidate(
        pd.DataFrame({"label": [1, 2], "type_i.mean": [0.2, 0.8]}), path, manifest
    )

    assert predictions["model_prediction"].tolist() == ["i", "i"]
    assert predictions["model_id"].tolist() == ["toy_type_i", "toy_type_i"]
