import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from fibertypeqc.model_manifest import ModelManifest
from fibertypeqc.semantic_model import (
    apply_semantic_predictions,
    load_semantic_model,
    predict_semantic_candidate,
)


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


def _manifest(tmp_path, **overrides):
    values = {
        "model_id": "toy",
        "task": "fiber_identity",
        "feature_schema_version": "multiplanel_features.v1",
        "required_markers": frozenset({"laminin", "type_i"}),
        "outputs": ("i", "iix"),
        "source_path": tmp_path / "toy.yaml",
        "features": ("type_i.mean", "area"),
    }
    values.update(overrides)
    return ModelManifest(**values)


def _fit(features, frame=False):
    from sklearn.tree import DecisionTreeClassifier

    x = pd.DataFrame({"type_i.mean": [0.9, 0.1], "area": [10.0, 10.0]})[features]
    model = DecisionTreeClassifier(random_state=0)
    model.fit(x if frame else x.to_numpy(), ["i", "iix"])
    return model


def test_bare_estimator_uses_manifest_feature_order(tmp_path):
    path = tmp_path / "bare.joblib"
    joblib.dump(_fit(["type_i.mean", "area"]), path)
    diagnostics = pd.DataFrame({"label": [1, 2], "area": [10.0, 10.0], "type_i.mean": [0.8, 0.2]})

    predictions = predict_semantic_candidate(diagnostics, path, _manifest(tmp_path))

    assert predictions["model_prediction"].tolist() == ["i", "iix"]
    assert {"prob_i", "prob_iix"} <= set(predictions.columns)


def test_bare_estimator_requires_manifest_features(tmp_path):
    path = tmp_path / "bare.joblib"
    joblib.dump(_fit(["type_i.mean", "area"]), path)

    with pytest.raises(ValueError, match="must declare ordered 'features'"):
        load_semantic_model(path, _manifest(tmp_path, features=None))


def test_model_classes_and_feature_count_must_match_manifest(tmp_path):
    path = tmp_path / "bare.joblib"
    joblib.dump(_fit(["type_i.mean", "area"]), path)

    with pytest.raises(ValueError, match="declares outputs"):
        load_semantic_model(path, _manifest(tmp_path, outputs=("i", "iia", "iix")))
    with pytest.raises(ValueError, match="expects 2 features"):
        load_semantic_model(path, _manifest(tmp_path, features=("type_i.mean",)))


def test_named_feature_order_mismatch_is_rejected(tmp_path):
    path = tmp_path / "named.joblib"
    joblib.dump(_fit(["area", "type_i.mean"], frame=True), path)
    diagnostics = pd.DataFrame({"label": [1], "area": [10.0], "type_i.mean": [0.8]})

    with pytest.raises(ValueError, match="different order"):
        predict_semantic_candidate(diagnostics, path, _manifest(tmp_path))


def test_non_finite_features_are_rejected(tmp_path):
    path = tmp_path / "bare.joblib"
    joblib.dump(_fit(["type_i.mean", "area"]), path)
    diagnostics = pd.DataFrame({"label": [1], "area": [10.0], "type_i.mean": [float("nan")]})

    with pytest.raises(ValueError, match="non-finite features for 1 fibers"):
        predict_semantic_candidate(diagnostics, path, _manifest(tmp_path))


def test_apply_semantic_predictions_replaces_calls_and_flags_uncertain_fibers():
    fibers = pd.DataFrame(
        {"label": [1, 2], "fiber_type": ["type1", "unknown"], "prob_iia": [float("nan")] * 2}
    )
    predictions = pd.DataFrame(
        {
            "label": [1, 2],
            "model_prediction": ["i", "iix"],
            "model_id": ["toy", "toy"],
            "task": ["fiber_identity"] * 2,
            "prob_i": [0.9, 0.45],
            "prob_iix": [0.1, 0.55],
        }
    )

    out = apply_semantic_predictions(
        fibers,
        predictions,
        confidence_threshold=0.7,
        margin_threshold=0.25,
        classifier_path="models/toy.joblib",
    )

    assert out["fiber_type"].tolist() == ["i", "iix"]
    assert "prob_iia" not in out.columns
    assert out["model_confidence"].round(2).tolist() == [0.9, 0.55]
    assert out["needs_review"].tolist() == [False, True]
    assert set(out["classification_method"]) == {"semantic_model"}
