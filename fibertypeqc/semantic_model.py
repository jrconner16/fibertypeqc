"""Explicit inference for semantic, panel-aware candidate models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fibertypeqc.model_manifest import ModelManifest


def load_semantic_model(model_path: Path, manifest: ModelManifest) -> tuple[Any, list[str]]:
    """Load a semantic model and its ordered features, checked against the manifest.

    Accepts either a ``{"model": estimator, "features": [...]}`` bundle or a bare fitted
    estimator whose ordered features are declared in the manifest. Feature order, feature count,
    and output classes must agree with the manifest when it declares them.
    """
    loaded = joblib.load(model_path)
    if isinstance(loaded, dict):
        if "model" not in loaded or "features" not in loaded:
            raise ValueError("Semantic candidate bundle must contain 'model' and 'features'.")
        model = loaded["model"]
        features = list(loaded["features"])
        if manifest.features is not None and tuple(features) != manifest.features:
            raise ValueError(
                f"Model '{manifest.model_id}' bundle features differ from its manifest features."
            )
    else:
        model = loaded
        if manifest.features is None:
            raise ValueError(
                f"Model '{manifest.model_id}' is a bare estimator; its manifest must declare "
                "ordered 'features'."
            )
        features = list(manifest.features)
    if not hasattr(model, "predict"):
        raise ValueError(f"Model '{manifest.model_id}' artifact is not a fitted estimator.")
    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None and int(n_features) != len(features):
        raise ValueError(
            f"Model '{manifest.model_id}' expects {n_features} features, but {len(features)} "
            "are declared."
        )
    classes = getattr(model, "classes_", None)
    if classes is not None and manifest.task == "fiber_identity":
        model_classes = [str(value) for value in classes]
        if sorted(model_classes) != sorted(manifest.outputs):
            raise ValueError(
                f"Model '{manifest.model_id}' predicts {model_classes}, but its manifest declares "
                f"outputs {list(manifest.outputs)}."
            )
    return model, features


def predict_semantic_candidate(
    diagnostics: pd.DataFrame, model_path: Path, manifest: ModelManifest
) -> pd.DataFrame:
    """Run a manifest-declared semantic model on per-fiber diagnostics."""
    model, features = load_semantic_model(model_path, manifest)
    missing = sorted(set(features) - set(diagnostics.columns))
    if missing:
        raise ValueError(f"Model '{manifest.model_id}' is missing features: {', '.join(missing)}.")
    inputs = diagnostics.loc[:, features]
    if getattr(model, "feature_names_in_", None) is not None:
        trained_names = [str(name) for name in model.feature_names_in_]
        if trained_names != features:
            raise ValueError(
                f"Model '{manifest.model_id}' was trained on features in a different order "
                "than declared."
            )
    else:
        # Estimators fitted on arrays take positional inputs in the declared order.
        inputs = inputs.to_numpy(dtype=float)
    finite = np.isfinite(np.asarray(inputs, dtype=float))
    if not finite.all():
        bad = int((~finite).any(axis=1).sum())
        raise ValueError(
            f"Model '{manifest.model_id}' received non-finite features for {bad} fibers."
        )
    prediction = np.asarray(model.predict(inputs)).astype(str)
    out = pd.DataFrame({"label": diagnostics["label"], "model_prediction": prediction})
    out["model_id"] = manifest.model_id
    out["task"] = manifest.task
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(inputs)
        for name, values in zip(model.classes_, proba.T, strict=True):
            out[f"prob_{name}"] = values
    return out


def apply_semantic_predictions(
    fibers: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    confidence_threshold: float,
    margin_threshold: float,
    classifier_path: str,
) -> pd.DataFrame:
    """Make semantic-model predictions the fiber calls in the fiber table.

    Replaces rule/legacy call columns with the model's class, per-class probabilities,
    confidence (top probability), margin (top minus second), and review flag.
    """
    merged = fibers.drop(
        columns=[column for column in fibers.columns if column.startswith("prob_")]
    ).merge(predictions, on="label", how="left", validate="one_to_one")
    if merged["model_prediction"].isna().any():
        raise ValueError("Semantic predictions are missing for some fibers.")
    prob_columns = [column for column in merged.columns if column.startswith("prob_")]
    merged["fiber_type"] = merged["model_prediction"].astype(str)
    if prob_columns:
        probabilities = np.sort(merged[prob_columns].to_numpy(dtype=float), axis=1)
        merged["model_confidence"] = probabilities[:, -1]
        merged["model_margin"] = (
            probabilities[:, -1] - probabilities[:, -2]
            if probabilities.shape[1] > 1
            else probabilities[:, -1]
        )
        merged["needs_review"] = (merged["model_confidence"] < confidence_threshold) | (
            merged["model_margin"] < margin_threshold
        )
    merged["classification_method"] = "semantic_model"
    merged["fiber_type_source"] = "model_prediction"
    merged["classifier_path"] = classifier_path
    return merged.drop(columns=["model_prediction", "task"])
