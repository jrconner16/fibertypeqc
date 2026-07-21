"""Explicit inference for semantic, panel-aware candidate models."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from fibertypeqc.model_manifest import ModelManifest


def predict_semantic_candidate(
    diagnostics: pd.DataFrame, model_path: Path, manifest: ModelManifest
) -> pd.DataFrame:
    """Run a manifest-declared semantic candidate without altering legacy calls."""
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        raise ValueError("Semantic candidate must be a joblib bundle with 'model' and 'features'.")
    features = list(bundle["features"])
    missing = sorted(set(features) - set(diagnostics.columns))
    if missing:
        raise ValueError(f"Model '{manifest.model_id}' is missing features: {', '.join(missing)}.")
    model = bundle["model"]
    prediction = model.predict(diagnostics.loc[:, features]).astype(str)
    out = pd.DataFrame({"label": diagnostics["label"], "model_prediction": prediction})
    out["model_id"] = manifest.model_id
    out["task"] = manifest.task
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(diagnostics.loc[:, features])
        for name, values in zip(model.classes_, proba.T, strict=False):
            out[f"prob_{name}"] = values
    return out
