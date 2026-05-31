"""Confidence and probability-weighted composition helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

PROBABILITY_COLUMNS = ("prob_iib", "prob_iia", "prob_iix")


def normalized_entropy(
    probabilities: pd.DataFrame | np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return entropy normalized to 0..1 for class-probability rows."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probabilities must be a 2D array or DataFrame")
    row_sum = probs.sum(axis=1, keepdims=True)
    probs = np.divide(probs, row_sum, out=np.zeros_like(probs), where=row_sum > 0)
    entropy = -(probs * np.log(np.clip(probs, eps, 1.0))).sum(axis=1)
    return entropy / np.log(probs.shape[1])


def add_probability_metrics(
    fibers: pd.DataFrame,
    probability_columns: tuple[str, ...] = PROBABILITY_COLUMNS,
) -> pd.DataFrame:
    """Add normalized entropy from model probability columns."""
    missing = [col for col in probability_columns if col not in fibers.columns]
    if missing:
        raise ValueError(f"Missing probability columns: {missing}")
    out = fibers.copy()
    out["model_entropy_norm"] = normalized_entropy(out[list(probability_columns)])
    return out


def soft_type_composition(
    fibers: pd.DataFrame,
    probability_columns: tuple[str, ...] = PROBABILITY_COLUMNS,
) -> pd.Series:
    """Return probability-weighted fiber-type composition as percentages."""
    missing = [col for col in probability_columns if col not in fibers.columns]
    if missing:
        raise ValueError(f"Missing probability columns: {missing}")
    probs = fibers[list(probability_columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    total = float(len(probs))
    if total == 0:
        return pd.Series({col.replace("prob_", ""): np.nan for col in probability_columns})
    return probs.sum().rename(lambda col: col.replace("prob_", "")) / total * 100.0

