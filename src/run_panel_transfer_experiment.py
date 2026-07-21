"""Exploratory Type I/IIa transfer comparison across verified panel domains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.train_panel_baselines import (
    _fit,
    _manual_table,
    _metrics,
    _numeric_features,
    baseline_models,
    semantic_feature_columns,
)


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description="Run exploratory panel-transfer baselines.")
    for role in ("quad-train", "quad-heldout", "vivienne-train", "vivienne-heldout"):
        out.add_argument(f"--{role}-reviewed", type=Path, required=True)
        out.add_argument(f"--{role}-diagnostics", type=Path, required=True)
    out.add_argument("--output-dir", type=Path, required=True)
    return out


def _load(reviewed: Path, diagnostics: Path) -> pd.DataFrame:
    return _manual_table(
        pd.read_csv(reviewed, low_memory=False),
        pd.read_csv(diagnostics, low_memory=False),
        ("i", "iia"),
    )


def main() -> None:
    args = parser().parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output directory already exists: {args.output_dir}")
    quad_train = _load(args.quad_train_reviewed, args.quad_train_diagnostics)
    quad_heldout = _load(args.quad_heldout_reviewed, args.quad_heldout_diagnostics)
    viv_train = _load(args.vivienne_train_reviewed, args.vivienne_train_diagnostics)
    viv_heldout = _load(args.vivienne_heldout_reviewed, args.vivienne_heldout_diagnostics)
    datasets = {
        "quad_train": quad_train,
        "quad_heldout": quad_heldout,
        "vivienne_train": viv_train,
        "vivienne_heldout": viv_heldout,
    }
    features = semantic_feature_columns(quad_train, quad_heldout)
    for frame in (viv_train, viv_heldout):
        features = [column for column in features if column in frame.columns]
    if not features:
        raise ValueError("No semantic features are shared by every panel/domain dataset.")
    x = {name: _numeric_features(frame, features) for name, frame in datasets.items()}
    y = {name: frame["final_type"] for name, frame in datasets.items()}
    experiments = {
        "quad_to_quad": ("quad_train", "quad_heldout"),
        "vivienne_to_vivienne": ("vivienne_train", "vivienne_heldout"),
        "pooled_to_quad": (("quad_train", "vivienne_train"), "quad_heldout"),
        "pooled_to_vivienne": (("quad_train", "vivienne_train"), "vivienne_heldout"),
        "quad_to_vivienne": ("quad_train", "vivienne_heldout"),
        "vivienne_to_quad": ("vivienne_train", "quad_heldout"),
    }
    report: dict[str, object] = {
        "purpose": "exploratory transfer diagnostic; not model selection",
        "classes": ["i", "iia"],
        "features": features,
        "label_counts": {name: values.value_counts().to_dict() for name, values in y.items()},
        "experiments": {},
    }
    for name, (train_names, test_name) in experiments.items():
        if isinstance(train_names, str):
            train_names = (train_names,)
        train_x = pd.concat([x[item] for item in train_names], ignore_index=True)
        train_y = pd.concat([y[item] for item in train_names], ignore_index=True)
        result: dict[str, object] = {}
        for model_name, model in baseline_models().items():
            fitted = _fit(model, train_x, train_y)
            prediction = fitted.predict(x[test_name]).astype(str)
            result[model_name] = _metrics(y[test_name], prediction, ("i", "iia"))
        report["experiments"][name] = result
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "transfer_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Exploratory transfer report written to: {args.output_dir / 'transfer_report.json'}")


if __name__ == "__main__":
    main()
