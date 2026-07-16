"""Reclassify an existing batch from saved diagnostics and apply an IIa gate."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.analyze_iia_gate import (
    _apply_iia_gate,
    _gate_mask,
    _load_true_iia_reviewed,
    derive_iia_gate_thresholds,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an existing batch and reclassify saved feature diagnostics without "
            "rerunning segmentation, then apply the soft IIa gate."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--image-id",
        action="append",
        default=[],
        help="Optional image ID to copy and reclassify; repeat to select multiple images.",
    )
    parser.add_argument("--classifier-path", type=Path, required=True)
    parser.add_argument("--true-iia-reviewed-glob", type=str, required=True)
    parser.add_argument("--gate-quantile", type=float, default=0.01)
    return parser


def reclassify_and_gate(
    fibers: pd.DataFrame,
    diagnostics: pd.DataFrame,
    model: object,
    thresholds: dict[str, float],
    classifier_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace model calls using saved features, then demote unsupported IIa calls to IIx."""
    feature_columns = [str(column) for column in model.feature_names_in_]
    missing = sorted(set(feature_columns) - set(diagnostics.columns))
    if missing:
        raise ValueError(f"Diagnostics missing model features: {', '.join(missing)}")

    probabilities = np.asarray(model.predict_proba(diagnostics[feature_columns]), dtype=np.float32)
    classes = [str(label).lower() for label in model.classes_]
    predictions = pd.Series(model.predict(diagnostics[feature_columns]), index=diagnostics.index)
    predictions = predictions.astype(str).str.lower()
    probability_columns = {
        f"prob_{label}": probabilities[:, index]
        for index, label in enumerate(classes)
    }
    for label in ("iia", "iib", "iix"):
        probability_columns.setdefault(f"prob_{label}", np.nan)
    sorted_probabilities = np.sort(probabilities, axis=1)
    confidence = sorted_probabilities[:, -1]
    margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]

    diagnostics = diagnostics.copy()
    diagnostics["label"] = diagnostics["label"].astype(int)
    diagnostics["fiber_type"] = predictions
    diagnostics["fiber_type_source"] = "model_prediction"
    diagnostics["classification_method"] = "model"
    diagnostics["model_confidence"] = confidence
    diagnostics["model_margin"] = margin
    diagnostics["confidence"] = confidence
    diagnostics["classifier_path"] = str(classifier_path.resolve())
    for column, values in probability_columns.items():
        diagnostics[column] = values

    gate_ok = _gate_mask(diagnostics, thresholds)
    diagnostics["fiber_type_pre_iia_gate"] = predictions
    diagnostics["gate_iia_ok"] = gate_ok
    diagnostics["fiber_type"] = _apply_iia_gate(predictions, gate_ok)

    fibers = fibers.copy()
    fibers["label"] = fibers["label"].astype(int)
    update_columns = [
        "label",
        "fiber_type",
        "fiber_type_source",
        "classification_method",
        "model_confidence",
        "model_margin",
        "confidence",
        "classifier_path",
        "prob_iia",
        "prob_iib",
        "prob_iix",
        "fiber_type_pre_iia_gate",
        "gate_iia_ok",
    ]
    fibers = fibers.drop(
        columns=[column for column in update_columns if column != "label"],
        errors="ignore",
    )
    fibers = fibers.merge(
        diagnostics[update_columns],
        on="label",
        how="left",
        validate="one_to_one",
    )
    return fibers, diagnostics


def update_summary_paths(summary_path: Path, image_dir: Path, fibers: pd.DataFrame) -> None:
    summary = pd.read_csv(summary_path)
    if summary.empty:
        return
    stem = image_dir.name
    summary.loc[0, "fibers_path"] = str((image_dir / f"{stem}_fibers.csv").resolve())
    summary.loc[0, "feature_diagnostics_path"] = str(
        (image_dir / f"{stem}_feature_diagnostics.csv").resolve()
    )
    labels_path = image_dir / f"{stem}_cellpose_labels.tif"
    if labels_path.exists():
        summary.loc[0, "labels_path"] = str(labels_path.resolve())
    summary.loc[0, "n_fibers"] = float(len(fibers))
    counts = fibers["fiber_type"].value_counts()
    total = max(len(fibers), 1)
    summary.loc[0, "prop_type1"] = counts.get("iib", 0) / total
    summary.loc[0, "prop_type2"] = counts.get("iia", 0) / total
    summary.loc[0, "prop_unknown"] = counts.get("unknown", 0) / total
    summary.to_csv(summary_path, index=False)


def main() -> None:
    args = build_parser().parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")
    if args.output_dir.exists():
        raise SystemExit(f"Output directory already exists: {args.output_dir}")
    if not args.classifier_path.is_file():
        raise SystemExit(f"Classifier does not exist: {args.classifier_path}")

    model = joblib.load(args.classifier_path)
    true_iia = _load_true_iia_reviewed(args.true_iia_reviewed_glob)
    thresholds = derive_iia_gate_thresholds(true_iia, args.gate_quantile)
    selected_ids = [str(image_id).strip() for image_id in args.image_id if str(image_id).strip()]
    if selected_ids:
        args.output_dir.mkdir(parents=True)
        for image_id in selected_ids:
            source = args.input_dir / image_id
            if not source.is_dir():
                raise SystemExit(f"Image output directory does not exist: {source}")
            shutil.copytree(source, args.output_dir / image_id)
    else:
        shutil.copytree(args.input_dir, args.output_dir)

    image_dirs = sorted(path for path in args.output_dir.iterdir() if path.is_dir())
    results: list[dict[str, object]] = []
    for index, image_dir in enumerate(image_dirs, start=1):
        print(f"[{index}/{len(image_dirs)}] Reclassifying {image_dir.name}")
        fibers_paths = list(image_dir.glob("*_fibers.csv"))
        diagnostics_paths = list(image_dir.glob("*_feature_diagnostics.csv"))
        summary_paths = list(image_dir.glob("*_summary.csv"))
        if len(fibers_paths) != 1 or len(diagnostics_paths) != 1:
            results.append({"image_name": image_dir.name, "status": "skipped"})
            continue
        fibers, diagnostics = reclassify_and_gate(
            pd.read_csv(fibers_paths[0], low_memory=False),
            pd.read_csv(diagnostics_paths[0], low_memory=False),
            model,
            thresholds,
            args.classifier_path,
        )
        fibers.to_csv(fibers_paths[0], index=False)
        diagnostics.to_csv(diagnostics_paths[0], index=False)
        if len(summary_paths) == 1:
            update_summary_paths(summary_paths[0], image_dir, fibers)
        results.append({"image_name": image_dir.name, "status": "success", "n_fibers": len(fibers)})

    pd.DataFrame(results).to_csv(args.output_dir / "reclassify_batch_summary.csv", index=False)
    print(f"saved reclassified batch: {args.output_dir}")


if __name__ == "__main__":
    main()
