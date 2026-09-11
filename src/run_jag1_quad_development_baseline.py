"""Fit one development-only Jag1 Quad baseline without opening final-test mice.

This runner is intentionally narrow: it uses current reviewer decisions only,
excludes legacy supervision, and performs leave-one-development-mouse-out
diagnostics.  The resulting artifact is a development candidate, not a final
test result or a pipeline release model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tifffile
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

from src.io_utils import load_multichannel_image
from src.quantify_classify import QuantifyConfig, build_feature_diagnostics_table, quantify_labels

SEED = 20260910
ALLOWED_LABELS = {"i", "iia", "iib", "iix"}
FEATURES = (
    "area",
    "type_i.mean",
    "type_i.p75",
    "type_i.p90",
    "type_i.coverage_high",
    "type_i.snr_mean",
    "type_i.snr_p90",
    "type_iia.mean",
    "type_iia.p75",
    "type_iia.p90",
    "type_iia.coverage_high",
    "type_iia.snr_mean",
    "type_iia.snr_p90",
    "type_iib.mean",
    "type_iib.p75",
    "type_iib.p90",
    "type_iib.coverage_high",
    "type_iib.snr_mean",
    "type_iib.snr_p90",
)


def _model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        class_weight="balanced",
        max_features="sqrt",
        n_jobs=-1,
        random_state=SEED,
    )


def _load_manifest(path: Path, excluded_mouse: str) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype=str).fillna("")
    required = {
        "mouse_id",
        "cre_status",
        "image_id",
        "section_id",
        "split",
        "raw_image_path",
        "fiber_labels_path",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    development = manifest.loc[
        manifest["split"].eq("development") & ~manifest["mouse_id"].eq(excluded_mouse)
    ].copy()
    if development.empty:
        raise ValueError("No eligible development sections remain after exclusion")
    if development["split"].ne("development").any():
        raise AssertionError("Final-test sections must never enter the baseline")
    return development.sort_values(["mouse_id", "image_id"], kind="stable")


def _load_reviewed(paths: list[Path], eligible_images: set[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    required = {"mouse_id", "image_id", "section_id", "fiber_id", "sampling_stratum", "label"}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Review decisions not found: {path}")
        table = pd.read_csv(path, dtype={"fiber_id": "Int64"}).fillna("")
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"Review decisions missing columns in {path}: {missing}")
        table["review_source_path"] = str(path.resolve())
        frames.append(table)
    reviewed = pd.concat(frames, ignore_index=True)
    reviewed["label"] = reviewed["label"].astype(str).str.lower().str.strip()
    reviewed = reviewed.loc[
        reviewed["image_id"].astype(str).isin(eligible_images)
        & reviewed["label"].isin(ALLOWED_LABELS)
    ].copy()
    reviewed["fiber_id"] = pd.to_numeric(reviewed["fiber_id"], errors="raise").astype(int)
    duplicate = reviewed.duplicated(["image_id", "fiber_id"], keep=False)
    if duplicate.any():
        examples = reviewed.loc[duplicate, ["image_id", "fiber_id"]].head(5).to_dict("records")
        raise ValueError(f"Duplicate reviewer labels across sources: {examples}")
    if reviewed.empty:
        raise ValueError("No eligible current reviewer labels remain")
    return reviewed.sort_values(["mouse_id", "image_id", "fiber_id"], kind="stable")


def _section_features(section: pd.Series) -> pd.DataFrame:
    """Re-extract fixed observed-assay features from the reviewed segmentation."""
    raw = load_multichannel_image(Path(str(section.raw_image_path)))
    labels = np.asarray(tifffile.imread(str(section.fiber_labels_path)), dtype=np.int32)
    if raw.shape != (4, *labels.shape):
        raise ValueError(f"Expected four-channel raw image aligned to labels: {section.image_id}")
    quantified = quantify_labels(
        labels,
        raw,
        QuantifyConfig(
            type1_channel=3,  # Type IIb
            type2_channel=1,  # Type IIa
            i_channel=0,  # Type I
            typing_preprocess="global_subtract",
            typing_erode_px=2,
        ),
    )
    features = build_feature_diagnostics_table(
        quantified,
        QuantifyConfig(type1_channel=3, type2_channel=1, i_channel=0),
    )
    features.insert(0, "mouse_id", section.mouse_id)
    features.insert(1, "cre_status", section.cre_status)
    features.insert(2, "image_id", section.image_id)
    features.insert(3, "section_id", section.section_id)
    return features.rename(columns={"label": "pipeline_fiber_id"})


def _score(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, object]:
    actual = frame["label"].astype(str)
    return {
        "n": int(len(frame)),
        "accuracy": float(accuracy_score(actual, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, prediction)),
        "class_counts": {str(name): int(count) for name, count in actual.value_counts().items()},
    }


def _lomo_predictions(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    predictions: list[pd.DataFrame] = []
    summaries: dict[str, object] = {}
    for mouse in sorted(table["mouse_id"].unique()):
        train = table.loc[table["mouse_id"].ne(mouse)].copy()
        test = table.loc[table["mouse_id"].eq(mouse)].copy()
        if train["label"].nunique() != len(ALLOWED_LABELS):
            raise ValueError(f"Training fold without all four labels while holding out {mouse}")
        model = _model().fit(train.loc[:, FEATURES], train["label"])
        test["lomo_prediction"] = model.predict(test.loc[:, FEATURES])
        predictions.append(test)
        summaries[mouse] = _score(test, test["lomo_prediction"].to_numpy())
    out = pd.concat(predictions, ignore_index=True)
    random = out.loc[out["sampling_stratum"].eq("development_random")]
    summaries["all_reviewed"] = _score(out, out["lomo_prediction"].to_numpy())
    summaries["development_random"] = (
        _score(random, random["lomo_prediction"].to_numpy()) if not random.empty else None
    )
    for stratum, group in out.groupby("sampling_stratum", sort=True):
        summaries[f"stratum:{stratum}"] = _score(group, group["lomo_prediction"].to_numpy())
    return out, summaries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one conservative Jag1 Quad development baseline"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--review-decisions",
        type=Path,
        action="append",
        required=True,
        help="Current reviewer decision CSV; repeat for each current-review source.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-mouse", default="351545_R")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    manifest = _load_manifest(args.manifest.resolve(), args.exclude_mouse)
    reviewed = _load_reviewed(args.review_decisions, set(manifest["image_id"]))
    print(f"eligible development sections: {len(manifest)}", flush=True)
    print(f"eligible current reviewed labels: {len(reviewed)}", flush=True)
    feature_frames = []
    for number, (_, section) in enumerate(manifest.iterrows(), start=1):
        print(f"features {number}/{len(manifest)}: {section.image_id}", flush=True)
        feature_frames.append(_section_features(section))
    features = pd.concat(feature_frames, ignore_index=True)
    missing = sorted(set(FEATURES) - set(features.columns))
    if missing:
        raise ValueError(f"Feature extraction missing required columns: {missing}")
    joined = reviewed.merge(
        features,
        left_on=["mouse_id", "image_id", "section_id", "fiber_id"],
        right_on=["mouse_id", "image_id", "section_id", "pipeline_fiber_id"],
        how="left",
        validate="one_to_one",
    )
    if joined[list(FEATURES)].isna().any().any():
        raise ValueError("Some reviewed fibers lack finite extracted features")
    for column in FEATURES:
        joined[column] = pd.to_numeric(joined[column], errors="raise")
    if not np.isfinite(joined.loc[:, FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Some reviewed fibers have non-finite extracted features")
    predictions, metrics = _lomo_predictions(joined)
    final_model = _model().fit(joined.loc[:, FEATURES], joined["label"])
    args.output_dir.mkdir(parents=True)
    reviewed.to_csv(args.output_dir / "eligible_current_reviewed_labels.csv", index=False)
    joined.to_csv(args.output_dir / "development_training_rows.csv", index=False)
    predictions.to_csv(args.output_dir / "lomo_predictions.csv", index=False)
    joblib.dump(final_model, args.output_dir / "development_baseline_random_forest.joblib")
    metadata = {
        "purpose": "development_only_not_final_test_or_pipeline_release",
        "seed": SEED,
        "excluded_mouse": args.exclude_mouse,
        "legacy_supervision_used": False,
        "final_test_accessed": False,
        "review_sources": [str(path.resolve()) for path in args.review_decisions],
        "features": list(FEATURES),
        "model": {
            "family": "RandomForestClassifier",
            "n_estimators": 400,
            "min_samples_leaf": 3,
            "class_weight": "balanced",
        },
        "validation": "leave_one_development_mouse_out; development_random reported separately",
        "metrics": metrics,
        "classification_report_all_reviewed": classification_report(
            predictions["label"], predictions["lomo_prediction"], digits=3, zero_division=0
        ),
    }
    (args.output_dir / "baseline_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"saved development-only baseline: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
