"""Development-only Type I quadriceps smoke experiment.

This runner is deliberately not connected to ``run_pipeline``.  It derives a
direct-IIa evidence gate from manual IIa labels, then classifies the remaining
manual labels with a small random forest.  Cross-validation is fiber-level and
therefore diagnostic only; it is not a biological held-out evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

SMOKE_VERSION = "quad_type_i_smoke.v1"
GATE_FEATURES = (
    "type_iia.snr_mean",
    "type_iia.coverage_high",
    "type_iia.p90",
)
CLASSIFIER_FEATURES = (
    "type_i.mean",
    "type_i.p75",
    "type_i.p90",
    "type_i.coverage_high",
    "type_i.snr_mean",
    "type_i.snr_p90",
    "type_iia.mean",
    "type_iia.p90",
    "type_iia.coverage_high",
    "type_iia.snr_mean",
    "type_iib.mean",
    "type_iib.p90",
    "type_iib.coverage_high",
    "type_iib.snr_mean",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a development-only QUAD Type I smoke experiment. It is not a "
            "pipeline model and must not be used for biological performance claims."
        )
    )
    parser.add_argument("--reviewed-fibers", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--gate-quantile",
        type=float,
        default=0.10,
        help="Lower manual-IIa quantile used for each direct-IIa gate threshold.",
    )
    parser.add_argument(
        "--include-provisional-iix",
        action="store_true",
        help="Include manually labeled IIx in the non-IIa classifier despite provisional status.",
    )
    return parser


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _manual_training_table(
    reviewed_fibers: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    include_provisional_iix: bool,
) -> pd.DataFrame:
    _require_columns(reviewed_fibers, ("label", "final_type", "label_source"), "reviewed fibers")
    _require_columns(diagnostics, ("label", *GATE_FEATURES, *CLASSIFIER_FEATURES), "diagnostics")
    labels = reviewed_fibers.loc[
        reviewed_fibers["label_source"].fillna("").eq("manual_gold"),
        ["label", "final_type"],
    ].copy()
    allowed = {"i", "iia", "iib"}
    if include_provisional_iix:
        allowed.add("iix")
    labels["final_type"] = labels["final_type"].fillna("").astype(str).str.lower().str.strip()
    labels = labels.loc[labels["final_type"].isin(allowed)].copy()
    if labels.empty:
        raise ValueError("No eligible manual labels found.")
    if labels["label"].duplicated().any():
        raise ValueError("Manual labels contain duplicate fiber labels.")
    table = labels.merge(diagnostics, on="label", how="inner", validate="one_to_one")
    if len(table) != len(labels):
        raise ValueError("Some manual labels are missing from diagnostics.")
    numeric = [*GATE_FEATURES, *CLASSIFIER_FEATURES]
    for column in numeric:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if table[numeric].isna().any().any():
        raise ValueError("Manual training rows contain missing semantic marker features.")
    return table.reset_index(drop=True)


def derive_iia_gate(manual_iia: pd.DataFrame, quantile: float) -> dict[str, float]:
    if not 0 < quantile < 0.5:
        raise ValueError("gate quantile must be between 0 and 0.5.")
    if manual_iia.empty:
        raise ValueError("At least one manual IIa label is required for the IIa gate.")
    return {column: float(manual_iia[column].quantile(quantile)) for column in GATE_FEATURES}


def iia_gate_mask(table: pd.DataFrame, thresholds: dict[str, float]) -> pd.Series:
    mask = pd.Series(True, index=table.index)
    for column, threshold in thresholds.items():
        mask &= pd.to_numeric(table[column], errors="coerce").ge(threshold)
    return mask.fillna(False)


def _classifier() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        class_weight="balanced",
        max_features="sqrt",
        n_jobs=-1,
        random_state=0,
    )


def cross_validated_smoke(table: pd.DataFrame, gate_quantile: float) -> pd.DataFrame:
    class_counts = table["final_type"].value_counts()
    folds = min(5, int(class_counts.min()))
    if folds < 2:
        raise ValueError(
            "Each included class needs at least two labels for smoke cross-validation."
        )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    predictions: list[pd.DataFrame] = []
    for fold, (train_index, test_index) in enumerate(
        splitter.split(table, table["final_type"]), start=1
    ):
        train = table.iloc[train_index]
        test = table.iloc[test_index].copy()
        thresholds = derive_iia_gate(train.loc[train["final_type"].eq("iia")], gate_quantile)
        non_iia_train = train.loc[~train["final_type"].eq("iia")]
        model = _classifier()
        model.fit(non_iia_train[list(CLASSIFIER_FEATURES)], non_iia_train["final_type"])
        gate_ok = iia_gate_mask(test, thresholds)
        prediction = pd.Series(index=test.index, dtype="object")
        prediction.loc[gate_ok] = "iia"
        prediction.loc[~gate_ok] = model.predict(test.loc[~gate_ok, list(CLASSIFIER_FEATURES)])
        test["fold"] = fold
        test["iia_gate_passed"] = gate_ok.to_numpy()
        test["smoke_prediction"] = prediction.astype(str).to_numpy()
        predictions.append(test)
    return pd.concat(predictions, ignore_index=True)


def train_final_smoke(table: pd.DataFrame, gate_quantile: float) -> dict[str, object]:
    thresholds = derive_iia_gate(table.loc[table["final_type"].eq("iia")], gate_quantile)
    non_iia = table.loc[~table["final_type"].eq("iia")]
    model = _classifier()
    model.fit(non_iia[list(CLASSIFIER_FEATURES)], non_iia["final_type"])
    return {
        "smoke_version": SMOKE_VERSION,
        "gate_quantile": gate_quantile,
        "iia_gate_thresholds": thresholds,
        "classifier_features": list(CLASSIFIER_FEATURES),
        "non_iia_classifier": model,
    }


def smoke_metrics(predictions: pd.DataFrame) -> dict[str, object]:
    actual = predictions["final_type"].astype(str)
    predicted = predictions["smoke_prediction"].astype(str)
    labels = sorted(actual.unique())
    return {
        "smoke_version": SMOKE_VERSION,
        "evaluation": "fiber-level cross-validation within one development image; not held-out",
        "n_manual_labels": int(len(predictions)),
        "class_counts": {str(key): int(value) for key, value in actual.value_counts().items()},
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output directory already exists: {args.output_dir}")
    reviewed = pd.read_csv(args.reviewed_fibers, low_memory=False)
    diagnostics = pd.read_csv(args.diagnostics, low_memory=False)
    table = _manual_training_table(
        reviewed,
        diagnostics,
        include_provisional_iix=bool(args.include_provisional_iix),
    )
    predictions = cross_validated_smoke(table, args.gate_quantile)
    metrics = smoke_metrics(predictions)
    model_bundle = train_final_smoke(table, args.gate_quantile)

    args.output_dir.mkdir(parents=True)
    table.to_csv(args.output_dir / "manual_training_rows.csv", index=False)
    predictions.to_csv(args.output_dir / "cross_validated_predictions.csv", index=False)
    (args.output_dir / "smoke_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output_dir / "model_metadata.json").write_text(
        json.dumps(
            {key: value for key, value in model_bundle.items() if key != "non_iia_classifier"},
            indent=2,
        )
        + "\n"
    )
    joblib.dump(model_bundle, args.output_dir / "quad_type_i_smoke.joblib")

    print("DEVELOPMENT-ONLY: fiber-level CV within one image; not a held-out result.")
    print(json.dumps(metrics, indent=2))
    print(f"saved smoke outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
