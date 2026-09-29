"""Evaluate the frozen soft IIa gate on Notch3 leave-one-mouse-out predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support

CLASSES = ["iia", "iix", "iib"]


def _features(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    required = [
        "type_iia.mean",
        "type_iib.mean",
        "type_iia.snr_mean",
        "type_iib.snr_mean",
        "type_iia.coverage_high",
        "type_iib.coverage_high",
    ]
    missing = sorted(set(required).difference(out.columns))
    if missing:
        raise ValueError(f"Feature table missing IIa-gate columns: {missing}")
    for column in required:
        out[column] = pd.to_numeric(out[column], errors="raise")
    out["type_iia.cov_x_snr"] = out["type_iia.coverage_high"] * out["type_iia.snr_mean"]
    out["type_iib.cov_x_snr"] = out["type_iib.coverage_high"] * out["type_iib.snr_mean"]
    return out


def _thresholds(training: pd.DataFrame, quantile: float) -> dict[str, float]:
    true_iia = training.loc[training["target_label"].eq("iia")]
    if true_iia.empty:
        raise ValueError("A training fold has no true IIa rows.")
    upper = 1.0 - quantile
    return {
        "min_iia_snr": float(true_iia["type_iia.snr_mean"].quantile(quantile)),
        "min_iia_coverage": float(true_iia["type_iia.coverage_high"].quantile(quantile)),
        "min_iia_cov_x_snr": float(true_iia["type_iia.cov_x_snr"].quantile(quantile)),
        "max_iib_snr": float(true_iia["type_iib.snr_mean"].quantile(upper)),
        "max_iib_coverage": float(true_iia["type_iib.coverage_high"].quantile(upper)),
    }


def _gate_ok(table: pd.DataFrame, thresholds: dict[str, float]) -> pd.Series:
    return (
        table["type_iia.snr_mean"].ge(thresholds["min_iia_snr"])
        & table["type_iia.coverage_high"].ge(thresholds["min_iia_coverage"])
        & table["type_iia.cov_x_snr"].ge(thresholds["min_iia_cov_x_snr"])
        & table["type_iib.snr_mean"].le(thresholds["max_iib_snr"])
        & table["type_iib.coverage_high"].le(thresholds["max_iib_coverage"])
        & table["type_iia.mean"].gt(table["type_iib.mean"])
    )


def _metric_row(
    actual: pd.Series, predicted: pd.Series, *, strategy: str
) -> dict[str, float | str]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=CLASSES, zero_division=0
    )
    result: dict[str, float | str] = {
        "strategy": strategy,
        "macro_f1": float(
            f1_score(actual, predicted, labels=CLASSES, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
    }
    for index, label in enumerate(CLASSES):
        result[f"{label}_precision"] = float(precision[index])
        result[f"{label}_recall"] = float(recall[index])
        result[f"{label}_f1"] = float(f1[index])
    return result


def evaluate(
    features: pd.DataFrame, oof_predictions: pd.DataFrame, quantile: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["image_id", "group_id", "pipeline_label_id", "target_label"]
    predictions = oof_predictions.loc[
        (oof_predictions["condition"].eq("notch3_only"))
        & (oof_predictions["evaluation_scope"].eq("grouped_resampling_notch3"))
    ].copy()
    if len(predictions) != len(features):
        raise ValueError("OOF prediction row count does not equal the development feature table.")
    oof_columns = {
        "predicted_label": "oof_predicted_label",
        "model_confidence": "oof_model_confidence",
        "model_margin": "oof_model_margin",
    }
    joined = features.merge(
        predictions.loc[:, [*keys, *oof_columns]].rename(columns=oof_columns),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(features):
        raise ValueError("Feature/OOF join changed row count.")
    joined = _features(joined)
    threshold_rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    for group_id, test in joined.groupby("group_id", sort=True):
        train = joined.loc[~joined["group_id"].eq(group_id)]
        thresholds = _thresholds(train, quantile)
        gate_ok = _gate_ok(test, thresholds)
        frame = test.loc[
            :, keys + ["oof_predicted_label", "oof_model_confidence", "oof_model_margin"]
        ].rename(columns={value: key for key, value in oof_columns.items()})
        frame["gate_iia_ok"] = gate_ok.to_numpy()
        frame["gated_predicted_label"] = frame["predicted_label"]
        rejected = frame["predicted_label"].eq("iia") & ~frame["gate_iia_ok"]
        frame.loc[rejected, "gated_predicted_label"] = "iix"
        frame["base_review"] = (frame["model_confidence"] < 0.70) | (frame["model_margin"] < 0.25)
        frame["gate_rejected_iia"] = rejected
        frame["review_with_gate_rejection"] = frame["base_review"] | rejected
        frames.append(frame)
        threshold_rows.append({"heldout_group_id": group_id, **thresholds})
    output = pd.concat(frames, ignore_index=True)
    metric_rows: list[dict[str, object]] = []
    for scope_type, scope_id, subset in [
        ("overall", "all", output),
        *[("group", group, group_df) for group, group_df in output.groupby("group_id", sort=True)],
    ]:
        for strategy, column in [
            ("raw_logistic", "predicted_label"),
            ("soft_iia_gate_q0.01", "gated_predicted_label"),
        ]:
            metric_rows.append(
                {
                    "scope_type": scope_type,
                    "scope_id": scope_id,
                    "rows": len(subset),
                    **_metric_row(subset["target_label"], subset[column], strategy=strategy),
                    "base_review_rate": float(subset["base_review"].mean()),
                    "gate_rejected_iia_rate": float(subset["gate_rejected_iia"].mean()),
                    "review_with_gate_rejection_rate": float(
                        subset["review_with_gate_rejection"].mean()
                    ),
                }
            )
    return output, pd.DataFrame(metric_rows), pd.DataFrame(threshold_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-features", type=Path, required=True)
    parser.add_argument("--comparison-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate-quantile", type=float, default=0.01)
    args = parser.parse_args()
    if not 0.0 < args.gate_quantile < 0.5:
        raise ValueError("--gate-quantile must be between 0 and 0.5.")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    predictions, metrics, thresholds = evaluate(
        pd.read_csv(args.development_features, low_memory=False),
        pd.read_csv(args.comparison_predictions, low_memory=False),
        args.gate_quantile,
    )
    args.output_dir.mkdir(parents=True)
    predictions.to_csv(args.output_dir / "oof_gate_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    thresholds.to_csv(args.output_dir / "fold_thresholds.csv", index=False)
    provenance = {
        "gate_quantile": args.gate_quantile,
        "protected_rows_read": 0,
        "model_refit": False,
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"OOF rows: {len(predictions)}")
    print(f"fold thresholds: {len(thresholds)}")


if __name__ == "__main__":
    main()
