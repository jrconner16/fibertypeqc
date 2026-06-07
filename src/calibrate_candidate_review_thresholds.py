from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from src.analyze_iia_gate import (
    _apply_iia_gate,
    _gate_mask,
    derive_iia_gate_thresholds,
)
from src.evaluate_against_audit_benchmark import CANONICAL_CLASSES
from src.quantify_classify import FROZEN_ALPHA_BASELINE_FEATURES
from src.train_candidate_from_feature_table import _expanded_feature_columns, _load_feature_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate candidate-specific needs_review thresholds against the protected "
            "manual eval holdout."
        )
    )
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--benchmark-split", type=Path, required=True)
    parser.add_argument("--classifier-path", type=Path, required=True)
    parser.add_argument("--true-iia-reviewed-glob", type=str, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument(
        "--feature-set",
        type=str,
        default="baseline",
        choices=["baseline", "expanded"],
        help="Feature set used by the candidate model artifact.",
    )
    parser.add_argument(
        "--gate-quantile",
        type=float,
        default=0.01,
        help="IIa gate quantile for the current candidate pipeline.",
    )
    parser.add_argument(
        "--confidence-thresholds",
        type=str,
        default="0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
        help="Comma-separated candidate confidence thresholds to evaluate.",
    )
    parser.add_argument(
        "--margin-thresholds",
        type=str,
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50",
        help="Comma-separated candidate margin thresholds to evaluate.",
    )
    return parser


def _parse_thresholds(raw: str, *, low: float, high: float, name: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        value = float(text)
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"At least one {name} is required.")
    return values


def _load_benchmark_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {"image_id", "label", "audit_final_label", "manual_supervision_split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    out = df.copy()
    out["image_id"] = out["image_id"].astype(str)
    out["label"] = out["label"].astype(int)
    out["audit_final_label"] = (
        out["audit_final_label"].fillna("").astype(str).str.lower().str.strip()
    )
    out["manual_supervision_split"] = (
        out["manual_supervision_split"].fillna("").astype(str).str.lower().str.strip()
    )
    return out


def _load_true_iia_reviewed(pattern: str) -> pd.DataFrame:
    files = sorted(Path().glob(pattern))
    if not files:
        raise ValueError(f"No true_iia_hunt reviewed files matched: {pattern}")
    frames = [pd.read_csv(path, low_memory=False) for path in files]
    out = pd.concat(frames, ignore_index=True)
    out["audit_corrected_type"] = (
        out.get("audit_corrected_type", pd.Series("", index=out.index))
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )
    out = out.loc[out["audit_corrected_type"].eq("iia")].copy()
    if out.empty:
        raise ValueError("No confirmed IIa rows found in true_iia_hunt reviewed files.")
    return out


def _feature_columns(feature_table: pd.DataFrame, feature_set: str) -> list[str]:
    if feature_set == "baseline":
        return list(FROZEN_ALPHA_BASELINE_FEATURES)
    if feature_set == "expanded":
        return _expanded_feature_columns(feature_table)
    raise ValueError(f"Unsupported feature_set: {feature_set}")


def _candidate_probs(model: object, x: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    classes = [str(c).lower() for c in model.classes_]
    proba = np.asarray(model.predict_proba(x), dtype=np.float64)
    order = np.argsort(proba, axis=1)[:, ::-1]
    top1 = proba[np.arange(len(proba)), order[:, 0]]
    top2 = proba[np.arange(len(proba)), order[:, 1]]
    pred = np.array(classes, dtype=object)[order[:, 0]]
    return (
        pd.Series(pred.astype(str)),
        pd.Series(top1.astype(float)),
        pd.Series((top1 - top2).astype(float)),
    )


def _review_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    review_flag: pd.Series,
) -> dict[str, float]:
    y_true = y_true.astype(str).str.lower()
    y_pred = y_pred.astype(str).str.lower()
    review_flag = review_flag.fillna(False).astype(bool)
    errors = ~y_true.eq(y_pred)
    flagged_errors = review_flag & errors
    unflagged = ~review_flag
    unflagged_errors = unflagged & errors
    total = len(y_true)
    total_errors = int(errors.sum())
    return {
        "n_rows": int(total),
        "review_rate": float(review_flag.mean()) if total else 0.0,
        "review_n": int(review_flag.sum()),
        "total_errors": total_errors,
        "error_capture_rate": float(flagged_errors.sum() / total_errors) if total_errors else 0.0,
        "unflagged_error_rate": float(unflagged_errors.sum() / max(int(unflagged.sum()), 1)),
        "unflagged_accuracy": float(
            accuracy_score(y_true.loc[unflagged], y_pred.loc[unflagged])
        )
        if int(unflagged.sum())
        else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def _choose_recommended_threshold(metrics: pd.DataFrame) -> pd.Series:
    frozen = metrics.loc[metrics["strategy"].eq("pipeline_current")]
    if frozen.empty:
        raise ValueError("Missing pipeline_current row in calibration metrics.")
    frozen_row = frozen.iloc[0]
    candidate = metrics.loc[metrics["strategy"].eq("candidate_gated_sweep")].copy()
    within_budget = candidate.loc[
        candidate["review_rate"].le(float(frozen_row["review_rate"]))
    ].copy()
    if within_budget.empty:
        within_budget = candidate.copy()
    within_budget = within_budget.sort_values(
        [
            "error_capture_rate",
            "review_rate",
            "unflagged_error_rate",
            "balanced_accuracy",
            "accuracy",
            "confidence_threshold",
            "margin_threshold",
        ],
        ascending=[False, True, True, False, False, True, True],
        kind="stable",
    )
    return within_budget.iloc[0]


def calibrate_candidate_review_thresholds(
    feature_table: pd.DataFrame,
    benchmark_split: pd.DataFrame,
    model: object,
    true_iia_reviewed: pd.DataFrame,
    *,
    feature_set: str,
    gate_quantile: float,
    confidence_thresholds: list[float],
    margin_thresholds: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    holdout = benchmark_split.loc[
        benchmark_split["manual_supervision_split"].eq("manual_eval_holdout")
        & benchmark_split["audit_final_label"].isin(CANONICAL_CLASSES)
    ].copy()
    if holdout.empty:
        raise ValueError("No manual_eval_holdout rows available for calibration.")

    feature_columns = _feature_columns(feature_table, feature_set)
    eval_metadata_cols = [
        "split",
        "fiber_type",
        "needs_review",
        "model_confidence",
        "model_margin",
    ]
    feature_merge_cols = [
        "image_id",
        "label",
        *feature_columns,
        "type1_mean",
        "type2_mean",
        "type1_coverage",
        "type2_coverage",
        "type1_cov_x_snr",
        "type1_snr_mean",
        "type2_cov_x_snr",
        "type2_snr_mean",
    ]
    holdout_base = holdout[
        [
            c
            for c in [
                "image_id",
                "label",
                "audit_final_label",
                "manual_supervision_split",
                *eval_metadata_cols,
            ]
            if c in holdout.columns
        ]
    ].copy()
    merged = holdout_base.merge(
        feature_table[[c for c in dict.fromkeys(feature_merge_cols) if c in feature_table.columns]],
        on=["image_id", "label"],
        how="left",
        validate="one_to_one",
    )
    valid = merged[feature_columns].notna().all(axis=1)
    if not valid.all():
        missing = int((~valid).sum())
        raise ValueError(f"{missing} holdout rows are missing candidate features.")

    pred, conf, margin = _candidate_probs(model, merged[feature_columns])
    thresholds = derive_iia_gate_thresholds(true_iia_reviewed, gate_quantile)
    gate_ok = _gate_mask(merged, thresholds)
    gated_pred = _apply_iia_gate(pred, gate_ok)

    merged = merged.copy()
    merged["candidate_pred_raw"] = pred.to_numpy()
    merged["candidate_pred_gated"] = gated_pred.to_numpy()
    merged["candidate_model_confidence"] = conf.to_numpy()
    merged["candidate_model_margin"] = margin.to_numpy()
    merged["candidate_gate_iia_ok"] = gate_ok.to_numpy()

    y_true = merged["audit_final_label"].astype(str).str.lower()
    metric_rows: list[dict[str, object]] = []

    frozen_metrics = _review_metrics(
        y_true,
        merged["fiber_type"].astype(str).str.lower(),
        merged["needs_review"].fillna(False).astype(bool),
    )
    metric_rows.append(
        {
            "strategy": "pipeline_current",
            "confidence_threshold": np.nan,
            "margin_threshold": np.nan,
            **frozen_metrics,
        }
    )

    default_review = (
        merged["candidate_model_confidence"].lt(0.55)
        | merged["candidate_model_margin"].lt(0.15)
    )
    default_metrics = _review_metrics(y_true, merged["candidate_pred_gated"], default_review)
    metric_rows.append(
        {
            "strategy": "candidate_gated_default",
            "confidence_threshold": 0.55,
            "margin_threshold": 0.15,
            **default_metrics,
        }
    )

    for conf_thr in confidence_thresholds:
        for margin_thr in margin_thresholds:
            review_flag = (
                merged["candidate_model_confidence"].lt(conf_thr)
                | merged["candidate_model_margin"].lt(margin_thr)
            )
            metrics = _review_metrics(y_true, merged["candidate_pred_gated"], review_flag)
            metric_rows.append(
                {
                    "strategy": "candidate_gated_sweep",
                    "confidence_threshold": float(conf_thr),
                    "margin_threshold": float(margin_thr),
                    **metrics,
                }
            )

    metrics = pd.DataFrame(metric_rows)
    recommended = _choose_recommended_threshold(metrics)

    recommended_flag = (
        merged["candidate_model_confidence"].lt(float(recommended["confidence_threshold"]))
        | merged["candidate_model_margin"].lt(float(recommended["margin_threshold"]))
    )
    merged["pipeline_error"] = ~y_true.eq(merged["fiber_type"].astype(str).str.lower())
    merged["pipeline_needs_review"] = merged["needs_review"].fillna(False).astype(bool)
    merged["candidate_error"] = ~y_true.eq(merged["candidate_pred_gated"].astype(str).str.lower())
    merged["candidate_needs_review_default"] = default_review.astype(bool)
    merged["candidate_needs_review_recommended"] = recommended_flag.astype(bool)

    report_lines = [
        "Candidate needs_review threshold calibration on manual_eval_holdout",
        "",
        f"feature_set={feature_set}",
        f"gate_quantile={gate_quantile:.4f}",
        "",
        "Reference rows:",
        metrics.loc[
            metrics["strategy"].isin(["pipeline_current", "candidate_gated_default"])
        ].to_string(index=False),
        "",
        "Recommended candidate thresholds:",
        (
            f"confidence_threshold={float(recommended['confidence_threshold']):.2f}, "
            f"margin_threshold={float(recommended['margin_threshold']):.2f}"
        ),
        (
            f"review_rate={float(recommended['review_rate']):.4f}, "
            f"error_capture_rate={float(recommended['error_capture_rate']):.4f}, "
            f"unflagged_error_rate={float(recommended['unflagged_error_rate']):.4f}, "
            f"unflagged_accuracy={float(recommended['unflagged_accuracy']):.4f}"
        ),
        "",
        "Top candidate sweep rows (sorted by review_rate then unflagged_error_rate):",
        metrics.loc[metrics["strategy"].eq("candidate_gated_sweep")]
        .sort_values(
            ["review_rate", "unflagged_error_rate", "balanced_accuracy", "accuracy"],
            ascending=[True, True, False, False],
            kind="stable",
        )
        .head(15)
        .to_string(index=False),
    ]
    return metrics, merged, "\n".join(report_lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    feature_table = _load_feature_table(args.feature_table)
    benchmark_split = _load_benchmark_split(args.benchmark_split)
    model = joblib.load(args.classifier_path)
    true_iia_reviewed = _load_true_iia_reviewed(args.true_iia_reviewed_glob)
    confidence_thresholds = _parse_thresholds(
        args.confidence_thresholds,
        low=0.0,
        high=1.0,
        name="confidence threshold",
    )
    margin_thresholds = _parse_thresholds(
        args.margin_thresholds,
        low=0.0,
        high=1.0,
        name="margin threshold",
    )
    metrics, predictions, report = calibrate_candidate_review_thresholds(
        feature_table,
        benchmark_split,
        model,
        true_iia_reviewed,
        feature_set=args.feature_set,
        gate_quantile=args.gate_quantile,
        confidence_thresholds=confidence_thresholds,
        margin_thresholds=margin_thresholds,
    )

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(args.metrics_output, index=False)
    predictions.to_csv(args.predictions_output, index=False)
    args.report_output.write_text(report, encoding="utf-8")

    print(f"saved metrics: {args.metrics_output}")
    print(f"saved predictions: {args.predictions_output}")
    print(f"saved report: {args.report_output}")


if __name__ == "__main__":
    main()
