from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

from src.evaluate_against_audit_benchmark import CANONICAL_CLASSES
from src.train_candidate_from_feature_table import _load_feature_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare current pipeline, weighted model, and strict evidence-gated IIa "
            "strategies against the manual eval holdout."
        )
    )
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--benchmark-split", type=Path, required=True)
    parser.add_argument("--weighted-predictions", type=Path, required=True)
    parser.add_argument("--true-iia-reviewed-glob", type=str, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument(
        "--myosight-summary",
        type=Path,
        default=None,
        help="Optional consolidated MyoSight image summary for secondary image-level comparison.",
    )
    parser.add_argument(
        "--myosight-output",
        type=Path,
        default=None,
        help="Optional output CSV for image-level MyoSight comparison.",
    )
    parser.add_argument(
        "--candidate-name",
        type=str,
        default="baseline_gb",
        help="Weighted candidate_name to compare against pipeline_current.",
    )
    parser.add_argument(
        "--gate-quantiles",
        type=str,
        default="0.01,0.03,0.05,0.10,0.15",
        help=(
            "Comma-separated lower quantiles used to derive IIa gate thresholds. "
            "Smaller values are softer gates."
        ),
    )
    return parser


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


def _load_weighted_predictions(path: Path, candidate_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {"image_id", "label", "candidate_name", "predicted_fiber_type"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    out = df.loc[df["candidate_name"].astype(str) == candidate_name].copy()
    if out.empty:
        raise ValueError(f"No rows found for candidate_name={candidate_name!r} in {path}")
    out["image_id"] = out["image_id"].astype(str)
    out["label"] = out["label"].astype(int)
    out["predicted_fiber_type"] = (
        out["predicted_fiber_type"].fillna("").astype(str).str.lower().str.strip()
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


def derive_iia_gate_thresholds(
    true_iia_reviewed: pd.DataFrame, gate_quantile: float = 0.10
) -> dict[str, float]:
    upper_quantile = 1.0 - gate_quantile
    return {
        "min_type2_snr_mean": float(
            pd.to_numeric(true_iia_reviewed["type2_snr_mean"]).quantile(gate_quantile)
        ),
        "min_type2_coverage": float(
            pd.to_numeric(true_iia_reviewed["type2_coverage"]).quantile(gate_quantile)
        ),
        "min_type2_cov_x_snr": float(
            pd.to_numeric(true_iia_reviewed["type2_cov_x_snr"]).quantile(gate_quantile)
        ),
        "max_type1_snr_mean": float(
            pd.to_numeric(true_iia_reviewed["type1_snr_mean"]).quantile(upper_quantile)
        ),
        "max_type1_coverage": float(
            pd.to_numeric(true_iia_reviewed["type1_coverage"]).quantile(upper_quantile)
        ),
    }


def derive_iib_redirect_thresholds(
    true_iib_reviewed: pd.DataFrame, gate_quantile: float = 0.10
) -> dict[str, float]:
    upper_quantile = 1.0 - gate_quantile
    return {
        "min_type1_snr_mean": float(
            pd.to_numeric(true_iib_reviewed["type1_snr_mean"]).quantile(gate_quantile)
        ),
        "min_type1_coverage": float(
            pd.to_numeric(true_iib_reviewed["type1_coverage"]).quantile(gate_quantile)
        ),
        "min_type1_cov_x_snr": float(
            pd.to_numeric(true_iib_reviewed["type1_cov_x_snr"]).quantile(gate_quantile)
        ),
        "max_type2_snr_mean": float(
            pd.to_numeric(true_iib_reviewed["type2_snr_mean"]).quantile(upper_quantile)
        ),
        "max_type2_coverage": float(
            pd.to_numeric(true_iib_reviewed["type2_coverage"]).quantile(upper_quantile)
        ),
    }


def _resolve_feature_column(df: pd.DataFrame, base_name: str) -> str:
    candidates = [base_name, f"{base_name}_x", f"{base_name}_y"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(base_name)


def _gate_mask(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.Series:
    type2_snr_mean = _resolve_feature_column(df, "type2_snr_mean")
    type2_coverage = _resolve_feature_column(df, "type2_coverage")
    type2_cov_x_snr = _resolve_feature_column(df, "type2_cov_x_snr")
    type1_snr_mean = _resolve_feature_column(df, "type1_snr_mean")
    type1_coverage = _resolve_feature_column(df, "type1_coverage")
    type2_mean = _resolve_feature_column(df, "type2_mean")
    type1_mean = _resolve_feature_column(df, "type1_mean")
    return (
        pd.to_numeric(df[type2_snr_mean], errors="coerce").ge(
            thresholds["min_type2_snr_mean"]
        )
        & pd.to_numeric(df[type2_coverage], errors="coerce").ge(
            thresholds["min_type2_coverage"]
        )
        & pd.to_numeric(df[type2_cov_x_snr], errors="coerce").ge(
            thresholds["min_type2_cov_x_snr"]
        )
        & pd.to_numeric(df[type1_snr_mean], errors="coerce").le(
            thresholds["max_type1_snr_mean"]
        )
        & pd.to_numeric(df[type1_coverage], errors="coerce").le(
            thresholds["max_type1_coverage"]
        )
        & pd.to_numeric(df[type2_mean], errors="coerce").gt(
            pd.to_numeric(df[type1_mean], errors="coerce")
        )
    )


def _iib_redirect_mask(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.Series:
    type1_snr_mean = _resolve_feature_column(df, "type1_snr_mean")
    type1_coverage = _resolve_feature_column(df, "type1_coverage")
    type1_cov_x_snr = _resolve_feature_column(df, "type1_cov_x_snr")
    type2_snr_mean = _resolve_feature_column(df, "type2_snr_mean")
    type2_coverage = _resolve_feature_column(df, "type2_coverage")
    type1_mean = _resolve_feature_column(df, "type1_mean")
    type2_mean = _resolve_feature_column(df, "type2_mean")
    return (
        pd.to_numeric(df[type1_snr_mean], errors="coerce").ge(
            thresholds["min_type1_snr_mean"]
        )
        & pd.to_numeric(df[type1_coverage], errors="coerce").ge(
            thresholds["min_type1_coverage"]
        )
        & pd.to_numeric(df[type1_cov_x_snr], errors="coerce").ge(
            thresholds["min_type1_cov_x_snr"]
        )
        & pd.to_numeric(df[type2_snr_mean], errors="coerce").le(
            thresholds["max_type2_snr_mean"]
        )
        & pd.to_numeric(df[type2_coverage], errors="coerce").le(
            thresholds["max_type2_coverage"]
        )
        & pd.to_numeric(df[type1_mean], errors="coerce").gt(
            pd.to_numeric(df[type2_mean], errors="coerce")
        )
    )


def _apply_iia_gate(pred: pd.Series, gate_ok: pd.Series) -> pd.Series:
    out = pred.astype(str).str.lower().copy()
    demote = out.eq("iia") & ~gate_ok.fillna(False)
    out.loc[demote] = "iix"
    return out


def _apply_iia_gate_with_iib_redirect(
    pred: pd.Series, iia_gate_ok: pd.Series, iib_redirect_ok: pd.Series
) -> pd.Series:
    out = pred.astype(str).str.lower().copy()
    demote = out.eq("iia") & ~iia_gate_ok.fillna(False)
    redirect_to_iib = demote & iib_redirect_ok.fillna(False)
    out.loc[demote] = "iix"
    out.loc[redirect_to_iib] = "iib"
    return out


def _metric_row(name: str, y_true: pd.Series, y_pred: pd.Series) -> dict[str, object]:
    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["iia"],
        average=None,
        zero_division=0,
    )
    return {
        "strategy": name,
        "n_rows": int(len(y_true)),
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "iia_precision": float(prec[0]),
        "iia_recall": float(rec[0]),
        "iia_f1": float(f1[0]),
        "iia_support": int(support[0]),
    }


def _parse_gate_quantiles(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        value = float(text)
        if not 0.0 < value < 0.5:
            raise ValueError(f"gate quantile must be between 0 and 0.5, got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one gate quantile is required.")
    return values


def _summarize_image_level(
    predictions: pd.DataFrame, strategy_col: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for image_id, sub in predictions.groupby("image_id", sort=False):
        labels = sub[strategy_col].astype(str).str.lower()
        total = int(labels.isin(CANONICAL_CLASSES).sum())
        row: dict[str, object] = {"image_id": image_id, "pred_total_fibers": total}
        for fiber_type in CANONICAL_CLASSES:
            count = int(labels.eq(fiber_type).sum())
            row[f"pred_{fiber_type}_n"] = count
            row[f"pred_{fiber_type}_pct"] = count / total * 100.0 if total else pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


def build_myosight_comparison(
    predictions: pd.DataFrame,
    myosight_summary: pd.DataFrame,
    *,
    candidate_name: str,
) -> pd.DataFrame:
    strategy_cols = {
        "pipeline_current": "pred_pipeline_current",
        candidate_name: f"pred_{candidate_name}",
        f"{candidate_name}_gated_iia_best": f"pred_{candidate_name}_gated_iia",
    }
    merged = myosight_summary.copy()
    out_frames = []
    for strategy_name, strategy_col in strategy_cols.items():
        image_summary = _summarize_image_level(predictions, strategy_col)
        joined = merged.merge(
            image_summary,
            on="image_id",
            how="inner",
            validate="one_to_one",
        ).copy()
        joined["strategy"] = strategy_name
        for fiber_type in CANONICAL_CLASSES:
            joined[f"{fiber_type}_pct_diff_pred_minus_myosight"] = (
                pd.to_numeric(joined[f"pred_{fiber_type}_pct"], errors="coerce")
                - pd.to_numeric(joined[f"myosight_{fiber_type}_pct"], errors="coerce") * 100.0
            )
            joined[f"{fiber_type}_pct_abs_diff"] = joined[
                f"{fiber_type}_pct_diff_pred_minus_myosight"
            ].abs()
        out_frames.append(joined)
    return pd.concat(out_frames, ignore_index=True)


def analyze_iia_gate(
    feature_table: pd.DataFrame,
    benchmark_split: pd.DataFrame,
    weighted_predictions: pd.DataFrame,
    true_iia_reviewed: pd.DataFrame,
    *,
    candidate_name: str = "baseline_gb",
    gate_quantiles: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    gate_quantiles = gate_quantiles or [0.10]
    holdout = benchmark_split.loc[
        benchmark_split["manual_supervision_split"].eq("manual_eval_holdout")
        & benchmark_split["audit_final_label"].isin(CANONICAL_CLASSES)
    ].copy()

    keep_cols = [
        "image_id",
        "label",
        "split",
        "fiber_type",
        "needs_review",
        "model_confidence",
        "model_margin",
        "type1_mean",
        "type2_mean",
        "type1_coverage",
        "type2_coverage",
        "type1_cov_x_snr",
        "type1_snr_mean",
        "type2_cov_x_snr",
        "type2_snr_mean",
    ]
    merged = holdout.merge(
        feature_table[[c for c in keep_cols if c in feature_table.columns]],
        on=["image_id", "label"],
        how="left",
        validate="one_to_one",
    ).merge(
        weighted_predictions[["image_id", "label", "predicted_fiber_type"]],
        on=["image_id", "label"],
        how="left",
        validate="one_to_one",
    )
    merged = merged.rename(columns={"predicted_fiber_type": f"pred_{candidate_name}"})

    split_col = _resolve_feature_column(merged, "split")
    fiber_type_col = _resolve_feature_column(merged, "fiber_type")
    merged["pred_pipeline_current"] = merged[fiber_type_col].astype(str).str.lower()
    merged[f"pred_{candidate_name}"] = (
        merged[f"pred_{candidate_name}"].fillna("").astype(str).str.lower()
    )

    y_true = merged["audit_final_label"].astype(str).str.lower()
    true_iib_reviewed = (
        benchmark_split.loc[benchmark_split["audit_final_label"].eq("iib"), ["image_id", "label"]]
        .drop_duplicates()
        .merge(
            feature_table[
                [
                    c
                    for c in [
                        "image_id",
                        "label",
                        "type1_mean",
                        "type2_mean",
                        "type1_coverage",
                        "type2_coverage",
                        "type1_cov_x_snr",
                        "type1_snr_mean",
                        "type2_cov_x_snr",
                        "type2_snr_mean",
                    ]
                    if c in feature_table.columns
                ]
            ],
            on=["image_id", "label"],
            how="left",
            validate="one_to_one",
        )
    )
    metric_rows = [
        _metric_row("pipeline_current", y_true, merged["pred_pipeline_current"]),
        _metric_row(candidate_name, y_true, merged[f"pred_{candidate_name}"]),
    ]
    threshold_blocks = []
    best_quantile = gate_quantiles[0]
    best_gated_metrics: dict[str, float] | None = None
    best_gate_ok: pd.Series | None = None
    best_iib_redirect_ok: pd.Series | None = None
    best_pipeline_pred: pd.Series | None = None
    best_candidate_pred: pd.Series | None = None
    best_pipeline_pred_redirect: pd.Series | None = None
    best_candidate_pred_redirect: pd.Series | None = None
    for gate_quantile in gate_quantiles:
        thresholds = derive_iia_gate_thresholds(true_iia_reviewed, gate_quantile)
        iib_thresholds = derive_iib_redirect_thresholds(true_iib_reviewed, gate_quantile)
        gate_ok = _gate_mask(merged, thresholds)
        iib_redirect_ok = _iib_redirect_mask(merged, iib_thresholds)
        pred_pipeline_gated = _apply_iia_gate(merged["pred_pipeline_current"], gate_ok)
        pred_candidate_gated = _apply_iia_gate(merged[f"pred_{candidate_name}"], gate_ok)
        pred_pipeline_redirect = _apply_iia_gate_with_iib_redirect(
            merged["pred_pipeline_current"], gate_ok, iib_redirect_ok
        )
        pred_candidate_redirect = _apply_iia_gate_with_iib_redirect(
            merged[f"pred_{candidate_name}"], gate_ok, iib_redirect_ok
        )
        pipeline_name = f"pipeline_gated_iia_q{gate_quantile:.2f}"
        candidate_gate_name = f"{candidate_name}_gated_iia_q{gate_quantile:.2f}"
        pipeline_redirect_name = f"pipeline_gated_iia_iib_redirect_q{gate_quantile:.2f}"
        candidate_redirect_name = (
            f"{candidate_name}_gated_iia_iib_redirect_q{gate_quantile:.2f}"
        )
        metric_rows.append(_metric_row(pipeline_name, y_true, pred_pipeline_gated))
        candidate_metrics = _metric_row(candidate_gate_name, y_true, pred_candidate_gated)
        metric_rows.append(candidate_metrics)
        metric_rows.append(_metric_row(pipeline_redirect_name, y_true, pred_pipeline_redirect))
        candidate_redirect_metrics = _metric_row(
            candidate_redirect_name, y_true, pred_candidate_redirect
        )
        metric_rows.append(candidate_redirect_metrics)
        threshold_blocks.append(
            "\n".join(
                [
                    f"Gate q={gate_quantile:.2f}:",
                    *(f"- {k}: {v:.6f}" for k, v in thresholds.items()),
                    *(f"- redirect_{k}: {v:.6f}" for k, v in iib_thresholds.items()),
                ]
            )
        )
        best_this_quantile = max(
            [candidate_metrics, candidate_redirect_metrics],
            key=lambda row: (row["balanced_accuracy"], row["accuracy"]),
        )
        if (
            best_gated_metrics is None
            or best_this_quantile["balanced_accuracy"] > best_gated_metrics["balanced_accuracy"]
            or (
                best_this_quantile["balanced_accuracy"]
                == best_gated_metrics["balanced_accuracy"]
                and best_this_quantile["accuracy"] > best_gated_metrics["accuracy"]
            )
        ):
            best_quantile = gate_quantile
            best_gated_metrics = best_this_quantile
            best_gate_ok = gate_ok
            best_iib_redirect_ok = iib_redirect_ok
            best_pipeline_pred = pred_pipeline_gated
            best_candidate_pred = pred_candidate_gated
            best_pipeline_pred_redirect = pred_pipeline_redirect
            best_candidate_pred_redirect = pred_candidate_redirect

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["balanced_accuracy", "accuracy"],
        ascending=[False, False],
        kind="stable",
    )
    assert best_gated_metrics is not None
    assert best_gate_ok is not None
    assert best_iib_redirect_ok is not None
    assert best_pipeline_pred is not None
    assert best_candidate_pred is not None
    assert best_pipeline_pred_redirect is not None
    assert best_candidate_pred_redirect is not None

    report_lines = [
        "IIa gate thresholds derived from confirmed true_iia_hunt positives:",
        *threshold_blocks,
        "",
        "Strategy comparison on manual_eval_holdout:",
        metrics.to_string(index=False),
        "",
        (
            f"Best gated {candidate_name} quantile: q={best_quantile:.2f} "
            f"(balanced_accuracy={best_gated_metrics['balanced_accuracy']:.6f}, "
            f"accuracy={best_gated_metrics['accuracy']:.6f})"
        ),
    ]
    evidence_cols = [
        _resolve_feature_column(merged, "type1_mean"),
        _resolve_feature_column(merged, "type2_mean"),
        _resolve_feature_column(merged, "type1_coverage"),
        _resolve_feature_column(merged, "type2_coverage"),
        _resolve_feature_column(merged, "type1_snr_mean"),
        _resolve_feature_column(merged, "type2_snr_mean"),
        _resolve_feature_column(merged, "type1_cov_x_snr"),
        _resolve_feature_column(merged, "type2_cov_x_snr"),
    ]
    prediction_cols = [
        "image_id",
        "label",
        split_col,
        "audit_final_label",
        fiber_type_col,
        "pred_pipeline_current",
        f"pred_{candidate_name}",
    ]
    predictions = merged.loc[
        :, [c for c in prediction_cols if c in merged.columns] + evidence_cols
    ].copy()
    predictions["best_gate_quantile"] = best_quantile
    predictions["gate_iia_ok"] = best_gate_ok
    predictions["gate_iib_redirect_ok"] = best_iib_redirect_ok
    predictions["pred_pipeline_gated_iia"] = best_pipeline_pred
    predictions[f"pred_{candidate_name}_gated_iia"] = best_candidate_pred
    predictions["pred_pipeline_gated_iia_iib_redirect"] = best_pipeline_pred_redirect
    predictions[f"pred_{candidate_name}_gated_iia_iib_redirect"] = best_candidate_pred_redirect
    return metrics, predictions, "\n".join(report_lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    feature_table = _load_feature_table(args.feature_table)
    benchmark_split = _load_benchmark_split(args.benchmark_split)
    weighted_predictions = _load_weighted_predictions(
        args.weighted_predictions, args.candidate_name
    )
    true_iia_reviewed = _load_true_iia_reviewed(args.true_iia_reviewed_glob)
    gate_quantiles = _parse_gate_quantiles(args.gate_quantiles)
    metrics, predictions, report = analyze_iia_gate(
        feature_table,
        benchmark_split,
        weighted_predictions,
        true_iia_reviewed,
        candidate_name=args.candidate_name,
        gate_quantiles=gate_quantiles,
    )

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics_output, index=False)
    predictions.to_csv(args.predictions_output, index=False)
    args.report_output.write_text(report, encoding="utf-8")

    if args.myosight_summary is not None:
        myosight_summary = pd.read_csv(args.myosight_summary, low_memory=False)
        myosight_comparison = build_myosight_comparison(
            predictions, myosight_summary, candidate_name=args.candidate_name
        )
        if args.myosight_output is not None:
            args.myosight_output.parent.mkdir(parents=True, exist_ok=True)
            myosight_comparison.to_csv(args.myosight_output, index=False)
        summary = (
            myosight_comparison.groupby("strategy")
            .agg(
                n_images=("image_id", "nunique"),
                mean_abs_iia_pct_diff=("iia_pct_abs_diff", "mean"),
                mean_abs_iib_pct_diff=("iib_pct_abs_diff", "mean"),
                mean_abs_iix_pct_diff=("iix_pct_abs_diff", "mean"),
            )
            .reset_index()
            .sort_values(
                ["mean_abs_iia_pct_diff", "mean_abs_iib_pct_diff", "mean_abs_iix_pct_diff"],
                ascending=[True, True, True],
                kind="stable",
            )
        )
        with args.report_output.open("a", encoding="utf-8") as fh:
            fh.write("\nSecondary MyoSight image-level comparison:\n")
            fh.write(summary.to_string(index=False))
            fh.write("\n")

    print(metrics.to_string(index=False))
    print(f"saved metrics: {args.metrics_output}")
    print(f"saved predictions: {args.predictions_output}")
    print(f"saved report: {args.report_output}")
    if args.myosight_output is not None:
        print(f"saved myosight comparison: {args.myosight_output}")


if __name__ == "__main__":
    main()
