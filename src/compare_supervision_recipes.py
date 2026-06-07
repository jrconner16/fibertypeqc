from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

from src.analyze_iia_gate import (
    _apply_iia_gate,
    _gate_mask,
    derive_iia_gate_thresholds,
)
from src.evaluate_against_audit_benchmark import CANONICAL_CLASSES
from src.quantify_classify import FROZEN_ALPHA_BASELINE_FEATURES
from src.train_candidate_from_feature_table import _boosting_model, _load_feature_table
from src.train_weighted_candidate_from_audit import _load_manual_split

RECIPE_ORDER = (
    "manual_only_high",
    "manual_high_myo_medium",
    "manual_high_myo_medium_baseline_light",
    "manual_high_myo_medium_baseline_light_soft_iia_q0.01",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare focused supervision recipes using baseline_gb features/model on the "
            "protected manual eval holdout."
        )
    )
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--manual-split", type=Path, required=True)
    parser.add_argument("--matched-myosight-audit", type=Path, required=True)
    parser.add_argument("--true-iia-reviewed-glob", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manual-weight", type=float, default=8.0)
    parser.add_argument("--myosight-weight", type=float, default=3.0)
    parser.add_argument("--baseline-weight", type=float, default=0.5)
    return parser


def _load_matched_myosight(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {"image_id", "label", "myosight_label", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    out = df.copy()
    out["image_id"] = out["image_id"].astype(str)
    out["label"] = out["label"].astype(int)
    out["myosight_label"] = out["myosight_label"].fillna("").astype(str).str.lower().str.strip()
    out["split"] = out["split"].fillna("").astype(str).str.lower().str.strip()
    out = out.loc[out["myosight_label"].isin(CANONICAL_CLASSES)].copy()
    return out.drop_duplicates(subset=["image_id", "label"], keep="first")


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


def _prepare_base_table(
    feature_table: pd.DataFrame,
    manual_split: pd.DataFrame,
    matched_myosight: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manual_keep = manual_split[
        ["image_id", "label", "audit_final_label", "manual_supervision_split"]
    ].copy()
    myo_keep = matched_myosight[["image_id", "label", "myosight_label"]].copy()
    merged = feature_table.merge(
        manual_keep,
        on=["image_id", "label"],
        how="left",
        validate="one_to_one",
    ).merge(
        myo_keep,
        on=["image_id", "label"],
        how="left",
        validate="one_to_one",
    )
    merged["audit_final_label"] = merged["audit_final_label"].fillna("").astype(str).str.lower()
    merged["manual_supervision_split"] = (
        merged["manual_supervision_split"].fillna("").astype(str).str.lower()
    )
    merged["myosight_label"] = merged["myosight_label"].fillna("").astype(str).str.lower()

    is_manual_eval = (
        merged["manual_supervision_split"].eq("manual_eval_holdout")
        & merged["audit_final_label"].isin(CANONICAL_CLASSES)
    )
    eval_rows = merged.loc[is_manual_eval].copy()
    if eval_rows.empty:
        raise ValueError("No manual_eval_holdout rows available for evaluation.")
    return merged, eval_rows


def _assemble_recipe_training_rows(
    merged: pd.DataFrame,
    *,
    recipe_name: str,
    manual_weight: float,
    myosight_weight: float,
    baseline_weight: float,
) -> pd.DataFrame:
    manual_mask = (
        merged["manual_supervision_split"].eq("manual_train")
        & merged["audit_final_label"].isin(CANONICAL_CLASSES)
    )
    myo_mask = (
        merged["split"].astype(str).eq("train")
        & merged["myosight_label"].isin(CANONICAL_CLASSES)
        & ~manual_mask
    )
    baseline_mask = (
        merged["split"].astype(str).eq("train")
        & ~manual_mask
        & ~myo_mask
        & merged["fiber_type"].astype(str).isin(CANONICAL_CLASSES)
    )

    if recipe_name == "manual_only_high":
        include_mask = manual_mask
    elif recipe_name == "manual_high_myo_medium":
        include_mask = manual_mask | myo_mask
    elif recipe_name == "manual_high_myo_medium_baseline_light":
        include_mask = manual_mask | myo_mask | baseline_mask
    else:
        raise ValueError(f"Unknown recipe: {recipe_name}")

    train_df = merged.loc[include_mask].copy()
    if train_df.empty:
        raise ValueError(f"No training rows selected for recipe {recipe_name!r}")

    train_df["training_label"] = train_df["fiber_type"].astype(str).str.lower()
    train_df["sample_weight"] = baseline_weight
    train_df["training_source"] = "baseline"

    manual_sel = manual_mask.loc[train_df.index]
    myo_sel = myo_mask.loc[train_df.index]
    baseline_sel = baseline_mask.loc[train_df.index]

    train_df.loc[baseline_sel, "training_label"] = (
        train_df.loc[baseline_sel, "fiber_type"].astype(str).str.lower()
    )
    train_df.loc[baseline_sel, "sample_weight"] = float(baseline_weight)
    train_df.loc[baseline_sel, "training_source"] = "baseline"

    train_df.loc[myo_sel, "training_label"] = train_df.loc[myo_sel, "myosight_label"]
    train_df.loc[myo_sel, "sample_weight"] = float(myosight_weight)
    train_df.loc[myo_sel, "training_source"] = "myosight"

    train_df.loc[manual_sel, "training_label"] = train_df.loc[manual_sel, "audit_final_label"]
    train_df.loc[manual_sel, "sample_weight"] = float(manual_weight)
    train_df.loc[manual_sel, "training_source"] = "manual"
    return train_df


def _metric_row(recipe_name: str, y_true: pd.Series, y_pred: pd.Series) -> dict[str, object]:
    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    iia_mask = y_true.eq("iia")
    pred_iia = y_pred.eq("iia")
    tp = int((iia_mask & pred_iia).sum())
    fp = int((~iia_mask & pred_iia).sum())
    fn = int((iia_mask & ~pred_iia).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "recipe": recipe_name,
        "n_rows": int(len(y_true)),
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "iia_precision": precision,
        "iia_recall": recall,
        "iia_f1": f1,
        "iia_support": int(iia_mask.sum()),
    }


def compare_supervision_recipes(
    feature_table: pd.DataFrame,
    manual_split: pd.DataFrame,
    matched_myosight: pd.DataFrame,
    true_iia_reviewed: pd.DataFrame,
    *,
    manual_weight: float = 8.0,
    myosight_weight: float = 3.0,
    baseline_weight: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    merged, eval_df = _prepare_base_table(feature_table, manual_split, matched_myosight)
    thresholds = derive_iia_gate_thresholds(true_iia_reviewed, gate_quantile=0.01)
    y_true = eval_df["audit_final_label"].astype(str).str.lower()

    feature_columns = list(FROZEN_ALPHA_BASELINE_FEATURES)
    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    models: dict[str, object] = {}
    training_summaries: list[dict[str, object]] = []

    base_recipes = RECIPE_ORDER[:3]
    for recipe_name in base_recipes:
        train_df = _assemble_recipe_training_rows(
            merged,
            recipe_name=recipe_name,
            manual_weight=manual_weight,
            myosight_weight=myosight_weight,
            baseline_weight=baseline_weight,
        )
        model = _boosting_model()
        model.fit(
            train_df[feature_columns],
            train_df["training_label"].astype(str),
            sample_weight=train_df["sample_weight"].astype(float),
        )
        pred = pd.Series(
            model.predict(eval_df[feature_columns]),
            index=eval_df.index,
            name="predicted_fiber_type",
        ).astype(str).str.lower()
        metrics_rows.append(_metric_row(recipe_name, y_true, pred))
        pred_df = eval_df[
            [
                col
                for col in (
                    "image_id",
                    "label",
                    "split",
                    "fiber_type",
                    "audit_final_label",
                    "manual_supervision_split",
                    "needs_review",
                    "model_confidence",
                    "model_margin",
                )
                if col in eval_df.columns
            ]
        ].copy()
        pred_df["recipe"] = recipe_name
        pred_df["predicted_fiber_type"] = pred.to_numpy()
        prediction_frames.append(pred_df)
        models[recipe_name] = model
        training_summaries.append(
            {
                "recipe": recipe_name,
                "train_rows": int(len(train_df)),
                "manual_rows": int(train_df["training_source"].eq("manual").sum()),
                "myosight_rows": int(train_df["training_source"].eq("myosight").sum()),
                "baseline_rows": int(train_df["training_source"].eq("baseline").sum()),
            }
        )

        if recipe_name == "manual_high_myo_medium_baseline_light":
            gate_ok = _gate_mask(eval_df, thresholds)
            gated_pred = _apply_iia_gate(pred, gate_ok)
            gated_name = "manual_high_myo_medium_baseline_light_soft_iia_q0.01"
            metrics_rows.append(_metric_row(gated_name, y_true, gated_pred))
            gated_df = pred_df.copy()
            gated_df["recipe"] = gated_name
            gated_df["predicted_fiber_type"] = gated_pred.to_numpy()
            gated_df["gate_iia_ok"] = gate_ok.to_numpy()
            prediction_frames.append(gated_df)

    metrics_df = pd.DataFrame(metrics_rows).merge(
        pd.DataFrame(training_summaries),
        on="recipe",
        how="left",
    )
    metrics_df = metrics_df.sort_values(
        ["balanced_accuracy", "accuracy"],
        ascending=[False, False],
        kind="stable",
    )
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    report = {
        "thresholds": thresholds,
        "models": models,
        "feature_columns": feature_columns,
        "eval_rows": int(len(eval_df)),
    }
    return metrics_df, predictions_df, report


def main() -> None:
    args = build_parser().parse_args()
    feature_table = _load_feature_table(args.feature_table)
    manual_split = _load_manual_split(args.manual_split)
    matched_myosight = _load_matched_myosight(args.matched_myosight_audit)
    true_iia_reviewed = _load_true_iia_reviewed(args.true_iia_reviewed_glob)

    metrics_df, predictions_df, report = compare_supervision_recipes(
        feature_table,
        manual_split,
        matched_myosight,
        true_iia_reviewed,
        manual_weight=args.manual_weight,
        myosight_weight=args.myosight_weight,
        baseline_weight=args.baseline_weight,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "supervision_recipe_metrics.csv"
    predictions_path = args.output_dir / "supervision_recipe_predictions.csv"
    report_path = args.output_dir / "supervision_recipe_report.txt"
    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    best_recipe = str(metrics_df.iloc[0]["recipe"])
    best_model = report["models"].get(best_recipe)
    if best_model is not None:
        joblib.dump(best_model, args.output_dir / f"{best_recipe}.joblib")

    report_lines = [
        "Supervision recipe comparison on manual_eval_holdout:",
        metrics_df.to_string(index=False),
        "",
        "Soft IIa gate thresholds (q=0.01) used for gated recipe:",
        *(f"- {k}: {v:.6f}" for k, v in report["thresholds"].items()),
        "",
        f"Eval rows: {report['eval_rows']}",
        "",
    ]
    best_recipe_rows = predictions_df.loc[predictions_df["recipe"].eq(best_recipe)].copy()
    if not best_recipe_rows.empty and best_model is not None:
        report_lines.append("Best recipe classification report:")
        report_lines.append(
            classification_report(
                best_recipe_rows["audit_final_label"].astype(str),
                best_recipe_rows["predicted_fiber_type"].astype(str),
                zero_division=0,
            )
        )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(metrics_df.to_string(index=False))
    print(f"saved metrics: {metrics_path}")
    print(f"saved predictions: {predictions_path}")
    print(f"saved report: {report_path}")


if __name__ == "__main__":
    main()
