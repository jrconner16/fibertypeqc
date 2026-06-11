from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

from src.compare_myosight_pipeline import (
    counts_by_type,
    discover_myosight,
    image_id_from_myosight_path,
    natural_key,
    normalize_type,
    numeric_summary,
    pct,
)
from src.compare_supervision_recipes import (
    _assemble_recipe_training_rows,
    _load_matched_myosight,
    _prepare_base_table,
)
from src.quantify_classify import QuantifyConfig, _add_model_signal_qc, FROZEN_ALPHA_BASELINE_FEATURES
from src.run_batch import V0_PARAMS
from src.train_candidate_from_feature_table import _boosting_model, _load_feature_table
from src.train_weighted_candidate_from_audit import _load_manual_split

COHORT_FIBER_COLUMNS = [
    "label",
    "area",
    "area_um2",
    "fiber_type",
    "type1_signal_evidence",
    "type2_signal_evidence",
]
COHORT_DIAGNOSTIC_COLUMNS = ["label", *FROZEN_ALPHA_BASELINE_FEATURES]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep manual_high_myo_medium_baseline_light weights and score each candidate "
            "on the protected manual holdout plus the MyoSight 32-image cohort without "
            "rerunning segmentation."
        )
    )
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--manual-split", type=Path, required=True)
    parser.add_argument("--matched-myosight-audit", type=Path, required=True)
    parser.add_argument("--myosight-root", type=Path, required=True)
    parser.add_argument(
        "--direct-root",
        type=Path,
        default=Path("outputs/myosight_validation_all_diagnostics"),
        help="Root containing direct-image fibers and feature_diagnostics tables.",
    )
    parser.add_argument(
        "--trusted-section-root",
        type=Path,
        default=Path("outputs/myosight_validation_candidate_sections_trusted_baseline_gb_soft_iia_q001"),
        help="Trusted section-export root with fibers and feature_diagnostics tables.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manual-weight", type=float, default=8.0)
    parser.add_argument(
        "--myosight-weights",
        type=str,
        default="2.0,3.0,4.0,5.0",
        help="Comma-separated matched-MyoSight weights to evaluate.",
    )
    parser.add_argument(
        "--baseline-weights",
        type=str,
        default="0.1,0.25,0.5,0.75,1.0",
        help="Comma-separated baseline/self-label weights to evaluate.",
    )
    parser.add_argument(
        "--section-image-id",
        action="append",
        default=[],
        help="Optional explicit section image IDs. When omitted, discover from trusted-section-root.",
    )
    return parser


def _parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        text = part.strip()
        if text:
            values.append(float(text))
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def _discover_section_ids(trusted_section_root: Path, explicit: list[str]) -> set[str]:
    if explicit:
        return {str(v).strip() for v in explicit if str(v).strip()}
    return {path.name for path in trusted_section_root.iterdir() if path.is_dir()}


def _discover_cohort_ids(direct_root: Path, trusted_section_root: Path, section_ids: set[str]) -> list[str]:
    direct_ids = {path.name for path in direct_root.iterdir() if path.is_dir()}
    trusted_ids = {path.name for path in trusted_section_root.iterdir() if path.is_dir()}
    return sorted((direct_ids - section_ids) | (trusted_ids & section_ids), key=natural_key)


def _class_metrics(y_true: pd.Series, y_pred: pd.Series, label: str) -> dict[str, float]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true.astype(str),
        y_pred.astype(str),
        labels=[label],
        zero_division=0,
    )
    return {
        f"{label}_precision": float(precision[0]),
        f"{label}_recall": float(recall[0]),
        f"{label}_f1": float(f1[0]),
        f"{label}_support": int(support[0]),
    }


def _evaluate_holdout(
    feature_table: pd.DataFrame,
    manual_split: pd.DataFrame,
    matched_myosight: pd.DataFrame,
    *,
    manual_weight: float,
    myosight_weight: float,
    baseline_weight: float,
) -> tuple[object, pd.DataFrame]:
    merged, eval_df = _prepare_base_table(feature_table, manual_split, matched_myosight)
    train_df = _assemble_recipe_training_rows(
        merged,
        recipe_name="manual_high_myo_medium_baseline_light",
        manual_weight=manual_weight,
        myosight_weight=myosight_weight,
        baseline_weight=baseline_weight,
    )
    feature_columns = list(FROZEN_ALPHA_BASELINE_FEATURES)
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
    ).astype(str)
    y_true = eval_df["audit_final_label"].astype(str)
    row = {
        "holdout_accuracy": float(accuracy_score(y_true, pred)),
        "holdout_balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "train_rows": int(len(train_df)),
        "manual_rows": int(train_df["training_source"].eq("manual").sum()),
        "myosight_rows": int(train_df["training_source"].eq("myosight").sum()),
        "baseline_rows": int(train_df["training_source"].eq("baseline").sum()),
    }
    row.update(_class_metrics(y_true, pred, "iib"))
    row.update(_class_metrics(y_true, pred, "iix"))
    return model, pd.DataFrame([row])


def _load_pipeline_pair(root: Path, image_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_dir = root / image_id
    fibers_path = image_dir / f"{image_id}_fibers.csv"
    diag_path = image_dir / f"{image_id}_feature_diagnostics.csv"
    if not fibers_path.exists():
        raise FileNotFoundError(f"Missing fibers file for {image_id}: {fibers_path}")
    if not diag_path.exists():
        raise FileNotFoundError(f"Missing diagnostics file for {image_id}: {diag_path}")
    fiber_cols = pd.read_csv(fibers_path, nrows=0).columns.tolist()
    diag_cols = pd.read_csv(diag_path, nrows=0).columns.tolist()
    fiber_usecols = [col for col in COHORT_FIBER_COLUMNS if col in fiber_cols]
    diag_usecols = [col for col in COHORT_DIAGNOSTIC_COLUMNS if col in diag_cols]
    return (
        pd.read_csv(fibers_path, low_memory=False, usecols=fiber_usecols),
        pd.read_csv(diag_path, low_memory=False, usecols=diag_usecols),
    )


def _load_cohort_cache(
    *,
    myosight_root: Path,
    direct_root: Path,
    trusted_section_root: Path,
    section_ids: set[str],
) -> dict[str, dict[str, object]]:
    myo_map_raw = discover_myosight(myosight_root)
    image_ids = _discover_cohort_ids(direct_root, trusted_section_root, section_ids)
    cache: dict[str, dict[str, object]] = {}
    for image_id in image_ids:
        myo_path = myo_map_raw.get(image_id)
        if myo_path is None:
            raise FileNotFoundError(f"Missing MyoSight result for cohort image_id={image_id}")
        myo = pd.read_csv(myo_path, sep="," if myo_path.suffix.lower() == ".csv" else "\t")
        source_root = trusted_section_root if image_id in section_ids else direct_root
        fibers, diagnostics = _load_pipeline_pair(source_root, image_id)
        cache[image_id] = {
            "myo_path": myo_path,
            "myo": myo,
            "fibers": fibers,
            "diagnostics": diagnostics,
        }
    return cache


def _reclassify_fibers(
    fibers: pd.DataFrame,
    diagnostics: pd.DataFrame,
    model,
) -> pd.DataFrame:
    feat_cols = [c for c in getattr(model, "feature_names_in_", []) if c in diagnostics.columns]
    missing = sorted(set(getattr(model, "feature_names_in_", [])) - set(feat_cols))
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(missing[:10])}")
    x = diagnostics[feat_cols]
    pred = pd.Series(model.predict(x), index=diagnostics.index, dtype="object").astype(str)
    proba = np.asarray(model.predict_proba(x), dtype=np.float32)
    classes = [str(c).lower() for c in model.classes_]
    proba_df = pd.DataFrame(index=diagnostics.index)
    for cls, values in zip(classes, proba.T, strict=False):
        proba_df[f"prob_{cls}"] = values
    for col in ("prob_iib", "prob_iia", "prob_iix"):
        if col not in proba_df.columns:
            proba_df[col] = np.nan
    sorted_proba = np.sort(proba, axis=1)
    model_confidence = sorted_proba[:, -1].astype(np.float32)
    model_margin = (
        sorted_proba[:, -1] - sorted_proba[:, -2] if proba.shape[1] > 1 else sorted_proba[:, -1]
    ).astype(np.float32)

    out = fibers.drop(
        columns=[
            "fiber_type",
            "fiber_type_source",
            "classification_method",
            "model_confidence",
            "model_margin",
            "prob_iib",
            "prob_iia",
            "prob_iix",
            "needs_review",
            "confidence",
            "classifier_path",
            "typing_signal_qc_flags",
            "has_signal_qc_warning",
            "fiber_type_pre_iia_gate",
            "gate_iia_ok",
            "gate_iib_redirect_ok",
        ],
        errors="ignore",
    ).copy()
    out["fiber_type"] = pred.astype(str)
    out["fiber_type_source"] = "model_prediction"
    out["classification_method"] = "model"
    out["model_confidence"] = model_confidence
    out["model_margin"] = model_margin
    out["prob_iib"] = proba_df["prob_iib"].to_numpy()
    out["prob_iia"] = proba_df["prob_iia"].to_numpy()
    out["prob_iix"] = proba_df["prob_iix"].to_numpy()
    out["confidence"] = model_confidence
    out["classifier_path"] = "in_memory_sweep"
    out = _add_model_signal_qc(
        out,
        QuantifyConfig(
            model_confidence_threshold=float(V0_PARAMS["model_confidence_threshold"]),
            model_margin_threshold=float(V0_PARAMS["model_margin_threshold"]),
        ),
    )
    return out


def _summarize_pair_frames(
    image_id: str,
    myo_path: Path,
    myo: pd.DataFrame,
    pipe: pd.DataFrame,
) -> dict[str, object]:
    myo = myo.copy()
    pipe = pipe.copy()
    myo["label_norm"] = myo["Label"].map(normalize_type)
    pipe["label_norm"] = pipe["fiber_type"].map(normalize_type)

    myo_total = int(len(myo))
    pipe_total = int(len(pipe))
    myo_counts = counts_by_type(myo["label_norm"])
    pipe_counts = counts_by_type(pipe["label_norm"])

    row: dict[str, object] = {
        "image_id": image_id,
        "myosight_path": str(myo_path),
        "pipeline_total_fibers": pipe_total,
        "myosight_total_fibers": myo_total,
        "fiber_count_diff": pipe_total - myo_total,
        "fiber_count_ratio_pipeline_to_myosight": float(pipe_total / myo_total)
        if myo_total
        else float("nan"),
    }
    for typ in ("iib", "iia", "iix", "hybrid", "uncertain", "exclude"):
        myo_n = myo_counts[typ]
        pipe_n = pipe_counts[typ]
        row[f"myosight_{typ}_n"] = myo_n
        row[f"pipeline_{typ}_n"] = pipe_n
        row[f"myosight_{typ}_pct"] = pct(myo_n, myo_total)
        row[f"pipeline_{typ}_pct"] = pct(pipe_n, pipe_total)
        row[f"{typ}_pct_diff_pipeline_minus_myosight"] = (
            row[f"pipeline_{typ}_pct"] - row[f"myosight_{typ}_pct"]
        )
    if "Area" in myo.columns:
        row.update(numeric_summary(myo["Area"], "myosight_area"))
    pipeline_area_col = "area_um2" if "area_um2" in pipe.columns else "area"
    if pipeline_area_col in pipe.columns:
        row["pipeline_area_column"] = pipeline_area_col
        row.update(numeric_summary(pipe[pipeline_area_col], "pipeline_area"))
        row["area_median_ratio_pipeline_to_myosight"] = (
            row["pipeline_area_median"] / row["myosight_area_median"]
            if row.get("myosight_area_median")
            else float("nan")
        )
    row["pipeline_needs_review_n"] = int(
        pipe.get("needs_review", pd.Series(False, index=pipe.index))
        .fillna(False)
        .astype(bool)
        .sum()
    )
    row["pipeline_signal_warning_n"] = int(
        pipe.get("has_signal_qc_warning", pd.Series(False, index=pipe.index))
        .fillna(False)
        .astype(bool)
        .sum()
    )
    return row


def _cohort_image_summary(
    *,
    cohort_cache: dict[str, dict[str, object]],
    model,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for image_id in sorted(cohort_cache, key=natural_key):
        entry = cohort_cache[image_id]
        myo_path = entry["myo_path"]
        myo = entry["myo"]
        fibers = entry["fibers"]
        diagnostics = entry["diagnostics"]
        reclassified = _reclassify_fibers(fibers, diagnostics, model)
        rows.append(
            _summarize_pair_frames(
                image_id_from_myosight_path(myo_path),
                myo_path,
                myo,
                reclassified,
            )
        )
    return pd.DataFrame(rows)


def _aggregate_gap_metrics(summary: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for fiber_type in ("iib", "iix"):
        gaps = (
            pd.to_numeric(summary[f"pipeline_{fiber_type}_pct"], errors="coerce")
            - pd.to_numeric(summary[f"myosight_{fiber_type}_pct"], errors="coerce")
        ).abs()
        out[f"cohort_mean_abs_gap_{fiber_type}"] = float(gaps.mean())
    return out


def sweep_candidates(args: argparse.Namespace) -> None:
    feature_table = _load_feature_table(args.feature_table)
    manual_split = _load_manual_split(args.manual_split)
    matched_myosight = _load_matched_myosight(args.matched_myosight_audit)
    myosight_weights = _parse_float_list(args.myosight_weights)
    baseline_weights = _parse_float_list(args.baseline_weights)
    section_ids = _discover_section_ids(args.trusted_section_root, args.section_image_id)
    cohort_cache = _load_cohort_cache(
        myosight_root=args.myosight_root,
        direct_root=args.direct_root,
        trusted_section_root=args.trusted_section_root,
        section_ids=section_ids,
    )
    total_variants = len(myosight_weights) * len(baseline_weights)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(exist_ok=True)
    (args.output_dir / "image_summaries").mkdir(exist_ok=True)

    rows: list[dict[str, object]] = []
    variant_index = 0
    for myo_weight in myosight_weights:
        for baseline_weight in baseline_weights:
            variant_index += 1
            model, holdout = _evaluate_holdout(
                feature_table,
                manual_split,
                matched_myosight,
                manual_weight=args.manual_weight,
                myosight_weight=myo_weight,
                baseline_weight=baseline_weight,
            )
            tag = (
                f"manual{args.manual_weight:g}_myo{myo_weight:g}_base{baseline_weight:g}"
                .replace(".", "p")
            )
            print(
                f"[{variant_index}/{total_variants}] evaluating {tag}",
                flush=True,
            )
            joblib.dump(model, args.output_dir / "models" / f"{tag}.joblib")
            summary = _cohort_image_summary(
                cohort_cache=cohort_cache,
                model=model,
            )
            summary_path = args.output_dir / "image_summaries" / f"{tag}_image_summary.csv"
            summary.to_csv(summary_path, index=False)

            row = {
                "candidate_tag": tag,
                "manual_weight": float(args.manual_weight),
                "myosight_weight": float(myo_weight),
                "baseline_weight": float(baseline_weight),
                "image_summary_path": str(summary_path),
            }
            row.update(holdout.iloc[0].to_dict())
            row.update(_aggregate_gap_metrics(summary))
            rows.append(row)
            pd.DataFrame(rows).sort_values(
                ["cohort_mean_abs_gap_iix", "cohort_mean_abs_gap_iib", "holdout_balanced_accuracy"],
                ascending=[True, True, False],
                kind="stable",
            ).to_csv(args.output_dir / "sweep_summary_partial.csv", index=False)

    out = pd.DataFrame(rows).sort_values(
        ["cohort_mean_abs_gap_iix", "cohort_mean_abs_gap_iib", "holdout_balanced_accuracy"],
        ascending=[True, True, False],
        kind="stable",
    )
    out.to_csv(args.output_dir / "sweep_summary.csv", index=False)
    print(out.to_string(index=False))
    print(f"saved sweep: {args.output_dir / 'sweep_summary.csv'}")


def main() -> None:
    args = build_parser().parse_args()
    sweep_candidates(args)


if __name__ == "__main__":
    main()
