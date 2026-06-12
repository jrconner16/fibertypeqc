from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.analyze_iia_gate import (
    _apply_iia_gate,
    _gate_mask,
    derive_iia_gate_thresholds,
)
from src.sweep_candidate_weights_on_cohort import _discover_cohort_ids, _discover_section_ids
from src.train_candidate_from_feature_table import _load_feature_table


NUMERIC_FEATURES = [
    "candidate_model_confidence",
    "candidate_model_margin",
    "type1_mean",
    "type2_mean",
    "type1_coverage",
    "type2_coverage",
    "type1_cov_x_snr",
    "type1_snr_mean",
    "type2_cov_x_snr",
    "type2_snr_mean",
]
CAT_FEATURES = [
    "type1_signal_evidence",
    "type2_signal_evidence",
    "input_kind",
    "genotype",
    "timepoint",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a small predicted-IIx correction-risk ranker from reviewed pilot rows and "
            "score the gated-IIx cohort."
        )
    )
    parser.add_argument("--reviewed-glob", type=str, required=True)
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--classifier-path", type=Path, required=True)
    parser.add_argument("--true-iia-reviewed-glob", type=str, required=True)
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--trusted-section-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate-quantile", type=float, default=0.01)
    return parser


def _load_reviewed_iix(pattern: str) -> pd.DataFrame:
    files = sorted(Path().glob(pattern))
    if not files:
        raise ValueError(f"No reviewed files matched: {pattern}")
    df = pd.concat([pd.read_csv(path, low_memory=False) for path in files], ignore_index=True)
    for col in [
        "candidate_pred_gated",
        "audit_corrected_type",
        "type1_signal_evidence",
        "type2_signal_evidence",
        "input_kind",
        "genotype",
        "timepoint",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.lower().str.strip()
    for col in ["audit_is_uncertain", "audit_is_excluded"]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    df["label"] = df["label"].astype(int)
    df["image_id"] = df["image_id"].astype(str)
    df = df.loc[df["candidate_pred_gated"].eq("iix")].copy()
    accepted = (
        (df["audit_corrected_type"].eq("") | df["audit_corrected_type"].eq("iix"))
        & ~df["audit_is_uncertain"]
        & ~df["audit_is_excluded"]
    )
    df["needs_correction"] = ~accepted
    return df


def _build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CAT_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )


def _prepare_training_table(reviewed_iix: pd.DataFrame, feature_table: pd.DataFrame) -> pd.DataFrame:
    keep = ["image_id", "label", *[c for c in NUMERIC_FEATURES if c in feature_table.columns]]
    merged = reviewed_iix.merge(
        feature_table[keep].copy(),
        on=["image_id", "label"],
        how="left",
        validate="one_to_one",
    )
    return merged


def _score_rows(model: Pipeline, df: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict_proba(df[NUMERIC_FEATURES + CAT_FEATURES])[:, 1], index=df.index)


def _leave_one_image_out_eval(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    scored_frames: list[pd.DataFrame] = []
    image_ids = sorted(df["image_id"].astype(str).unique())
    for image_id in image_ids:
        test = df.loc[df["image_id"].eq(image_id)].copy()
        train = df.loc[~df["image_id"].eq(image_id)].copy()
        if test.empty or train["needs_correction"].nunique() < 2:
            continue
        model = _build_pipeline()
        model.fit(train[NUMERIC_FEATURES + CAT_FEATURES], train["needs_correction"].astype(int))
        test["risk_score_model"] = _score_rows(model, test)
        test = test.sort_values(
            ["risk_score_model", "candidate_model_confidence", "candidate_model_margin", "label"],
            ascending=[False, True, True, True],
            kind="stable",
        ).reset_index(drop=True)
        test["rank_within_image_model"] = test.index + 1
        scored_frames.append(test)
        corrected = test.loc[test["needs_correction"]].copy()
        row = {
            "image_id": image_id,
            "n_rows": int(len(test)),
            "n_corrected": int(len(corrected)),
            "auc": float(roc_auc_score(test["needs_correction"].astype(int), test["risk_score_model"]))
            if test["needs_correction"].nunique() > 1
            else float("nan"),
        }
        for top_n in (5, 10, 20):
            row[f"top_{top_n}_catch_n"] = int(corrected["rank_within_image_model"].le(top_n).sum())
            row[f"top_{top_n}_catch_rate"] = (
                float(corrected["rank_within_image_model"].le(top_n).mean())
                if len(corrected)
                else float("nan")
            )
        rows.append(row)
    metrics = pd.DataFrame(rows)
    scored = pd.concat(scored_frames, ignore_index=True) if scored_frames else df.iloc[0:0].copy()
    return metrics, scored


def _load_true_iia_reviewed(pattern: str) -> pd.DataFrame:
    files = sorted(Path().glob(pattern))
    if not files:
        raise ValueError(f"No true_iia_hunt reviewed files matched: {pattern}")
    out = pd.concat([pd.read_csv(path, low_memory=False) for path in files], ignore_index=True)
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


def _build_gated_iix_cohort(
    *,
    model_path: Path,
    true_iia_reviewed: pd.DataFrame,
    gate_quantile: float,
    direct_root: Path,
    trusted_section_root: Path,
) -> pd.DataFrame:
    classifier = joblib.load(model_path)
    thresholds = derive_iia_gate_thresholds(true_iia_reviewed, gate_quantile)
    section_ids = _discover_section_ids(trusted_section_root, [])
    image_ids = _discover_cohort_ids(direct_root, trusted_section_root, section_ids)
    rows: list[pd.DataFrame] = []
    for image_id in image_ids:
        root = trusted_section_root if image_id in section_ids else direct_root
        fibers = pd.read_csv(root / image_id / f"{image_id}_fibers.csv", low_memory=False)
        diagnostics = pd.read_csv(
            root / image_id / f"{image_id}_feature_diagnostics.csv", low_memory=False
        )
        feat_cols = [c for c in classifier.feature_names_in_ if c in diagnostics.columns]
        x = diagnostics[feat_cols]
        pred = pd.Series(classifier.predict(x), index=diagnostics.index).astype(str).str.lower()
        proba = classifier.predict_proba(x)
        conf = pd.Series(proba.max(axis=1), index=diagnostics.index)
        top2 = (
            pd.DataFrame(proba)
            .apply(lambda row: row.nlargest(2).values, axis=1, result_type="expand")
        )
        margin = pd.Series(top2[0] - top2[1], index=diagnostics.index)
        gate_ok = _gate_mask(diagnostics, thresholds)
        pred_gated = _apply_iia_gate(pred, gate_ok)
        sub = diagnostics.copy()
        sub["image_id"] = image_id
        sub["label"] = fibers["label"].astype(int)
        sub["candidate_pred_gated"] = pred_gated.astype(str).str.lower()
        sub["candidate_model_confidence"] = conf.astype(float)
        sub["candidate_model_margin"] = margin.astype(float)
        sub["type1_signal_evidence"] = (
            fibers["type1_signal_evidence"].fillna("").astype(str).str.lower().str.strip()
        )
        sub["type2_signal_evidence"] = (
            fibers["type2_signal_evidence"].fillna("").astype(str).str.lower().str.strip()
        )
        sub["input_kind"] = (
            "section_tiff_export" if image_id in section_ids else "direct_czi"
        )
        sub["genotype"] = ""
        sub["timepoint"] = ""
        rows.append(
            sub.loc[
                sub["candidate_pred_gated"].eq("iix"),
                [
                    "image_id",
                    "label",
                    *[c for c in NUMERIC_FEATURES if c in sub.columns],
                    *[c for c in CAT_FEATURES if c in sub.columns],
                ],
            ].copy()
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    args = build_parser().parse_args()
    reviewed_iix = _load_reviewed_iix(args.reviewed_glob)
    feature_table = _load_feature_table(args.feature_table)
    training = _prepare_training_table(reviewed_iix, feature_table)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    loo_metrics, loo_scored = _leave_one_image_out_eval(training)
    loo_metrics.to_csv(args.output_dir / "pilot_iix_risk_ranker_leave_one_image_out.csv", index=False)
    loo_scored.to_csv(args.output_dir / "pilot_iix_risk_ranker_scored.csv", index=False)

    model = _build_pipeline()
    model.fit(training[NUMERIC_FEATURES + CAT_FEATURES], training["needs_correction"].astype(int))
    joblib.dump(model, args.output_dir / "iix_review_risk_ranker.joblib")

    true_iia = _load_true_iia_reviewed(args.true_iia_reviewed_glob)
    cohort = _build_gated_iix_cohort(
        model_path=args.classifier_path,
        true_iia_reviewed=true_iia,
        gate_quantile=args.gate_quantile,
        direct_root=args.direct_root,
        trusted_section_root=args.trusted_section_root,
    )
    cohort["risk_score_model"] = _score_rows(model, cohort)
    cohort = cohort.sort_values(
        ["image_id", "risk_score_model", "candidate_model_confidence", "candidate_model_margin", "label"],
        ascending=[True, False, True, True, True],
        kind="stable",
    )
    cohort["rank_within_image_model"] = cohort.groupby("image_id").cumcount() + 1
    cohort.to_csv(args.output_dir / "gated_iix_risk_ranked_cohort_model.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    for top_n in (5, 10, 20):
        selected = cohort.loc[cohort["rank_within_image_model"].le(top_n)].copy()
        summary_rows.append(
            {
                "policy": f"model_top_{top_n}_iix_per_image",
                "review_n": int(len(selected)),
                "n_images_with_iix": int(cohort["image_id"].nunique()),
                "cohort_total_fibers": 113721,
                "cohort_review_rate": float(len(selected) / 113721.0),
                "mean_reviews_per_image": float(selected.groupby("image_id").size().mean())
                if not selected.empty
                else 0.0,
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "gated_iix_risk_ranker_topn_summary.csv", index=False
    )
    print("saved:", args.output_dir)


if __name__ == "__main__":
    main()
