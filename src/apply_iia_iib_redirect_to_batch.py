from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.analyze_iia_gate import (
    _apply_iia_gate_with_iib_redirect,
    _gate_mask,
    _iib_redirect_mask,
    _load_benchmark_split,
    _load_feature_table,
    _load_true_iia_reviewed,
    derive_iia_gate_thresholds,
    derive_iib_redirect_thresholds,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an existing candidate batch output directory and apply the "
            "post-model IIa gate with optional IIb redirect using saved diagnostics."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--benchmark-split", type=Path, required=True)
    parser.add_argument("--true-iia-reviewed-glob", type=str, required=True)
    parser.add_argument("--gate-quantile", type=float, default=0.01)
    return parser


def _derive_thresholds(
    feature_table: pd.DataFrame,
    benchmark_split: pd.DataFrame,
    true_iia_reviewed: pd.DataFrame,
    gate_quantile: float,
) -> tuple[dict[str, float], dict[str, float]]:
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
    iia_thresholds = derive_iia_gate_thresholds(true_iia_reviewed, gate_quantile)
    iib_thresholds = derive_iib_redirect_thresholds(true_iib_reviewed, gate_quantile)
    return iia_thresholds, iib_thresholds


def _apply_to_image_dir(
    image_dir: Path, iia_thresholds: dict[str, float], iib_thresholds: dict[str, float]
) -> None:
    fiber_matches = list(image_dir.glob("*_fibers.csv"))
    diag_matches = list(image_dir.glob("*_feature_diagnostics.csv"))
    if len(fiber_matches) != 1 or len(diag_matches) != 1:
        return
    fibers_path = fiber_matches[0]
    diagnostics_path = diag_matches[0]
    fibers = pd.read_csv(fibers_path, low_memory=False)
    diagnostics = pd.read_csv(diagnostics_path, low_memory=False)
    pred_col = (
        "fiber_type_pre_iia_gate"
        if "fiber_type_pre_iia_gate" in diagnostics.columns
        else "fiber_type"
    )
    pred = diagnostics[pred_col].fillna("").astype(str).str.lower().str.strip()
    gate_ok = _gate_mask(diagnostics, iia_thresholds)
    iib_redirect_ok = _iib_redirect_mask(diagnostics, iib_thresholds)
    redirected = _apply_iia_gate_with_iib_redirect(pred, gate_ok, iib_redirect_ok)

    diagnostics = diagnostics.copy()
    diagnostics["fiber_type_pre_iia_gate"] = pred
    diagnostics["gate_iia_ok"] = gate_ok
    diagnostics["gate_iib_redirect_ok"] = iib_redirect_ok
    diagnostics["fiber_type"] = redirected
    diagnostics.to_csv(diagnostics_path, index=False)

    fibers = fibers.copy()
    fibers["label"] = fibers["label"].astype(int)
    update = pd.DataFrame(
        {
            "label": diagnostics["label"].astype(int),
            "fiber_type": redirected,
            "fiber_type_pre_iia_gate": pred,
            "gate_iia_ok": gate_ok,
            "gate_iib_redirect_ok": iib_redirect_ok,
        }
    )
    fibers = fibers.drop(
        columns=["fiber_type_pre_iia_gate", "gate_iia_ok", "gate_iib_redirect_ok"],
        errors="ignore",
    ).merge(update, on="label", how="left", validate="one_to_one", suffixes=("", "_new"))
    fibers["fiber_type"] = fibers["fiber_type_new"].fillna(fibers["fiber_type"]).astype(str)
    fibers = fibers.drop(columns=["fiber_type_new"])
    fibers.to_csv(fibers_path, index=False)


def main() -> None:
    args = build_parser().parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")
    if args.output_dir.exists():
        raise SystemExit(f"Output directory already exists: {args.output_dir}")
    feature_table = _load_feature_table(args.feature_table)
    benchmark_split = _load_benchmark_split(args.benchmark_split)
    true_iia_reviewed = _load_true_iia_reviewed(args.true_iia_reviewed_glob)
    iia_thresholds, iib_thresholds = _derive_thresholds(
        feature_table, benchmark_split, true_iia_reviewed, args.gate_quantile
    )

    shutil.copytree(args.input_dir, args.output_dir)
    for image_dir in sorted(p for p in args.output_dir.iterdir() if p.is_dir()):
        _apply_to_image_dir(image_dir, iia_thresholds, iib_thresholds)
    print(f"saved redirected batch dir: {args.output_dir}")


if __name__ == "__main__":
    main()
