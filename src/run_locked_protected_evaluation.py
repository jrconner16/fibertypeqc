"""Fit the frozen development model once and evaluate it on sealed Notch3 holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.run_grouped_feature_ablation import (
    _build_model,
    _fold_metrics,
    _load_yaml,
    feature_columns,
    numeric_feature_table,
    prepare_development_table,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--development-features", type=Path, required=True)
    parser.add_argument("--protected-features", type=Path, required=True)
    parser.add_argument("--feature-set", default="F1")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def prepare_protected_table(table: pd.DataFrame, contract: dict[str, object]) -> pd.DataFrame:
    required = {
        "image_id",
        "group_id",
        "pipeline_label_id",
        "target_label",
        "label_authority",
        "evidence_role",
        "supervision_role",
        "eligible_for_model_fitting",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Protected feature table missing required columns: {missing}")
    if table.duplicated(["image_id", "pipeline_label_id"]).any():
        raise ValueError("Protected feature table has duplicate image/pipeline-label keys.")
    if not table["label_authority"].astype(str).eq("reviewed_myosight").all():
        raise ValueError("Protected labels must be uniformly reviewed_myosight.")
    if not table["evidence_role"].astype(str).eq("protected_final_evaluation").all():
        raise ValueError("Protected rows must be declared protected_final_evaluation.")
    if not table["supervision_role"].astype(str).eq("protected_final_evaluation").all():
        raise ValueError("Protected rows must be declared protected_final_evaluation.")
    if table["eligible_for_model_fitting"].astype(str).str.lower().eq("true").any():
        raise ValueError("Protected rows must never be eligible for model fitting.")
    classes = [str(value) for value in contract["target"]["classes"]]
    out = table.copy()
    out["target_label"] = out["target_label"].astype(str).str.strip().str.lower()
    if set(out["target_label"]).difference(classes):
        raise ValueError("Protected table has labels outside the frozen three-class target.")
    expected = contract["evidence_roles"]["protected_evaluation"]
    if out["group_id"].nunique() != int(expected["groups"]):
        raise ValueError("Protected table does not contain the frozen number of mouse groups.")
    return out.reset_index(drop=True)


def evaluate_locked_model(
    development: pd.DataFrame,
    protected: pd.DataFrame,
    contract: dict[str, object],
    feature_set: str,
) -> tuple[pd.DataFrame, pd.DataFrame, object, list[str]]:
    columns = feature_columns(contract, feature_set)
    model = _build_model(contract)
    model.fit(numeric_feature_table(development, columns), development["target_label"].astype(str))
    probabilities_raw = model.predict_proba(numeric_feature_table(protected, columns))
    classes = [str(value) for value in contract["target"]["classes"]]
    model_classes = list(model.named_steps["model"].classes_)
    probabilities = probabilities_raw[:, [model_classes.index(name) for name in classes]]
    predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
    predictions = protected.loc[
        :, ["image_id", "group_id", "pipeline_label_id", "target_label"]
    ].copy()
    predictions.insert(0, "feature_set", feature_set)
    predictions["predicted_label"] = predicted
    for index, name in enumerate(classes):
        predictions[f"prob_{name}"] = probabilities[:, index]
    predictions["model_confidence"] = probabilities.max(axis=1)
    predictions["model_margin"] = (
        np.sort(probabilities, axis=1)[:, -1] - np.sort(probabilities, axis=1)[:, -2]
    )

    rows: list[dict[str, object]] = []
    for group_id, indices in predictions.groupby("group_id", sort=True).groups.items():
        selected = np.asarray(list(indices), dtype=int)
        rows.append(
            {
                "scope_type": "mouse",
                "scope_id": group_id,
                "rows": int(len(selected)),
                **_fold_metrics(
                    protected.loc[selected, "target_label"].to_numpy(),
                    predicted[selected],
                    probabilities[selected],
                    classes,
                    contract,
                ),
            }
        )
    rows.append(
        {
            "scope_type": "overall",
            "scope_id": "all",
            "rows": int(len(protected)),
            **_fold_metrics(
                protected["target_label"].to_numpy(), predicted, probabilities, classes, contract
            ),
        }
    )
    return predictions, pd.DataFrame(rows), model, columns


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    contract = _load_yaml(args.contract)
    development = prepare_development_table(
        pd.read_csv(args.development_features, low_memory=False), contract
    )
    protected = prepare_protected_table(
        pd.read_csv(args.protected_features, low_memory=False), contract
    )
    predictions, metrics, model, columns = evaluate_locked_model(
        development, protected, contract, args.feature_set
    )
    args.output_dir.mkdir(parents=True)
    predictions.to_csv(args.output_dir / "protected_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "protected_metrics.csv", index=False)
    joblib.dump(
        {"feature_set": args.feature_set, "feature_columns": columns, "pipeline": model},
        args.output_dir / "locked_final_fit.joblib",
    )
    shutil.copy2(args.contract, args.output_dir / "resolved_contract.yaml")
    provenance = {
        "feature_set": args.feature_set,
        "feature_columns": columns,
        "development_feature_sha256": _sha256(args.development_features),
        "protected_feature_sha256": _sha256(args.protected_features),
        "contract_sha256": _sha256(args.contract),
        "development_rows_fit": int(len(development)),
        "protected_rows_fit": 0,
        "protected_rows_evaluated": int(len(protected)),
        "protected_mice": int(protected["group_id"].nunique()),
    }
    (args.output_dir / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"protected rows evaluated: {len(protected)}")
    print(f"protected mice: {protected['group_id'].nunique()}")


if __name__ == "__main__":
    main()
