"""Fit locked Notch3 feature sets once and characterize fixed-mask Jag transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fibertypeqc.experimental_features import (
    RELATIVE_CHANNEL_FEATURES,
    add_relative_channel_features,
)
from src.run_grouped_feature_ablation import (
    _build_model,
    _fold_metrics,
    _load_yaml,
    feature_columns,
    numeric_feature_table,
    prepare_development_table,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the frozen Notch3 development recipe once, then apply it unchanged to the "
            "original-Jag fixed-mask transfer ledger."
        )
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--development-features", type=Path, required=True)
    parser.add_argument("--development-diagnostics-inventory", type=Path, required=True)
    parser.add_argument("--transfer-features", type=Path, required=True)
    parser.add_argument("--transfer-provenance", type=Path, required=True)
    parser.add_argument("--feature-set", action="append", dest="feature_sets")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_transfer_table(table: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    required = {
        "image_id",
        "group_id",
        "pipeline_label_id",
        "target_label",
        "label_authority",
        "evidence_role",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Transfer table missing required columns: {missing}")
    if table.duplicated(["image_id", "pipeline_label_id"]).any():
        raise ValueError("Transfer table contains duplicate image/pipeline-label keys.")
    if not table["label_authority"].astype(str).eq("myosight_derived").all():
        raise ValueError("Transfer table must be uniformly myosight_derived.")
    if not table["evidence_role"].astype(str).eq("development_transfer").all():
        raise ValueError("Transfer table must be uniformly development_transfer.")
    out = table.copy()
    out["target_label"] = out["target_label"].astype(str).str.strip().str.lower()
    unexpected = sorted(set(out["target_label"]).difference(classes))
    if unexpected:
        raise ValueError(f"Transfer table contains unexpected target labels: {unexpected}")
    return out.reset_index(drop=True)


def _with_relative_features(
    table: pd.DataFrame, columns: list[str], epsilon: float
) -> pd.DataFrame:
    if set(columns).intersection(RELATIVE_CHANNEL_FEATURES):
        return add_relative_channel_features(table, epsilon=epsilon)
    return table


def characterize_feature_set(
    development: pd.DataFrame,
    transfer: pd.DataFrame,
    contract: dict[str, object],
    feature_set: str,
) -> tuple[pd.DataFrame, pd.DataFrame, object, list[str]]:
    columns = feature_columns(contract, feature_set)
    epsilon = float(contract["numeric_contract"]["epsilon"])
    development = _with_relative_features(development, columns, epsilon)
    transfer = _with_relative_features(transfer, columns, epsilon)
    development_x = numeric_feature_table(development, columns)
    transfer_x = numeric_feature_table(transfer, columns)
    classes = [str(value) for value in contract["target"]["classes"]]

    model = _build_model(contract)
    model.fit(development_x, development["target_label"].astype(str))
    raw_probabilities = model.predict_proba(transfer_x)
    model_classes = list(model.named_steps["model"].classes_)
    order = [model_classes.index(name) for name in classes]
    probabilities = raw_probabilities[:, order]
    predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    margin = np.sort(probabilities, axis=1)[:, -1] - np.sort(probabilities, axis=1)[:, -2]

    predictions = transfer.loc[
        :, ["image_id", "group_id", "pipeline_label_id", "target_label"]
    ].copy()
    predictions.insert(0, "feature_set", feature_set)
    predictions["predicted_label"] = predicted
    for index, name in enumerate(classes):
        predictions[f"prob_{name}"] = probabilities[:, index]
    predictions["model_confidence"] = confidence
    predictions["model_margin"] = margin

    metric_rows: list[dict[str, object]] = []
    for image_id, indices in predictions.groupby("image_id", sort=True).groups.items():
        selected = np.asarray(list(indices), dtype=int)
        metric_rows.append(
            {
                "feature_set": feature_set,
                "scope_type": "image",
                "scope_id": image_id,
                "rows": int(len(selected)),
                **_fold_metrics(
                    transfer.loc[selected, "target_label"].to_numpy(),
                    predicted[selected],
                    probabilities[selected],
                    classes,
                    contract,
                ),
            }
        )
    metric_rows.append(
        {
            "feature_set": feature_set,
            "scope_type": "overall",
            "scope_id": "all",
            "rows": int(len(transfer)),
            **_fold_metrics(
                transfer["target_label"].to_numpy(), predicted, probabilities, classes, contract
            ),
        }
    )
    return predictions, pd.DataFrame(metric_rows), model, columns


def main() -> None:
    args = build_parser().parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    contract = _load_yaml(args.contract)
    selected = args.feature_sets or ["F1", "F2"]
    development = prepare_development_table(
        pd.read_csv(args.development_features, low_memory=False), contract
    )
    classes = [str(value) for value in contract["target"]["classes"]]
    transfer = prepare_transfer_table(
        pd.read_csv(args.transfer_features, low_memory=False), classes
    )

    args.output_dir.mkdir(parents=True)
    predictions: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    feature_columns_by_set: dict[str, list[str]] = {}
    for feature_set in selected:
        prediction, metric, model, columns = characterize_feature_set(
            development, transfer, contract, feature_set
        )
        predictions.append(prediction)
        metrics.append(metric)
        feature_columns_by_set[feature_set] = columns
        joblib.dump(
            {
                "feature_set": feature_set,
                "feature_columns": columns,
                "classes": classes,
                "fit_scope": "all_notch3_development_rows",
                "pipeline": model,
            },
            args.output_dir / f"{feature_set}_notch3_fit.joblib",
        )

    all_predictions = pd.concat(predictions, ignore_index=True)
    all_metrics = pd.concat(metrics, ignore_index=True)
    all_predictions.to_csv(args.output_dir / "transfer_predictions.csv", index=False)
    all_metrics.to_csv(args.output_dir / "transfer_metrics.csv", index=False)
    args.output_dir.joinpath("resolved_contract.yaml").write_text(args.contract.read_text())
    provenance = {
        "contract_sha256": _sha256(args.contract),
        "development_features_sha256": _sha256(args.development_features),
        "development_diagnostics_inventory_sha256": _sha256(args.development_diagnostics_inventory),
        "transfer_features_sha256": _sha256(args.transfer_features),
        "transfer_provenance_sha256": _sha256(args.transfer_provenance),
        "feature_sets": selected,
        "feature_columns": feature_columns_by_set,
        "development_rows": int(len(development)),
        "development_groups": int(development["group_id"].nunique()),
        "transfer_rows": int(len(transfer)),
        "transfer_images": int(transfer["image_id"].nunique()),
        "transfer_fit": "fit once on all Notch3 development rows; no Jag rows used for fitting",
    }
    (args.output_dir / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(f"feature sets: {', '.join(selected)}")
    print(f"development rows/groups: {len(development)} / {development['group_id'].nunique()}")
    print(f"transfer rows/images: {len(transfer)} / {transfer['image_id'].nunique()}")
    print(f"saved transfer characterization: {args.output_dir}")


if __name__ == "__main__":
    main()
