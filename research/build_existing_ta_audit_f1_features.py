"""Join predeclared existing-TA manual audit labels to rebuilt F1 diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

F1_COLUMNS = [
    "type_iib.mean",
    "type_iia.mean",
    "type_iia.p90",
    "type_iia.coverage_high",
    "type_iia.snr_mean",
    "type_iia.snr_p90",
    "type_iia.center_mean",
    "type_iia.edge_mean",
    "type_iib.p90",
    "type_iib.coverage_high",
    "type_iib.snr_mean",
    "type_iib.snr_p90",
    "type_iib.center_mean",
    "type_iib.edge_mean",
]
TARGET_CLASSES = {"iia", "iib", "iix"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_existing_ta_features(audit: pd.DataFrame, diagnostics_root: Path) -> pd.DataFrame:
    required = {"image_id", "label", "audit_final_label", "manual_supervision_weight"}
    missing = sorted(required.difference(audit.columns))
    if missing:
        raise ValueError(f"Audit table missing required columns: {missing}")
    selected = audit.loc[
        audit["manual_supervision_weight"].eq(1.0)
        & audit["audit_final_label"].astype(str).str.lower().isin(TARGET_CLASSES)
    ].copy()
    if selected.empty:
        raise ValueError("No eligible manual-audit three-class training rows.")
    if selected.duplicated(["image_id", "label"]).any():
        raise ValueError("Audit table has duplicate image/label keys after filtering.")

    frames: list[pd.DataFrame] = []
    for image_id, rows in selected.groupby("image_id", sort=True):
        image_id = str(image_id)
        diagnostics_path = diagnostics_root / image_id / f"{image_id}_feature_diagnostics.csv"
        if not diagnostics_path.is_file():
            raise FileNotFoundError(
                f"Missing rebuilt diagnostics for {image_id}: {diagnostics_path}"
            )
        diagnostics = pd.read_csv(diagnostics_path, low_memory=False)
        missing_features = sorted(set(F1_COLUMNS).difference(diagnostics.columns))
        if missing_features:
            raise ValueError(f"Diagnostics for {image_id} missing F1 columns: {missing_features}")
        if diagnostics["label"].duplicated().any():
            raise ValueError(f"Diagnostics for {image_id} has duplicate labels.")
        merged = rows.loc[:, ["image_id", "label", "audit_final_label"]].merge(
            diagnostics.loc[:, ["label", *F1_COLUMNS]],
            on="label",
            how="left",
            validate="one_to_one",
        )
        if merged[F1_COLUMNS].isna().any().any():
            raise ValueError(f"Audit labels are absent from rebuilt diagnostics for {image_id}.")
        frames.append(merged)
    result = pd.concat(frames, ignore_index=True)
    result.insert(0, "cohort_id", "og_jag_ta_manual_audit")
    result.insert(2, "group_id", result["image_id"].astype(str))
    result.insert(3, "pipeline_label_id", result["label"])
    result.insert(4, "target_label", result.pop("audit_final_label").astype(str).str.lower())
    result.insert(5, "label_authority", "manual_audit")
    result.insert(6, "evidence_role", "development_grouped_resampling")
    result.insert(7, "supervision_role", "development_model_selection")
    result.insert(8, "eligible_for_model_fitting", True)
    return result.drop(columns="label")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-table", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.provenance_output.exists():
        raise FileExistsError("Refusing to overwrite an existing output or provenance file.")
    audit = pd.read_csv(args.audit_table, low_memory=False)
    result = build_existing_ta_features(audit, args.diagnostics_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    provenance = {
        "audit_table_sha256": _sha256(args.audit_table),
        "rows": len(result),
        "groups": result["group_id"].nunique(),
        "class_counts": result["target_label"].value_counts().sort_index().to_dict(),
        "f1_columns": F1_COLUMNS,
        "protected_notch3_rows_read": 0,
    }
    args.provenance_output.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"rows: {len(result)}")
    print(f"source-image groups: {result['group_id'].nunique()}")


if __name__ == "__main__":
    main()
