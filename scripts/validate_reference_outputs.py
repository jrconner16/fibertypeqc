"""Validate the deterministic public reference inputs and generated outputs."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile

from fibertypeqc.model_manifest import load_model_manifest, validate_model_artifact

REFERENCE_CONTRACT_SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = REPO_ROOT / "examples/reference/reference_contract.json"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_contract(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if raw.get("schema_version") != REFERENCE_CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            f"Reference contract must use schema_version {REFERENCE_CONTRACT_SCHEMA_VERSION}."
        )
    return raw


def _assert_digest(path: Path, expected: str) -> None:
    actual = _digest(path)
    if actual != expected:
        raise ValueError(f"Digest mismatch for {path}: expected {expected}, got {actual}.")


def validate_reference_outputs(output_dir: Path, contract_path: Path = DEFAULT_CONTRACT) -> None:
    contract = _load_contract(contract_path)
    for relative_path, expected_digest in contract["fixtures"].items():
        _assert_digest(REPO_ROOT / relative_path, expected_digest)

    model_contract = contract["model"]
    model_path = REPO_ROOT / model_contract["artifact"]
    manifest = load_model_manifest(REPO_ROOT / model_contract["manifest"])
    validate_model_artifact(model_path, manifest)
    _assert_digest(model_path, model_contract["sha256"])

    outputs = {name: output_dir / filename for name, filename in contract["outputs"].items()}
    missing = [str(path) for path in outputs.values() if not path.is_file()]
    if missing:
        raise ValueError(f"Reference outputs are missing: {', '.join(missing)}")

    expected = contract["expected"]
    source_labels = np.asarray(
        tifffile.imread(REPO_ROOT / "examples/reference/synthetic_reference_labels.tif")
    )
    generated_labels = np.asarray(tifffile.imread(outputs["labels"]))
    if not np.array_equal(generated_labels, source_labels):
        raise ValueError("Generated reference labels differ from the supplied deterministic mask.")

    fibers = pd.read_csv(outputs["fibers"])
    missing_columns = sorted(set(expected["required_fiber_columns"]) - set(fibers.columns))
    if missing_columns:
        raise ValueError(f"Reference fiber table is missing columns: {', '.join(missing_columns)}")
    labels = fibers["label"].astype(int).tolist()
    if labels != expected["label_ids"]:
        raise ValueError(
            f"Reference label IDs differ: expected {expected['label_ids']}, got {labels}."
        )

    indexed = fibers.set_index(fibers["label"].astype(int))
    actual_types = {str(label): str(indexed.loc[label, "fiber_type"]) for label in labels}
    if actual_types != expected["fiber_type_by_label"]:
        raise ValueError(
            f"Reference fiber types differ: expected {expected['fiber_type_by_label']}, "
            f"got {actual_types}."
        )
    tolerance = float(expected["numeric_absolute_tolerance"])
    for raw_label, expected_value in expected["model_confidence_by_label"].items():
        label = int(raw_label)
        actual = float(indexed.loc[label, "model_confidence"])
        if not np.isclose(actual, float(expected_value), rtol=0.0, atol=tolerance):
            raise ValueError(
                f"Model confidence differs for label {label}: "
                f"expected {expected_value}, got {actual}."
            )

    summary = pd.read_csv(outputs["summary"], keep_default_na=False).iloc[0]
    if str(summary["qc_status"]) != expected["qc_status"]:
        raise ValueError(
            f"Reference QC status differs: expected {expected['qc_status']}, "
            f"got {summary['qc_status']}."
        )
    if str(summary["qc_reasons"]) != expected["qc_reasons"]:
        raise ValueError(
            f"Reference QC reasons differ: expected {expected['qc_reasons']!r}, "
            f"got {summary['qc_reasons']!r}."
        )

    run_manifest = json.loads(outputs["run_manifest"].read_text())
    panel = run_manifest.get("panel", {}).get("channels", {})
    expected_panel = {
        "dapi": None,
        "emhc": None,
        "laminin": 2,
        "type_i": None,
        "type_iia": 1,
        "type_iib": 0,
        "type_iix": None,
    }
    if panel != expected_panel:
        raise ValueError(f"Reference run panel differs: expected {expected_panel}, got {panel}.")
    if run_manifest.get("output_schema_version") != "legacy_fibers.v1":
        raise ValueError("Reference run used an unexpected fiber output schema.")
    if Path(str(run_manifest.get("source_image", ""))).is_absolute():
        raise ValueError("Reference run manifest must record a portable source-image path.")

    reviewed = pd.read_csv(outputs["reviewed_fibers"])
    reviewed_indexed = reviewed.set_index(reviewed["fiber_id"].astype(int))
    final_types = {
        str(label): str(reviewed_indexed.loc[label, "final_type"])
        for label in expected["label_ids"]
    }
    if final_types != expected["final_type_by_label"]:
        raise ValueError(
            f"Reference merge results differ: expected {expected['final_type_by_label']}, "
            f"got {final_types}."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory containing outputs from scripts.run_reference.",
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_reference_outputs(args.output_dir, args.contract)
    print(f"reference validation passed: {args.output_dir}")


if __name__ == "__main__":
    main()
