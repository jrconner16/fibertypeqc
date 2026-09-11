"""Score one existing Jag1 Quad section with the frozen private candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.run_jag1_quad_development_baseline import FEATURES, _section_features


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _locked_model(lock_dir: Path, manifest: Path):
    lock = json.loads((lock_dir / "final_evaluation_lock.json").read_text())
    model_path = lock_dir / "jag1_quad_four_class_random_forest_v2.joblib"
    if _sha256(model_path) != lock["locked_model"]["sha256"]:
        raise ValueError("Locked model digest does not match final_evaluation_lock.json")
    expected_manifest = lock["artifacts_before_copy"]["canonical_manifest"]["sha256"]
    if _sha256(manifest) != expected_manifest:
        raise ValueError("Manifest digest does not match the frozen candidate lock")
    metadata = json.loads((lock_dir / "baseline_metadata.json").read_text())
    if metadata.get("features") != list(FEATURES):
        raise ValueError("Frozen metadata feature list does not match the scoring feature contract")
    return joblib.load(model_path)


def score_section(section: pd.Series, model: object) -> tuple[pd.DataFrame, dict[str, object]]:
    features = _section_features(section)
    missing = sorted(set(FEATURES) - set(features.columns))
    if missing:
        raise ValueError(f"Missing frozen candidate features: {missing}")
    x = features.loc[:, FEATURES]
    prediction = model.predict(x).astype(str)
    probabilities = np.asarray(model.predict_proba(x), dtype=float)
    classes = [str(value) for value in model.classes_]
    out = features.loc[
        :, ["mouse_id", "cre_status", "image_id", "section_id", "pipeline_fiber_id"]
    ].copy()
    out = out.rename(columns={"pipeline_fiber_id": "fiber_id"})
    out["locked_prediction"] = prediction
    for index, label in enumerate(classes):
        out[f"prob_{label}"] = probabilities[:, index]
    ranked = np.sort(probabilities, axis=1)
    out["model_confidence"] = ranked[:, -1]
    out["model_margin"] = ranked[:, -1] - ranked[:, -2]
    counts = out["locked_prediction"].value_counts()
    summary: dict[str, object] = {
        "mouse_id": str(section.mouse_id),
        "cre_status": str(section.cre_status),
        "image_id": str(section.image_id),
        "section_id": str(section.section_id),
        "analysis_eligible": str(section.mouse_id) != "351545_R",
        "n_fibers": int(len(out)),
    }
    for label in classes:
        summary[f"n_{label}"] = int(counts.get(label, 0))
        summary[f"pct_{label}"] = float(counts.get(label, 0) / len(out))
    return out, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Score one locked Jag1 Quad candidate section")
    parser.add_argument("--lock-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    args = parser.parse_args()
    manifest = (
        pd.read_csv(args.manifest, dtype=str)
        .fillna("")
        .sort_values(["mouse_id", "image_id"], kind="stable")
        .reset_index(drop=True)
    )
    if args.task_index < 0 or args.task_index >= len(manifest):
        raise IndexError(f"--task-index must be in 0..{len(manifest) - 1}")
    section = manifest.iloc[args.task_index]
    output = args.output_dir.resolve()
    sections = output / "sections"
    sections.mkdir(parents=True, exist_ok=True)
    prediction_path = sections / f"{section.image_id}_locked_predictions.csv"
    summary_path = sections / f"{section.image_id}_locked_summary.json"
    if prediction_path.exists() or summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite section output for {section.image_id}")
    predictions, summary = score_section(
        section, _locked_model(args.lock_dir.resolve(), args.manifest.resolve())
    )
    predictions.to_csv(prediction_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"scored {section.image_id}: {prediction_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
