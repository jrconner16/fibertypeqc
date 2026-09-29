"""Generate the public-safe synthetic four-class (I/IIa/IIb/residual IIx) reference.

Everything here is synthetic: a four-channel image with square fibers, its label mask, a small
random forest fitted on features extracted from a separate synthetic training image, and a model
manifest that pins the feature-extraction settings. It exercises the pinned semantic-model path
end to end; it is not biological evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import tifffile
import yaml
from sklearn.ensemble import RandomForestClassifier

from fibertypeqc.model_manifest import FEATURE_EXTRACTION_FIELDS
from src.quantify_classify import (
    QuantifyConfig,
    build_feature_diagnostics_table,
    quantify_labels,
)

OUTPUT_DIR = Path("examples/reference_four_class")
CHANNELS = {"type_i": 0, "type_iia": 1, "laminin": 2, "type_iib": 3}
CLASSES = ("i", "iia", "iib", "iix")
FEATURES = [
    "area",
    *[
        f"{marker}.{suffix}"
        for marker in ("type_i", "type_iia", "type_iib")
        for suffix in ("mean", "p75", "p90", "coverage_high", "snr_mean", "snr_p90")
    ],
]
PINNED = QuantifyConfig(typing_preprocess="global_subtract", typing_erode_px=2)
FIBER = 24
STEP = 31


def _draw(n_rows: int, n_cols: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Grid of square fibers; each row cycles through the four classes."""
    rng = np.random.default_rng(seed)
    size = 5 + STEP * max(n_rows, n_cols)
    image = np.full((4, size, size), 100, dtype=np.uint16)
    labels = np.zeros((size, size), dtype=np.uint16)
    truth: list[str] = []
    label_id = 0
    for row in range(n_rows):
        for col in range(n_cols):
            label_id += 1
            y0, x0 = 5 + row * STEP, 5 + col * STEP
            region = np.s_[y0 : y0 + FIBER, x0 : x0 + FIBER]
            labels[region] = label_id
            fiber_class = CLASSES[(row + col) % len(CLASSES)]
            truth.append(fiber_class)
            for marker in ("type_i", "type_iia", "type_iib"):
                positive = marker == {"i": "type_i", "iia": "type_iia", "iib": "type_iib"}.get(
                    fiber_class
                )
                level = rng.integers(3000, 4000) if positive else rng.integers(140, 220)
                image[CHANNELS[marker]][region] = level
            lam = CHANNELS["laminin"]
            image[lam, y0 : y0 + FIBER, x0] = 5000
            image[lam, y0 : y0 + FIBER, x0 + FIBER - 1] = 5000
            image[lam, y0, x0 : x0 + FIBER] = 5000
            image[lam, y0 + FIBER - 1, x0 : x0 + FIBER] = 5000
    return image, labels, truth


def _features(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    cfg = QuantifyConfig(
        type1_channel=CHANNELS["type_iib"],
        type2_channel=CHANNELS["type_iia"],
        i_channel=CHANNELS["type_i"],
        **{field: getattr(PINNED, field) for field in FEATURE_EXTRACTION_FIELDS},
    )
    fibers = quantify_labels(labels.astype(np.int32), image, cfg)
    diagnostics = build_feature_diagnostics_table(fibers, cfg)
    return diagnostics.loc[:, FEATURES].to_numpy(dtype=float)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_image, train_labels, train_truth = _draw(8, 8, seed=1)
    model = RandomForestClassifier(n_estimators=25, random_state=0, min_samples_leaf=2)
    model.fit(_features(train_image, train_labels), train_truth)
    model_path = output_dir / "synthetic_four_class_rf.joblib"
    joblib.dump(model, model_path)

    image, labels, truth = _draw(4, 4, seed=2)
    image_path = output_dir / "synthetic_four_class.tif"
    labels_path = output_dir / "synthetic_four_class_labels.tif"
    tifffile.imwrite(
        image_path,
        image,
        imagej=True,
        resolution=(2.0, 2.0),
        metadata={"axes": "CYX", "unit": "um"},
    )
    tifffile.imwrite(labels_path, labels, imagej=True, metadata={"axes": "YX"})

    panel = {
        "channels": {
            "laminin": CHANNELS["laminin"],
            "dapi": None,
            "type_i": CHANNELS["type_i"],
            "type_iia": CHANNELS["type_iia"],
            "type_iib": CHANNELS["type_iib"],
            "type_iix": None,
            "emhc": None,
        },
        "classification": {
            "residual_inference": {
                "enabled": True,
                "target_class": "iix",
                "requires_negative_markers": ["i", "iia", "iib"],
            }
        },
    }
    (output_dir / "panel.yaml").write_text(
        "# Synthetic four-marker panel: IIx is inferred from absent I, IIa, and IIb signal.\n"
        + yaml.safe_dump(panel, sort_keys=False)
    )
    pinned = {field: getattr(PINNED, field) for field in sorted(FEATURE_EXTRACTION_FIELDS)}
    manifest = {
        "manifest_version": 1,
        "model_id": "synthetic_four_class_reference_v1",
        "task": "fiber_identity",
        "feature_schema_version": "multiplanel_features.v1",
        "required_markers": ["laminin", "type_i", "type_iia", "type_iib"],
        "outputs": list(CLASSES),
        "artifact": model_path.name,
        "artifact_sha256": _sha256(model_path),
        "intended_use": "Synthetic mechanics reference for the pinned four-class model path.",
        "features": FEATURES,
        "feature_extraction": pinned,
    }
    (output_dir / "synthetic_four_class_rf.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False)
    )
    expected = {str(label): fiber_class for label, fiber_class in enumerate(truth, start=1)}
    (output_dir / "expected_fiber_types.json").write_text(
        json.dumps(
            {"fiber_type_by_label": expected},
            indent=2,
        )
        + "\n"
    )
    print(f"wrote synthetic four-class reference to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    generate(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
