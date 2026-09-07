"""Extract frozen-contract F3 pixel features from cached P1 masks and source images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import find_objects

from src.io_utils import load_multichannel_image
from src.label_masks import erode_labels
from src.quantify_classify import _preprocess_typing_channel

MARKERS = ("iia", "iib")
F3_SUFFIXES = (
    "pixel_entropy_norm_32",
    "pixel_mad_over_positive_median",
    "pixel_iqr_over_positive_median",
    "p90_over_positive_mean",
    "center_edge_difference_over_sum",
)
EPSILON = 1.0e-6


def f3_columns() -> tuple[str, ...]:
    return tuple(f"type_{marker}.{suffix}" for marker in MARKERS for suffix in F3_SUFFIXES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_distribution_features(values: np.ndarray) -> tuple[float, float, float, float]:
    """Return entropy, MAD/median, IQR/median, and p90/mean for standardized pixels."""
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        raise ValueError("F3 cannot measure an empty typing-interior mask.")
    counts, _ = np.histogram(values, bins=32, range=(0.0, 32.0))
    probabilities = counts[counts > 0].astype(np.float64) / float(values.size)
    entropy = float(-(probabilities * np.log(probabilities)).sum() / np.log(32.0))
    positive = values[values > 0]
    if positive.size == 0:
        return entropy, 0.0, 0.0, 0.0
    median = float(np.median(positive))
    denominator = max(median, EPSILON)
    mad = float(np.median(np.abs(positive - median)))
    q25, q75, p90 = np.percentile(positive, [25.0, 75.0, 90.0])
    return (
        entropy,
        mad / denominator,
        float(q75 - q25) / denominator,
        float(p90) / max(float(positive.mean()), EPSILON),
    )


def extract_image_f3(
    *,
    labels: np.ndarray,
    image_chw: np.ndarray,
    marker_channels: dict[str, int],
    label_ids: np.ndarray,
    typing_erode_px: int = 2,
    typing_bg_quantile: float = 0.02,
    typing_tile_size: int = 256,
    typing_smooth_sigma: float = 0.9,
) -> pd.DataFrame:
    """Measure F3 on requested labels using P1 typing interiors and preprocessing."""
    labels = np.asarray(labels, dtype=np.int32)
    if labels.ndim != 2:
        raise ValueError("Cached labels must be a 2-D array.")
    if image_chw.ndim != 3 or tuple(image_chw.shape[1:]) != tuple(labels.shape):
        raise ValueError("Source image and cached label dimensions do not agree.")
    label_ids = np.asarray(label_ids, dtype=np.int32)
    if label_ids.size == 0 or np.any(label_ids <= 0) or len(np.unique(label_ids)) != len(label_ids):
        raise ValueError("Requested F3 label IDs must be unique positive integers.")
    available = set(np.unique(labels))
    missing = sorted(set(label_ids).difference(available))
    if missing:
        raise ValueError(f"Requested F3 labels are absent from cached mask: {missing[:5]}")
    if set(marker_channels) != set(MARKERS):
        raise ValueError("F3 requires exactly iia and iib channel mappings.")

    interiors = erode_labels(labels, typing_erode_px)
    # This is the existing F1 spatial convention accepted for F3: center is a two-pixel
    # erosion, edge is the original fiber mask excluding that center.
    centers = erode_labels(labels, typing_erode_px)
    edges = labels.copy()
    edges[centers > 0] = 0
    object_slices = find_objects(labels)
    result: dict[str, np.ndarray] = {"label": label_ids.copy()}

    for marker in MARKERS:
        channel_index = marker_channels[marker]
        if channel_index < 0 or channel_index >= image_chw.shape[0]:
            raise ValueError(f"F3 channel index {channel_index} is invalid for marker {marker!r}.")
        processed = _preprocess_typing_channel(
            image_chw[channel_index],
            mode="tile_subtract",
            bg_quantile=typing_bg_quantile,
            tile_size=typing_tile_size,
            smooth_sigma=typing_smooth_sigma,
        )
        tissue_values = processed[interiors > 0]
        if tissue_values.size == 0:
            raise ValueError("F3 tissue mask is empty.")
        tissue_median = float(np.median(tissue_values))
        tissue_mad = float(np.median(np.abs(tissue_values - tissue_median)))
        standardized = np.clip((processed - tissue_median) / max(tissue_mad, EPSILON), 0.0, 32.0)
        values = np.empty((len(label_ids), len(F3_SUFFIXES)), dtype=np.float32)
        for index, label_id in enumerate(label_ids):
            slc = object_slices[int(label_id) - 1]
            if slc is None:
                raise ValueError(f"Requested F3 label {label_id} has no object slice.")
            local_standardized = standardized[slc]
            interior = local_standardized[interiors[slc] == label_id]
            entropy, mad_ratio, iqr_ratio, p90_ratio = _positive_distribution_features(interior)
            center = local_standardized[centers[slc] == label_id]
            edge = local_standardized[edges[slc] == label_id]
            if center.size == 0 or edge.size == 0:
                raise ValueError(f"F3 center or edge mask is empty for label {label_id}.")
            center_mean, edge_mean = float(center.mean()), float(edge.mean())
            center_edge = (center_mean - edge_mean) / max(center_mean + edge_mean, EPSILON)
            values[index] = (entropy, mad_ratio, iqr_ratio, p90_ratio, center_edge)
        for index, suffix in enumerate(F3_SUFFIXES):
            result[f"type_{marker}.{suffix}"] = values[:, index]
    frame = pd.DataFrame(result)
    if not np.isfinite(frame.loc[:, f3_columns()].to_numpy(dtype=float)).all():
        raise ValueError("F3 extraction produced a missing or non-finite feature.")
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument(
        "--recorded-source-root",
        type=Path,
        required=True,
        help="Source-image root recorded in the baseline run JSON files.",
    )
    parser.add_argument("--source-image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    return parser


def resolve_source_image(
    recorded_source: str | Path,
    *,
    recorded_source_root: Path,
    source_image_root: Path,
) -> Path:
    """Map a run-record source path onto the caller-supplied local source root."""
    try:
        relative = Path(recorded_source).relative_to(recorded_source_root)
    except ValueError as exc:
        raise ValueError(
            f"Recorded source image is outside --recorded-source-root: {recorded_source}"
        ) from exc
    return source_image_root / relative


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists() or args.provenance_output.exists():
        raise FileExistsError("Refusing to overwrite an existing F3 artifact.")
    supervision = pd.read_csv(args.supervision, low_memory=False)
    required = {"image_id", "pipeline_label_id", "supervision_role", "eligible_for_model_fitting"}
    missing = sorted(required.difference(supervision.columns))
    if missing:
        raise ValueError(f"Supervision is missing required columns: {missing}")
    if not supervision["supervision_role"].eq("development_model_selection").all():
        raise ValueError("F3 extraction accepts development-model-selection rows only.")
    if not supervision["eligible_for_model_fitting"].astype(str).str.lower().eq("true").all():
        raise ValueError("F3 extraction accepts fitting-eligible rows only.")
    if supervision.duplicated(["image_id", "pipeline_label_id"]).any():
        raise ValueError("F3 supervision keys must be unique.")

    rows: list[pd.DataFrame] = []
    images: dict[str, dict[str, str]] = {}
    for image_id, subset in supervision.groupby("image_id", sort=True):
        run_paths = list((args.baseline_root / str(image_id)).glob("*_run.json"))
        label_paths = list((args.baseline_root / str(image_id)).glob("*_cellpose_labels.tif"))
        if len(run_paths) != 1 or len(label_paths) != 1:
            raise ValueError(f"Expected exactly one run JSON and mask TIFF for {image_id}.")
        run = json.loads(run_paths[0].read_text(encoding="utf-8"))
        panel = run.get("panel", {}).get("channels", {})
        marker_channels = {marker: int(panel[f"type_{marker}"]) for marker in MARKERS}
        source = resolve_source_image(
            run["source_image"],
            recorded_source_root=args.recorded_source_root,
            source_image_root=args.source_image_root,
        )
        if not source.exists():
            raise FileNotFoundError(f"Source CZI missing for {image_id}: {source}")
        features = extract_image_f3(
            labels=tifffile.imread(label_paths[0]),
            image_chw=load_multichannel_image(source),
            marker_channels=marker_channels,
            label_ids=pd.to_numeric(subset["pipeline_label_id"], errors="raise").to_numpy(),
        )
        features.insert(0, "image_id", str(image_id))
        rows.append(features)
        images[str(image_id)] = {
            "labels_sha256": _sha256(label_paths[0]),
            "run_sha256": _sha256(run_paths[0]),
            "source_sha256": _sha256(source),
        }
        print(f"extracted F3: {image_id} ({len(features)} labels)", flush=True)
    output = pd.concat(rows, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    provenance = {
        "feature_set": "F3",
        "rows": int(len(output)),
        "images": int(output["image_id"].nunique()),
        "source_supervision_sha256": _sha256(args.supervision),
        "frozen_typing_preprocessing": {
            "mode": "tile_subtract",
            "bg_quantile": 0.02,
            "tile_size": 256,
            "effective_smooth_sigma": 0.9,
            "typing_erode_px": 2,
        },
        "center_edge_mask": (
            "center=each original fiber eroded by 2 px; edge=original fiber excluding center"
        ),
        "features": list(f3_columns()),
        "protected_holdout_rows_read": 0,
        "input_hashes": images,
    }
    args.provenance_output.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"saved F3 rows: {len(output)}")


if __name__ == "__main__":
    main()
