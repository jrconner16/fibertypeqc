"""Segment DAPI nuclei and associate them with cached fiber labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

from src.dapi_preprocess import tile_background_subtract, tile_percentile_normalize
from src.io_utils import load_multichannel_image, save_dataframe, save_labels
from src.nuclear_association import associate_nuclei, summarize_fiber_nuclei
from src.preprocess_membrane import upsample_labels_nearest
from src.segment_cellpose import CellposeConfig, run_cellpose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--fiber-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dapi-channel", type=int, required=True)
    parser.add_argument(
        "--dapi-preprocess",
        choices=["raw", "tile_subtract", "tile_normalize"],
        default="raw",
        help="Optional local DAPI background correction before Cellpose.",
    )
    parser.add_argument("--dapi-tile-size", type=int, default=512)
    parser.add_argument("--dapi-background-quantile", type=float, default=0.02)
    parser.add_argument("--dapi-low-percentile", type=float, default=1.0)
    parser.add_argument("--dapi-high-percentile", type=float, default=99.8)
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("Y0", "Y1", "X0", "X1"),
        help="Optional pixel crop for fast troubleshooting: Y0 Y1 X0 X1.",
    )
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--diameter", type=float, default=15.0)
    parser.add_argument("--min-size", type=int, default=30)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellpose-model", type=str, default="cpsam")
    parser.add_argument("--bsize", type=int, default=256)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--cellpose-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize DAPI intensity before Cellpose (enabled by default).",
    )
    parser.add_argument(
        "--reuse-artifacts",
        action="store_true",
        help="Reuse an existing nuclear-label TIFF in the output directory.",
    )
    parser.add_argument("--min-overlap-fraction", type=float, default=0.5)
    parser.add_argument("--boundary-distance-px", type=float, default=3.0)
    parser.add_argument("--central-normalized-radius", type=float, default=0.5)
    args = parser.parse_args()
    if args.downsample_factor < 1:
        raise ValueError("--downsample-factor must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = load_multichannel_image(args.input)
    if args.dapi_channel < 0 or args.dapi_channel >= image.shape[0]:
        raise ValueError(
            f"DAPI channel {args.dapi_channel} is out of range for {image.shape[0]} channels"
        )
    fiber_labels = np.asarray(tifffile.imread(args.fiber_labels)).astype(np.int32)
    if fiber_labels.shape != image.shape[1:]:
        raise ValueError(
            f"Fiber labels have shape {fiber_labels.shape}, expected {image.shape[1:]}"
        )
    crop = args.crop
    if crop is not None:
        y0, y1, x0, x1 = crop
        height, width = image.shape[1:]
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise ValueError(f"Crop {crop} is outside image bounds {(height, width)}")
        image = image[:, y0:y1, x0:x1]
        fiber_labels = fiber_labels[y0:y1, x0:x1]

    stem = args.input.stem.replace(" ", "_")
    nuclei_labels_path = args.output_dir / f"{stem}_nuclei_labels.tif"
    dapi = image[args.dapi_channel]
    if args.dapi_preprocess == "tile_subtract":
        dapi = tile_background_subtract(
            dapi,
            tile_size=args.dapi_tile_size,
            background_quantile=args.dapi_background_quantile,
        )
    elif args.dapi_preprocess == "tile_normalize":
        dapi = tile_percentile_normalize(
            dapi,
            tile_size=args.dapi_tile_size,
            low_percentile=args.dapi_low_percentile,
            high_percentile=args.dapi_high_percentile,
        )
    factor = args.downsample_factor
    reused_nuclei_labels = bool(args.reuse_artifacts and nuclei_labels_path.exists())
    if reused_nuclei_labels:
        nuclei_labels = np.asarray(tifffile.imread(nuclei_labels_path)).astype(np.int32)
        if nuclei_labels.shape != dapi.shape:
            raise ValueError(
                f"Cached nuclear labels have shape {nuclei_labels.shape}, expected {dapi.shape}"
            )
        runtime_s = 0.0
        print(f"reused cached nuclear labels: {nuclei_labels_path}")
    else:
        dapi_model_input = dapi[::factor, ::factor]
        seg_cfg = CellposeConfig(
            pretrained_model=args.cellpose_model,
            diameter=None if args.diameter <= 0 else args.diameter,
            bsize=args.bsize,
            use_mps=(not args.cpu),
            normalize=bool(args.cellpose_normalize),
            min_size=int(args.min_size),
            cellprob_threshold=float(args.cellprob_threshold),
            flow_threshold=float(args.flow_threshold),
            channels=[0, 0] if args.cellpose_model == "nuclei" else None,
        )
        nuclei_small, runtime_s = run_cellpose(dapi_model_input, seg_cfg)
        nuclei_labels = upsample_labels_nearest(
            nuclei_small,
            target_shape=dapi.shape,
            factor=factor,
        ).astype(np.int32)
    nuclei_table, links_table = associate_nuclei(
        nuclei_labels,
        fiber_labels,
        min_overlap_fraction=args.min_overlap_fraction,
        boundary_distance_px=args.boundary_distance_px,
        central_normalized_radius=args.central_normalized_radius,
    )
    fiber_nuclei = summarize_fiber_nuclei(fiber_labels, nuclei_table)

    nuclei_path = args.output_dir / f"{stem}_nuclei.csv"
    links_path = args.output_dir / f"{stem}_nucleus_fiber_links.csv"
    fiber_nuclei_path = args.output_dir / f"{stem}_fiber_nuclei.csv"
    manifest_path = args.output_dir / f"{stem}_nuclear_run.json"
    save_labels(nuclei_labels_path, nuclei_labels)
    save_dataframe(nuclei_path, nuclei_table)
    save_dataframe(links_path, links_table)
    save_dataframe(fiber_nuclei_path, fiber_nuclei)
    manifest = {
        "schema_version": "nuclear_association.v1",
        "input": str(args.input),
        "fiber_labels": str(args.fiber_labels),
        "dapi_channel": int(args.dapi_channel),
        "dapi_preprocess": args.dapi_preprocess,
        "dapi_tile_size": int(args.dapi_tile_size),
        "dapi_background_quantile": float(args.dapi_background_quantile),
        "dapi_low_percentile": float(args.dapi_low_percentile),
        "dapi_high_percentile": float(args.dapi_high_percentile),
        "image_shape": list(image.shape),
        "crop": list(crop) if crop is not None else None,
        "downsample_factor": int(factor),
        "cellpose_model": args.cellpose_model,
        "diameter": None if args.diameter <= 0 else float(args.diameter),
        "min_size": int(args.min_size),
        "cellprob_threshold": float(args.cellprob_threshold),
        "flow_threshold": float(args.flow_threshold),
        "cellpose_channels": [0, 0] if args.cellpose_model == "nuclei" else None,
        "cellpose_normalize": bool(args.cellpose_normalize),
        "runtime_s": round(float(runtime_s), 2),
        "reused_nuclei_labels": reused_nuclei_labels,
        "association": {
            "min_overlap_fraction": float(args.min_overlap_fraction),
            "boundary_distance_px": float(args.boundary_distance_px),
            "central_normalized_radius": float(args.central_normalized_radius),
        },
        "outputs": {
            "nuclei_labels": str(nuclei_labels_path),
            "nuclei": str(nuclei_path),
            "links": str(links_path),
            "fiber_nuclei": str(fiber_nuclei_path),
        },
        "terminology": "fiber-associated nuclei; no automatic myonucleus call",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved nuclei labels: {nuclei_labels_path}")
    print(f"saved nuclei table: {nuclei_path}")
    print(f"saved nucleus-fiber links: {links_path}")
    print(f"saved fiber nuclear summary: {fiber_nuclei_path}")
    print(f"saved nuclear manifest: {manifest_path}")


if __name__ == "__main__":
    main()
