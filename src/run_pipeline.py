from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from fibertypeqc.config import resolve_channel_config
from src.io_utils import (
    ensure_dir,
    extract_pixel_size_um,
    label_summary,
    load_multichannel_image,
    save_dataframe,
    save_labels,
)
from src.preprocess_membrane import (
    PreprocessConfig,
    paste_crop_labels,
    preprocess_membrane_channel,
    upsample_labels_nearest,
)
from src.quantify_classify import (
    QCConfig,
    QuantifyConfig,
    apply_auto_profile,
    class_stats_with_ci,
    qc_flags_from_fibers,
    quantify_labels,
)
from src.segment_cellpose import CellposeConfig, run_cellpose


@contextmanager
def stage(index: int, total: int, name: str):
    print(f"[{index}/{total}] {name} ...", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"[{index}/{total}] done: {name} ({time.perf_counter() - t0:.1f}s)", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Membrane preprocess -> Cellpose -> quantify/classify")
    p.add_argument("--input", type=Path, required=True, help="Input CZI/TIFF")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory")

    p.add_argument(
        "--channel-config",
        type=Path,
        default=None,
        help=(
            "YAML file with panel-aware channel mapping under 'channels' "
            "and optional 'classification'."
        ),
    )
    p.add_argument(
        "--membrane-channel",
        type=int,
        default=None,
        help="Membrane/laminin channel index.",
    )
    p.add_argument(
        "--dapi-channel",
        type=int,
        default=None,
        help="Optional DAPI channel index.",
    )
    p.add_argument(
        "--i-channel",
        type=int,
        default=None,
        help="Optional type I marker channel index.",
    )
    p.add_argument(
        "--iia-channel",
        type=int,
        default=None,
        help="Optional IIa marker channel index.",
    )
    p.add_argument(
        "--iib-channel",
        type=int,
        default=None,
        help="Optional IIb marker channel index.",
    )
    p.add_argument(
        "--iix-channel",
        type=int,
        default=None,
        help="Optional IIx marker channel index.",
    )
    p.add_argument(
        "--type1-channel",
        type=int,
        default=None,
        help="Legacy alias for --iib-channel.",
    )
    p.add_argument(
        "--type2-channel",
        type=int,
        default=None,
        help="Legacy alias for --iia-channel.",
    )

    p.add_argument("--crop-auto", action="store_true", default=True)
    p.add_argument("--crop-ds", type=int, default=8)
    p.add_argument("--crop-pad", type=int, default=128)
    p.add_argument("--crop-min-size", type=int, default=2000)

    p.add_argument("--downsample-factor", type=int, default=2)
    p.add_argument("--bg-sigma", type=float, default=30.0)
    p.add_argument("--smooth-sigma", type=float, default=1.0)
    p.add_argument("--p-low", type=float, default=1.0)
    p.add_argument("--p-high", type=float, default=99.8)
    p.add_argument("--noise-floor", type=float, default=0.05)

    p.add_argument("--cellpose-model", type=str, default="cpsam")
    p.add_argument("--diameter", type=float, default=30.0)
    p.add_argument("--bsize", type=int, default=256)
    p.add_argument("--resample", action="store_true", default=False)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if MPS is available")
    p.add_argument(
        "--cellpose-normalize",
        action="store_true",
        help="Enable Cellpose normalization",
    )

    p.add_argument(
        "--threshold-mode",
        type=str,
        default="quantile",
        choices=["quantile", "otsu", "yen", "fixed"],
    )
    p.add_argument("--sensitivity", type=float, default=0.5, help="0..1 auto profile control")
    p.add_argument(
        "--mixed-strictness",
        type=float,
        default=0.7,
        help="0..1; higher => fewer mixed calls",
    )
    p.add_argument("--quantile", type=float, default=0.6)
    p.add_argument("--percentile-q", type=float, default=0.85)
    p.add_argument(
        "--no-percentile-gate",
        action="store_true",
        help="Disable P-quantile assist gate for typing",
    )
    p.add_argument("--type1-threshold", type=float, default=0.0)
    p.add_argument("--type2-threshold", type=float, default=0.0)
    p.add_argument(
        "--iib-threshold",
        type=float,
        default=None,
        help="Preferred alias for --type1-threshold.",
    )
    p.add_argument(
        "--iia-threshold",
        type=float,
        default=None,
        help="Preferred alias for --type2-threshold.",
    )
    p.add_argument(
        "--typing-preprocess",
        type=str,
        default="global_subtract",
        choices=["raw", "global_subtract", "tile_subtract", "gaussian_subtract"],
        help="Type-channel preprocessing. Default avoids Gaussian high-pass subtraction.",
    )
    p.add_argument(
        "--typing-bg-quantile",
        type=float,
        default=0.02,
        help="Low quantile subtracted for global_subtract or tile_subtract.",
    )
    p.add_argument("--typing-tile-size", type=int, default=512)
    p.add_argument("--typing-bg-sigma", type=float, default=24.0)
    p.add_argument("--typing-smooth-sigma", type=float, default=0.8)
    p.add_argument("--typing-erode-px", type=int, default=2)
    p.add_argument("--coverage-quantile", type=float, default=0.85)
    p.add_argument("--min-coverage", type=float, default=0.06)
    p.add_argument("--review-confidence-threshold", type=float, default=0.15)
    p.add_argument("--review-margin", type=float, default=0.05)
    p.add_argument(
        "--model-confidence-threshold",
        type=float,
        default=0.70,
        help="Flag model calls below this probability for review.",
    )
    p.add_argument(
        "--model-margin-threshold",
        type=float,
        default=0.25,
        help="Flag model calls when top probability minus runner-up is this small.",
    )
    p.add_argument(
        "--classifier-path",
        type=str,
        default=None,
        help="Optional sklearn model (.joblib/.pkl)",
    )

    p.add_argument("--bootstrap-reps", type=int, default=500)
    p.add_argument("--bootstrap-seed", type=int, default=0)

    p.add_argument("--qc-min-labels", type=int, default=300)
    p.add_argument("--qc-max-unknown-rate", type=float, default=0.35)
    p.add_argument("--qc-median-area-min", type=float, default=200.0)
    p.add_argument("--qc-median-area-max", type=float, default=15000.0)
    p.add_argument("--qc-max-type-corr", type=float, default=0.92)
    return p


def main() -> None:
    args = build_parser().parse_args()
    channel_cfg, channel_warnings = resolve_channel_config(
        channel_config_path=args.channel_config,
        i_channel=args.i_channel,
        iia_channel=args.iia_channel,
        iib_channel=args.iib_channel,
        iix_channel=args.iix_channel,
        dapi_channel=args.dapi_channel,
        type1_channel=args.type1_channel,
        type2_channel=args.type2_channel,
        membrane_channel=args.membrane_channel,
    )
    for warning in channel_warnings:
        print(f"Warning: {warning}", file=sys.stderr, flush=True)

    total_stages = 7
    t_all = time.perf_counter()

    with stage(1, total_stages, "prepare output + load image"):
        output_dir = ensure_dir(args.output_dir)
        image = load_multichannel_image(args.input)
        pixel_size_x_um, pixel_size_y_um = extract_pixel_size_um(args.input)
        n_channels = image.shape[0]

        channel_values = {
            "membrane": channel_cfg.membrane_channel,
            "dapi": channel_cfg.dapi_channel,
            "i": channel_cfg.i_channel,
            "iia": channel_cfg.iia_channel,
            "iib": channel_cfg.iib_channel,
            "iix": channel_cfg.iix_channel,
        }
        for name, index in channel_values.items():
            if index is None:
                continue
            if index < 0 or index >= n_channels:
                raise ValueError(
                    f"Invalid {name} channel {index} for image with {n_channels} channels"
                )
        if channel_cfg.iib_channel is None or channel_cfg.iia_channel is None:
            raise ValueError(
                "The current alpha typing path still requires both IIb and IIa marker channels. "
                "Panel-aware config is accepted, but non-IIa/IIb typing modes are not active yet."
            )

    with stage(2, total_stages, "preprocess membrane channel"):
        membrane = image[channel_cfg.membrane_channel]
        prep_cfg = PreprocessConfig(
            crop_auto=bool(args.crop_auto),
            crop_ds=args.crop_ds,
            crop_pad=args.crop_pad,
            crop_min_size=args.crop_min_size,
            downsample_factor=args.downsample_factor,
            bg_sigma=args.bg_sigma,
            smooth_sigma=args.smooth_sigma,
            p_low=args.p_low,
            p_high=args.p_high,
            noise_floor=args.noise_floor,
        )
        prep = preprocess_membrane_channel(membrane, prep_cfg)

    with stage(3, total_stages, "segment fibers with Cellpose"):
        seg_cfg = CellposeConfig(
            pretrained_model=args.cellpose_model,
            diameter=None if args.diameter <= 0 else args.diameter,
            bsize=args.bsize,
            resample=bool(args.resample),
            use_mps=(not args.cpu),
            normalize=bool(args.cellpose_normalize),
        )
        labels_model, runtime_s = run_cellpose(prep.membrane_model_input, seg_cfg)

    with stage(4, total_stages, "restore full-resolution labels"):
        labels_crop = upsample_labels_nearest(
            labels_model,
            target_shape=prep.membrane_crop.shape,
            factor=prep_cfg.downsample_factor,
        )
        labels = paste_crop_labels(labels_crop, prep.membrane_full.shape, prep.crop_slices)

        stem = args.input.stem.replace(" ", "_")
        labels_path = output_dir / f"{stem}_cellpose_labels.tif"
        save_labels(labels_path, labels)

    with stage(5, total_stages, "extract fiber features + classify types"):
        iib_threshold = args.iib_threshold
        iia_threshold = args.iia_threshold
        if args.type1_threshold != 0.0 and iib_threshold is None:
            iib_threshold = args.type1_threshold
            print(
                "Warning: --type1-threshold is a legacy alias for --iib-threshold",
                file=sys.stderr,
                flush=True,
            )
        if args.type2_threshold != 0.0 and iia_threshold is None:
            iia_threshold = args.type2_threshold
            print(
                "Warning: --type2-threshold is a legacy alias for --iia-threshold",
                file=sys.stderr,
                flush=True,
            )
        if iib_threshold is None:
            iib_threshold = 0.0
        if iia_threshold is None:
            iia_threshold = 0.0
        quant_cfg = QuantifyConfig(
            type1_channel=channel_cfg.type1_channel,
            type2_channel=channel_cfg.type2_channel,
            i_channel=channel_cfg.i_channel,
            iix_channel=channel_cfg.iix_channel,
            threshold_mode=args.threshold_mode,
            quantile=args.quantile,
            percentile_q=args.percentile_q,
            use_percentile_gate=(not args.no_percentile_gate),
            type1_threshold=iib_threshold,
            type2_threshold=iia_threshold,
            typing_preprocess=args.typing_preprocess,
            typing_bg_quantile=args.typing_bg_quantile,
            typing_tile_size=args.typing_tile_size,
            typing_bg_sigma=args.typing_bg_sigma,
            typing_smooth_sigma=args.typing_smooth_sigma,
            typing_erode_px=args.typing_erode_px,
            coverage_quantile=args.coverage_quantile,
            min_coverage=args.min_coverage,
            review_confidence_threshold=args.review_confidence_threshold,
            review_margin=args.review_margin,
            model_confidence_threshold=args.model_confidence_threshold,
            model_margin_threshold=args.model_margin_threshold,
            pixel_size_x_um=pixel_size_x_um,
            pixel_size_y_um=pixel_size_y_um,
            classifier_path=args.classifier_path,
        )
        quant_cfg = apply_auto_profile(
            quant_cfg,
            sensitivity=float(args.sensitivity),
            mixed_strictness=float(args.mixed_strictness),
        )
        fibers = quantify_labels(labels, image, quant_cfg)
        fibers_path = output_dir / f"{stem}_fibers.csv"
        save_dataframe(fibers_path, fibers)

    with stage(6, total_stages, "compute summary + QC"):
        qc_cfg = QCConfig(
            min_labels=args.qc_min_labels,
            max_unknown_rate=args.qc_max_unknown_rate,
            median_area_min=args.qc_median_area_min,
            median_area_max=args.qc_median_area_max,
            max_type_corr=args.qc_max_type_corr,
        )
        class_stats = class_stats_with_ci(
            fibers,
            bootstrap_reps=args.bootstrap_reps,
            seed=args.bootstrap_seed,
        )
        qc_stats = qc_flags_from_fibers(fibers, qc_cfg)

        summary = {
            "input": str(args.input),
            "labels_path": str(labels_path),
            "fibers_path": str(fibers_path),
            "runtime_s": round(float(runtime_s), 2),
            "membrane_channel": int(channel_cfg.membrane_channel),
            "dapi_channel": channel_cfg.dapi_channel,
            "i_channel": channel_cfg.i_channel,
            "iia_channel": channel_cfg.iia_channel,
            "iib_channel": channel_cfg.iib_channel,
            "iix_channel": channel_cfg.iix_channel,
            "type1_channel": int(channel_cfg.type1_channel),
            "type2_channel": int(channel_cfg.type2_channel),
            "crop_slices": str(prep.crop_slices),
            "downsample_factor": int(args.downsample_factor),
            "cellpose_model": args.cellpose_model,
            "cellpose_normalize": bool(args.cellpose_normalize),
            "sensitivity": float(args.sensitivity),
            "mixed_strictness": float(args.mixed_strictness),
            "typing_quantile_used": float(quant_cfg.quantile),
            "typing_min_coverage_used": float(quant_cfg.min_coverage),
            "typing_mixed_tolerance_used": float(quant_cfg.mixed_balance_tolerance),
            "typing_erode_px": int(quant_cfg.typing_erode_px),
            "typing_preprocess": str(quant_cfg.typing_preprocess),
            "typing_bg_quantile": float(quant_cfg.typing_bg_quantile),
            "typing_tile_size": int(quant_cfg.typing_tile_size),
            "typing_bg_sigma": float(quant_cfg.typing_bg_sigma),
            "typing_smooth_sigma": float(quant_cfg.typing_smooth_sigma),
            "model_confidence_threshold": float(quant_cfg.model_confidence_threshold),
            "model_margin_threshold": float(quant_cfg.model_margin_threshold),
            "pixel_size_x_um": pixel_size_x_um,
            "pixel_size_y_um": pixel_size_y_um,
        }
        summary.update(label_summary(labels))
        if "area_um2" in fibers.columns and len(fibers) > 0:
            summary.update(
                {
                    "area_um2_median": float(fibers["area_um2"].median()),
                    "area_um2_mean": float(fibers["area_um2"].mean()),
                    "area_um2_min": float(fibers["area_um2"].min()),
                    "area_um2_max": float(fibers["area_um2"].max()),
                }
            )
            for erode_px in quant_cfg.csa_erode_px:
                col = f"area_erode_{int(erode_px)}px_um2"
                if col in fibers.columns:
                    summary[f"{col}_median"] = float(fibers[col].median())
                    summary[f"{col}_mean"] = float(fibers[col].mean())
        summary.update(class_stats)
        summary.update(qc_stats)

    with stage(7, total_stages, "save summary"):
        summary_df = pd.DataFrame([summary])
        summary_path = output_dir / f"{stem}_summary.csv"
        save_dataframe(summary_path, summary_df)

    print("saved labels:", labels_path)
    print("saved fibers:", fibers_path)
    print("saved summary:", summary_path)
    print(f"total runtime: {time.perf_counter() - t_all:.1f}s")
    print("summary:", summary)


if __name__ == "__main__":
    main()
