from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from fibertypeqc.artifacts import (
    build_run_manifest,
    can_reuse_fiber_labels,
    load_run_manifest,
    write_run_manifest,
)
from fibertypeqc.config import resolve_channel_config
from fibertypeqc.model_manifest import (
    load_model_manifest,
    validate_model_artifact,
    validate_model_compatibility,
)
from fibertypeqc.panels import Panel, validate_requested_domains
from fibertypeqc.qc_contract import (
    build_qc_report,
    postrun_checks,
    qc_check,
    write_qc_report,
)
from fibertypeqc.semantic_model import predict_semantic_candidate
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
    build_feature_diagnostics_table,
    class_stats_with_ci,
    qc_flags_from_fibers,
    quantify_labels,
)
from src.run_nuclear_stage import run_nuclear_analysis
from src.segment_cellpose import CellposeConfig, run_cellpose


@contextmanager
def stage(index: int, total: int, name: str):
    print(f"[{index}/{total}] {name} ...", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"[{index}/{total}] done: {name} ({time.perf_counter() - t0:.1f}s)", flush=True)


def _cleanup_outputs_for_retain_mode(
    *,
    retain_mode: str,
    labels_path: Path,
    fibers_path: Path,
    diagnostics_path: Path | None,
    summary_path: Path,
) -> list[Path]:
    if retain_mode not in {"full", "tables", "summary"}:
        raise ValueError(f"Unsupported retain mode: {retain_mode}")

    keep_paths = {summary_path}
    if retain_mode in {"full", "tables"}:
        keep_paths.add(fibers_path)
    if retain_mode == "full":
        keep_paths.add(labels_path)
        if diagnostics_path is not None:
            keep_paths.add(diagnostics_path)
    elif retain_mode == "tables":
        if diagnostics_path is not None:
            keep_paths.add(diagnostics_path)

    removed: list[Path] = []
    for path in (labels_path, fibers_path, diagnostics_path):
        if path is None or path in keep_paths:
            continue
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Membrane preprocess -> Cellpose -> quantify/classify")
    p.add_argument("--input", type=Path, required=True, help="Input CZI/TIFF")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    p.add_argument(
        "--labels-path",
        type=Path,
        default=None,
        help=(
            "Explicit corrected fiber-label TIFF to quantify; skips Cellpose and preserves "
            "source labels."
        ),
    )

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
        "--panel-config",
        type=Path,
        default=None,
        help="Preferred alias for --channel-config; accepts the canonical semantic panel schema.",
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
        "--emhc-channel",
        type=int,
        default=None,
        help="Optional eMHC marker channel index for separate regeneration diagnostics.",
    )
    p.add_argument(
        "--requested-domain",
        action="append",
        choices=["fiber_geometry", "fiber_identity", "regeneration", "nuclear_pathology"],
        default=[],
        help=(
            "Explicit output domain to validate before processing; may be supplied more than once."
        ),
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
    p.add_argument(
        "--model-manifest",
        type=Path,
        default=None,
        help=(
            "Optional JSON/YAML sidecar for --classifier-path. Validated before Cellpose; "
            "new model manifests must declare their required observed markers."
        ),
    )
    p.add_argument(
        "--export-diagnostics",
        action="store_true",
        help=(
            "Write an optional *_feature_diagnostics.csv file for model/feature debugging. "
            "Does not change the stable fibers CSV."
        ),
    )
    p.add_argument(
        "--retain-mode",
        type=str,
        default="full",
        choices=["full", "tables", "summary"],
        help=(
            "Control which per-image outputs are retained after a successful run. "
            "'full' keeps labels, fibers, diagnostics, and summary; "
            "'tables' removes heavy label TIFFs but keeps CSV tables; "
            "'summary' keeps only the summary CSV."
        ),
    )
    p.add_argument(
        "--reuse-artifacts",
        choices=["auto", "never", "required"],
        default="never",
        help="Reuse compatible cached fiber labels from the same output directory.",
    )

    p.add_argument("--bootstrap-reps", type=int, default=500)
    p.add_argument("--bootstrap-seed", type=int, default=0)

    p.add_argument("--qc-min-labels", type=int, default=300)
    p.add_argument("--qc-max-unknown-rate", type=float, default=0.35)
    p.add_argument("--qc-median-area-min", type=float, default=200.0)
    p.add_argument("--qc-median-area-max", type=float, default=15000.0)
    p.add_argument("--qc-max-type-corr", type=float, default=0.92)
    p.add_argument(
        "--nuclei-downsample-factor",
        type=int,
        default=2,
        help="Downsample factor for automatic DAPI nuclear segmentation.",
    )
    p.add_argument(
        "--nuclei-diameter",
        type=float,
        default=15.0,
        help="Approximate nucleus diameter in pixels for automatic DAPI segmentation.",
    )
    p.add_argument("--nuclei-min-size", type=int, default=30)
    p.add_argument(
        "--nuclei-cellprob-threshold",
        type=float,
        default=0.0,
        help="Cellpose cell-probability threshold for DAPI nuclear segmentation.",
    )
    p.add_argument(
        "--nuclei-flow-threshold",
        type=float,
        default=0.4,
        help="Cellpose flow-error threshold for DAPI nuclear segmentation.",
    )
    p.add_argument(
        "--dapi-preprocess",
        choices=["raw", "tile_subtract", "tile_normalize"],
        default="raw",
        help="Optional preprocessing before automatic DAPI segmentation.",
    )
    p.add_argument("--dapi-tile-size", type=int, default=512)
    p.add_argument("--dapi-background-quantile", type=float, default=0.02)
    p.add_argument("--dapi-low-percentile", type=float, default=1.0)
    p.add_argument("--dapi-high-percentile", type=float, default=99.8)
    return p


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    stem = args.input.stem.replace(" ", "_")
    preflight_qc_path = output_dir / f"{stem}_preflight_qc.json"
    preflight_checks: list[dict[str, object]] = []
    preflight_context: dict[str, object] = {"input": str(args.input)}

    def fail_preflight(code: str, error: Exception, next_action: str) -> None:
        preflight_checks.append(qc_check(code, "fail", str(error), next_action))
        write_qc_report(
            preflight_qc_path,
            build_qc_report(
                stage="preflight",
                checks=preflight_checks,
                context=preflight_context,
            ),
        )

    try:
        if args.labels_path is not None and args.reuse_artifacts != "never":
            raise ValueError("--labels-path cannot be combined with --reuse-artifacts.")
        if args.channel_config is not None and args.panel_config is not None:
            raise ValueError("Use only one of --panel-config and --channel-config.")
        if args.model_manifest is not None and args.classifier_path is None:
            raise ValueError("--model-manifest requires --classifier-path.")
    except ValueError as exc:
        fail_preflight("preflight.arguments_valid", exc, "correct_command_arguments")
        raise
    preflight_checks.append(
        qc_check(
            "preflight.arguments_valid",
            "pass",
            "Command arguments are internally compatible.",
            "proceed_to_channel_config",
        )
    )

    config_path = args.panel_config or args.channel_config
    try:
        channel_cfg, channel_warnings = resolve_channel_config(
            channel_config_path=config_path,
            i_channel=args.i_channel,
            iia_channel=args.iia_channel,
            iib_channel=args.iib_channel,
            iix_channel=args.iix_channel,
            emhc_channel=args.emhc_channel,
            dapi_channel=args.dapi_channel,
            type1_channel=args.type1_channel,
            type2_channel=args.type2_channel,
            membrane_channel=args.membrane_channel,
        )
    except (OSError, ValueError) as exc:
        fail_preflight("preflight.channel_config_valid", exc, "correct_channel_config")
        raise
    preflight_checks.append(
        qc_check(
            "preflight.channel_config_valid",
            "pass",
            "Channel configuration loaded successfully.",
            "proceed_to_model_validation",
        )
    )
    for warning in channel_warnings:
        print(f"Warning: {warning}", file=sys.stderr, flush=True)
        preflight_checks.append(
            qc_check(
                "preflight.channel_config_warning",
                "warn",
                warning,
                "confirm_channel_mapping",
            )
        )
    try:
        model_manifest = (
            load_model_manifest(args.model_manifest) if args.model_manifest is not None else None
        )
        if model_manifest is not None:
            validate_model_artifact(Path(args.classifier_path), model_manifest)
    except (OSError, ValueError) as exc:
        fail_preflight("preflight.model_artifact_valid", exc, "select_verified_model_artifact")
        raise
    preflight_checks.append(
        qc_check(
            "preflight.model_artifact_valid",
            "pass",
            "Selected model artifact and manifest are readable and compatible."
            if model_manifest is not None
            else (
                "Selected legacy model has no sidecar; panel compatibility will use the "
                "legacy adapter."
                if args.classifier_path
                else "No classifier was selected; the compatible rule path will be used."
            ),
            "proceed_to_input_validation",
        )
    )
    semantic_candidate = (
        model_manifest is not None
        and model_manifest.feature_schema_version == "multiplanel_features.v1"
    )

    run_nuclei = channel_cfg.dapi_channel is not None
    total_stages = 8 if run_nuclei else 7
    t_all = time.perf_counter()

    with stage(1, total_stages, "prepare output + load image"):
        try:
            image = load_multichannel_image(args.input)
            pixel_size_x_um, pixel_size_y_um = extract_pixel_size_um(args.input)
        except (ImportError, OSError, ValueError) as exc:
            fail_preflight("preflight.input_readable", exc, "select_readable_input_image")
            raise
        preflight_checks.append(
            qc_check(
                "preflight.input_readable",
                "pass",
                "Input image loaded successfully.",
                "proceed_to_panel_validation",
                metrics={"image_shape": list(image.shape)},
            )
        )
        n_channels = image.shape[0]

        panel = Panel.from_channel_config(channel_cfg)
        try:
            panel.validate(image_channel_count=n_channels)
        except ValueError as exc:
            fail_preflight("preflight.panel_compatible", exc, "correct_channel_mapping")
            raise
        preflight_checks.append(
            qc_check(
                "preflight.panel_compatible",
                "pass",
                "Panel channels are compatible with the input image.",
                "proceed_to_domain_validation",
            )
        )
        try:
            validate_requested_domains(panel, tuple(args.requested_domain))
        except ValueError as exc:
            fail_preflight(
                "preflight.requested_domains_supported",
                exc,
                "remove_or_correct_requested_domain",
            )
            raise
        preflight_checks.append(
            qc_check(
                "preflight.requested_domains_supported",
                "pass",
                "Requested output domains are supported by the selected panel.",
                "proceed_to_model_compatibility",
            )
        )
        try:
            validate_model_compatibility(
                panel,
                model_manifest,
                require_legacy_model=bool(args.classifier_path),
            )
        except ValueError as exc:
            fail_preflight(
                "preflight.model_panel_compatible",
                exc,
                "select_compatible_panel_or_model",
            )
            raise
        preflight_checks.append(
            qc_check(
                "preflight.model_panel_compatible",
                "pass",
                "Selected model is compatible with the observed panel.",
                "proceed_to_pixel_size_check",
            )
        )
        has_pixel_size = pixel_size_x_um is not None and pixel_size_y_um is not None
        preflight_checks.append(
            qc_check(
                "preflight.pixel_size_available",
                "pass" if has_pixel_size else "warn",
                "Physical pixel size is available."
                if has_pixel_size
                else "Physical pixel size is unavailable; pixel-area outputs remain usable.",
                "proceed_to_processing"
                if has_pixel_size
                else "confirm_pixel_size_before_area_interpretation",
                metrics={"x_um": pixel_size_x_um, "y_um": pixel_size_y_um},
            )
        )
        preflight_context.update(
            {
                "image_shape": list(image.shape),
                "panel_channels": dict(panel.channels),
                "requested_domains": list(args.requested_domain),
                "model_id": model_manifest.model_id if model_manifest is not None else None,
            }
        )
        write_qc_report(
            preflight_qc_path,
            build_qc_report(
                stage="preflight",
                checks=preflight_checks,
                context=preflight_context,
            ),
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

        run_manifest_path = output_dir / f"{stem}_run.json"
        labels_path = output_dir / f"{stem}_cellpose_labels.tif"
        seg_manifest = {
            "model": args.cellpose_model,
            "diameter": None if args.diameter <= 0 else args.diameter,
            "bsize": args.bsize,
            "resample": bool(args.resample),
            "requested_device": "cpu" if args.cpu else "mps_or_cpu",
            "normalize": bool(args.cellpose_normalize),
        }
        preprocessing_manifest = {
            "crop_auto": bool(args.crop_auto),
            "crop_ds": args.crop_ds,
            "crop_pad": args.crop_pad,
            "crop_min_size": args.crop_min_size,
            "downsample_factor": args.downsample_factor,
            "bg_sigma": args.bg_sigma,
            "smooth_sigma": args.smooth_sigma,
            "p_low": args.p_low,
            "p_high": args.p_high,
            "noise_floor": args.noise_floor,
        }
        run_manifest = build_run_manifest(
            input_path=args.input,
            image_shape=tuple(image.shape),
            pixel_size_um=(pixel_size_x_um, pixel_size_y_um),
            panel_fingerprint=panel.fingerprint,
            panel_channels=panel.channels,
            segmentation=seg_manifest,
            preprocessing=preprocessing_manifest,
            classifier_path=None if semantic_candidate else args.classifier_path,
            model_manifest_path=args.model_manifest,
        )
        reused_labels = False
        labels = None
        if args.labels_path is not None:
            labels = np.asarray(tifffile.imread(args.labels_path)).astype(np.int32)
            if labels.shape != image.shape[1:]:
                raise ValueError(
                    f"Provided labels have shape {labels.shape}, expected {image.shape[1:]}."
                )
            reused_labels = True
            print(f"using provided corrected fiber labels: {args.labels_path}")
        elif args.reuse_artifacts != "never":
            previous_manifest = load_run_manifest(run_manifest_path)
            if (
                previous_manifest
                and labels_path.exists()
                and can_reuse_fiber_labels(previous_manifest, run_manifest)
            ):
                labels = tifffile.imread(labels_path)
                if labels.shape != image.shape[1:]:
                    raise ValueError(
                        f"Cached labels have shape {labels.shape}, expected {image.shape[1:]}."
                    )
                reused_labels = True
            elif args.reuse_artifacts == "required":
                raise ValueError(
                    "Compatible cached fiber labels are required but were not found "
                    "in the output directory."
                )
        write_run_manifest(run_manifest_path, run_manifest)

    with stage(3, total_stages, "segment fibers with Cellpose"):
        seg_cfg = CellposeConfig(
            pretrained_model=args.cellpose_model,
            diameter=None if args.diameter <= 0 else args.diameter,
            bsize=args.bsize,
            resample=bool(args.resample),
            use_mps=(not args.cpu),
            normalize=bool(args.cellpose_normalize),
        )
        if reused_labels:
            labels_model, runtime_s = None, 0.0
            if args.labels_path is None:
                print("reused compatible cached fiber labels")
        else:
            labels_model, runtime_s = run_cellpose(prep.membrane_model_input, seg_cfg)

    with stage(4, total_stages, "restore full-resolution labels"):
        if not reused_labels:
            labels_crop = upsample_labels_nearest(
                labels_model,
                target_shape=prep.membrane_crop.shape,
                factor=prep_cfg.downsample_factor,
            )
            labels = paste_crop_labels(labels_crop, prep.membrane_full.shape, prep.crop_slices)
            save_labels(labels_path, labels)
        elif args.labels_path is not None:
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
            emhc_channel=channel_cfg.emhc_channel,
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
            # Semantic candidate bundles consume the panel-aware diagnostics below.
            # They are not legacy classifiers and therefore must not be passed into
            # quantify_labels' legacy prediction path.
            classifier_path=None if semantic_candidate else args.classifier_path,
            collect_spatial_marker_features=bool(args.export_diagnostics),
        )
        quant_cfg = apply_auto_profile(
            quant_cfg,
            sensitivity=float(args.sensitivity),
            mixed_strictness=float(args.mixed_strictness),
        )
        fibers = quantify_labels(labels, image, quant_cfg)
        fibers_path = output_dir / f"{stem}_fibers.csv"
        save_dataframe(fibers_path, fibers)
        diagnostics_path = None
        semantic_predictions_path = None
        if args.export_diagnostics or semantic_candidate:
            diagnostics = build_feature_diagnostics_table(fibers, quant_cfg)
            if args.export_diagnostics:
                diagnostics_path = output_dir / f"{stem}_feature_diagnostics.csv"
                save_dataframe(diagnostics_path, diagnostics)
            if semantic_candidate:
                semantic_predictions_path = output_dir / f"{stem}_model_predictions.csv"
                predictions = predict_semantic_candidate(
                    diagnostics, args.classifier_path, model_manifest
                )
                save_dataframe(semantic_predictions_path, predictions)

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
        postrun_qc_path = output_dir / f"{stem}_postrun_qc.json"
        postrun_qc_stats = {**qc_stats, "n_labels": len(fibers)}
        postrun_report = build_qc_report(
            stage="postrun",
            checks=postrun_checks(
                postrun_qc_stats,
                min_labels=qc_cfg.min_labels,
                max_unknown_rate=qc_cfg.max_unknown_rate,
                median_area_min=qc_cfg.median_area_min,
                median_area_max=qc_cfg.median_area_max,
                max_type_corr=qc_cfg.max_type_corr,
            ),
            context={
                "input": str(args.input),
                "fibers_path": str(fibers_path),
                "review_required": bool(fibers.get("needs_review", pd.Series(dtype=bool)).any()),
            },
        )
        write_qc_report(postrun_qc_path, postrun_report)

        summary = {
            "input": str(args.input),
            "labels_path": str(labels_path),
            "fibers_path": str(fibers_path),
            "feature_diagnostics_path": (
                str(diagnostics_path) if diagnostics_path is not None else ""
            ),
            "semantic_predictions_path": (
                str(semantic_predictions_path) if semantic_predictions_path is not None else ""
            ),
            "run_manifest_path": str(run_manifest_path),
            "preflight_qc_path": str(preflight_qc_path),
            "postrun_qc_path": str(postrun_qc_path),
            "runtime_s": round(float(runtime_s), 2),
            "membrane_channel": int(channel_cfg.membrane_channel),
            "dapi_channel": channel_cfg.dapi_channel,
            "i_channel": channel_cfg.i_channel,
            "iia_channel": channel_cfg.iia_channel,
            "iib_channel": channel_cfg.iib_channel,
            "iix_channel": channel_cfg.iix_channel,
            "type1_channel": channel_cfg.type1_channel,
            "type2_channel": channel_cfg.type2_channel,
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

    nuclear_outputs: dict[str, Path] = {}
    if run_nuclei:
        with stage(8, total_stages, "segment DAPI nuclei + associate with fibers"):
            nuclear_outputs = run_nuclear_analysis(
                image=image,
                input_path=args.input,
                fiber_labels=np.asarray(labels).astype(np.int32),
                fiber_labels_path=labels_path,
                output_dir=output_dir / "nuclear",
                dapi_channel=int(channel_cfg.dapi_channel),
                dapi_preprocess=args.dapi_preprocess,
                dapi_tile_size=args.dapi_tile_size,
                dapi_background_quantile=args.dapi_background_quantile,
                dapi_low_percentile=args.dapi_low_percentile,
                dapi_high_percentile=args.dapi_high_percentile,
                downsample_factor=args.nuclei_downsample_factor,
                diameter=args.nuclei_diameter,
                min_size=args.nuclei_min_size,
                cellprob_threshold=args.nuclei_cellprob_threshold,
                flow_threshold=args.nuclei_flow_threshold,
                cpu=args.cpu,
                cellpose_normalize=True,
                reuse_artifacts=args.reuse_artifacts != "never",
            )
            summary["nuclear_output_dir"] = str(output_dir / "nuclear")
            summary["nuclear_manifest_path"] = str(nuclear_outputs["manifest"])
            save_dataframe(summary_path, pd.DataFrame([summary]))

    removed_outputs = _cleanup_outputs_for_retain_mode(
        retain_mode=args.retain_mode,
        labels_path=labels_path,
        fibers_path=fibers_path,
        diagnostics_path=diagnostics_path,
        summary_path=summary_path,
    )

    if labels_path.exists():
        print("saved labels:", labels_path)
    if fibers_path.exists():
        print("saved fibers:", fibers_path)
    if diagnostics_path is not None and diagnostics_path.exists():
        print("saved diagnostics:", diagnostics_path)
    if nuclear_outputs:
        print("saved nuclear outputs:", output_dir / "nuclear")
    print("saved summary:", summary_path)
    print("saved preflight QC:", preflight_qc_path)
    print("saved post-run QC:", postrun_qc_path)
    if removed_outputs:
        print(
            "removed retained-mode outputs:",
            ", ".join(str(path.name) for path in removed_outputs),
        )
    print(f"total runtime: {time.perf_counter() - t_all:.1f}s")
    print("summary:", summary)


if __name__ == "__main__":
    main()
