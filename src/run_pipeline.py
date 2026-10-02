from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from fibertypeqc.artifacts import (
    build_run_manifest,
    can_reuse_fiber_labels,
    file_sha256,
    load_run_manifest,
    portable_path,
    write_run_manifest,
)
from fibertypeqc.config import resolve_channel_config
from fibertypeqc.html_report import generate_result_report
from fibertypeqc.model_manifest import (
    load_model_manifest,
    validate_model_artifact,
    validate_model_compatibility,
)
from fibertypeqc.model_resolution import resolve_model_argument
from fibertypeqc.panels import Panel, validate_requested_domains
from fibertypeqc.qc_contract import (
    build_qc_report,
    postrun_checks,
    qc_check,
    write_qc_report,
)
from fibertypeqc.result_bundle import build_result_bundle, write_result_bundle
from fibertypeqc.semantic_model import apply_semantic_predictions, predict_semantic_candidate
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
    summary_classes,
)
from src.run_nuclear_stage import run_nuclear_analysis
from src.segment_cellpose import CellposeConfig, resolve_device, run_cellpose


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
        "--image-id",
        type=str,
        default=None,
        help=(
            "Identifier used to name outputs (default: input file stem). Spaces become "
            "underscores; path separators are not allowed."
        ),
    )
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

    crop_group = p.add_mutually_exclusive_group()
    crop_group.add_argument(
        "--crop-auto",
        dest="crop_auto",
        action="store_true",
        default=True,
        help="Automatically crop the membrane image to the detected tissue field (default).",
    )
    crop_group.add_argument(
        "--no-crop-auto",
        dest="crop_auto",
        action="store_false",
        help="Segment the complete membrane image without automatic tissue cropping.",
    )
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
        "--model",
        type=str,
        default=None,
        help=(
            "Registered model ID (see manifests/model_registry.v1.yaml) or a path to a registered "
            "model file (identified by digest). With an ID, private artifacts are read from "
            "$FIBERTYPEQC_MODEL_ROOT. Cannot be combined with --classifier-path or "
            "--model-manifest."
        ),
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
    p.add_argument(
        "--qc-max-uncertainty-rate",
        "--qc-max-unknown-rate",
        dest="qc_max_uncertainty_rate",
        type=float,
        default=0.35,
        help=(
            "Warn when more than this fraction of fibers is low-confidence or unresolved. "
            "--qc-max-unknown-rate is a deprecated alias."
        ),
    )
    p.add_argument(
        "--qc-max-residual-rate",
        type=float,
        default=None,
        help=(
            "Warn when more than this fraction of fibers is assigned the panel's residual "
            "(inferred-by-absence) class. Off by default until calibrated for the panel."
        ),
    )
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


# CLI flags whose values apply_auto_profile() derives from --sensitivity/--mixed-strictness.
# Explicit values for these flags are currently overridden; see auto_profile_overridden_flags().
AUTO_PROFILE_FLAG_FIELDS = {
    "--quantile": "quantile",
    "--no-percentile-gate": "use_percentile_gate",
    "--typing-bg-sigma": "typing_bg_sigma",
    "--typing-smooth-sigma": "typing_smooth_sigma",
    "--coverage-quantile": "coverage_quantile",
    "--min-coverage": "min_coverage",
    "--review-confidence-threshold": "review_confidence_threshold",
    "--review-margin": "review_margin",
}


# CLI flags that change marker features. A model manifest that pins feature extraction owns all
# of these; passing any of them with such a model is refused.
FEATURE_EXTRACTION_FLAGS = (
    "--threshold-mode",
    "--quantile",
    "--percentile-q",
    "--no-percentile-gate",
    "--typing-preprocess",
    "--typing-bg-quantile",
    "--typing-tile-size",
    "--typing-bg-sigma",
    "--typing-smooth-sigma",
    "--typing-erode-px",
    "--coverage-quantile",
    "--min-coverage",
    "--sensitivity",
    "--mixed-strictness",
)


def passed_flags(argv: list[str], flags: tuple[str, ...]) -> list[str]:
    passed = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
    return [flag for flag in flags if flag in passed]


def auto_profile_overridden_flags(argv: list[str]) -> list[str]:
    """Return explicitly passed flags whose values the auto profile replaces."""
    passed = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
    return [flag for flag in AUTO_PROFILE_FLAG_FIELDS if flag in passed]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.model is not None:
        if args.classifier_path is not None or args.model_manifest is not None:
            parser.error("--model cannot be combined with --classifier-path or --model-manifest")
        try:
            resolved = resolve_model_argument(args.model)
        except ValueError as exc:
            parser.error(str(exc))
        args.classifier_path = str(resolved.artifact_path)
        args.model_manifest = resolved.manifest_path
    output_dir = ensure_dir(args.output_dir)
    image_id = args.image_id if args.image_id is not None else args.input.stem
    if not image_id.strip() or "/" in image_id or "\\" in image_id or image_id in {".", ".."}:
        raise SystemExit(f"Invalid --image-id: {image_id!r}")
    stem = image_id.strip().replace(" ", "_")
    preflight_qc_path = output_dir / f"{stem}_preflight_qc.json"
    preflight_checks: list[dict[str, object]] = []
    preflight_context: dict[str, object] = {"input": portable_path(args.input)}

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
    overridden_flags = auto_profile_overridden_flags(sys.argv[1:])
    # With a model that pins feature extraction, the sensitivity profile is not used, so a
    # separate check below refuses conflicting flags instead of warning about the profile.
    pins_features = False
    if args.model_manifest is not None:
        try:
            pins_features = load_model_manifest(args.model_manifest).feature_extraction is not None
        except ValueError:
            pins_features = False  # reported by the manifest check below
    if overridden_flags and not pins_features:
        profile = apply_auto_profile(
            QuantifyConfig(),
            sensitivity=float(args.sensitivity),
            mixed_strictness=float(args.mixed_strictness),
        )
        effective = ", ".join(
            f"{AUTO_PROFILE_FLAG_FIELDS[flag]}={getattr(profile, AUTO_PROFILE_FLAG_FIELDS[flag])!r}"
            for flag in overridden_flags
        )
        warning = (
            f"{', '.join(overridden_flags)} were set explicitly but are derived from "
            f"--sensitivity={args.sensitivity} and --mixed-strictness={args.mixed_strictness}; "
            f"the explicit values are ignored (effective: {effective})."
        )
        print(f"Warning: {warning}", file=sys.stderr, flush=True)
        preflight_checks.append(
            qc_check(
                "preflight.typing_flags_overridden",
                "warn",
                warning,
                "adjust_sensitivity_or_omit_overridden_flags",
                metrics={"overridden_flags": overridden_flags},
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
    if model_manifest is not None and model_manifest.feature_extraction is not None:
        conflicting = passed_flags(sys.argv[1:], FEATURE_EXTRACTION_FLAGS)
        if conflicting:
            error = ValueError(
                f"Model '{model_manifest.model_id}' pins its feature-extraction settings; remove "
                f"{', '.join(conflicting)}."
            )
            fail_preflight("preflight.model_feature_settings", error, "remove_conflicting_flags")
            raise error
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
        if args.labels_path is not None:
            labels_source = f"provided:{file_sha256(args.labels_path)}"
            segmentation_device = "not_used"
        else:
            labels_source = "cellpose"
            segmentation_device = resolve_device(not args.cpu)
        seg_manifest = {
            "model": args.cellpose_model,
            "diameter": None if args.diameter <= 0 else args.diameter,
            "bsize": args.bsize,
            "resample": bool(args.resample),
            "device": segmentation_device,
            "normalize": bool(args.cellpose_normalize),
            "labels_source": labels_source,
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
        # Semantic and legacy classifiers are both identified by path and digest in the run record.
        manifest_classifier = args.classifier_path
        run_manifest = build_run_manifest(
            input_path=args.input,
            input_sha256=file_sha256(args.input),
            image_shape=tuple(image.shape),
            pixel_size_um=(pixel_size_x_um, pixel_size_y_um),
            panel_fingerprint=panel.fingerprint,
            panel_channels=panel.channels,
            segmentation=seg_manifest,
            preprocessing=preprocessing_manifest,
            classifier_path=manifest_classifier,
            model_manifest_path=args.model_manifest,
            classifier_sha256=(
                file_sha256(Path(manifest_classifier)) if manifest_classifier else None
            ),
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
        if model_manifest is not None and model_manifest.feature_extraction is not None:
            # Reproduce the model's training features exactly: pinned values, no profile.
            quant_cfg = replace(quant_cfg, **model_manifest.feature_extraction)
        else:
            quant_cfg = apply_auto_profile(
                quant_cfg,
                sensitivity=float(args.sensitivity),
                mixed_strictness=float(args.mixed_strictness),
            )
        fibers = quantify_labels(labels, image, quant_cfg)
        fibers_path = output_dir / f"{stem}_fibers.csv"
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
                if model_manifest.task == "fiber_identity":
                    fibers = apply_semantic_predictions(
                        fibers,
                        predictions,
                        confidence_threshold=quant_cfg.model_confidence_threshold,
                        margin_threshold=quant_cfg.model_margin_threshold,
                        classifier_path=portable_path(args.classifier_path),
                    )
        save_dataframe(fibers_path, fibers)

    with stage(6, total_stages, "compute summary + QC"):
        qc_cfg = QCConfig(
            min_labels=args.qc_min_labels,
            max_uncertainty_rate=args.qc_max_uncertainty_rate,
            median_area_min=args.qc_median_area_min,
            median_area_max=args.qc_median_area_max,
            max_type_corr=args.qc_max_type_corr,
            max_residual_rate=args.qc_max_residual_rate,
        )
        classes, canonicalize_labels = summary_classes(fibers)
        class_stats = class_stats_with_ci(
            fibers,
            classes=classes,
            bootstrap_reps=args.bootstrap_reps,
            seed=args.bootstrap_seed,
            canonicalize_labels=canonicalize_labels,
        )
        residual_target_class = (
            channel_cfg.residual_target_class if channel_cfg.residual_inference_enabled else None
        )
        qc_stats = qc_flags_from_fibers(
            fibers, qc_cfg, residual_target_class=residual_target_class
        )
        postrun_qc_path = output_dir / f"{stem}_postrun_qc.json"
        postrun_qc_stats = {**qc_stats, "n_labels": len(fibers)}
        postrun_report = build_qc_report(
            stage="postrun",
            checks=postrun_checks(
                postrun_qc_stats,
                min_labels=qc_cfg.min_labels,
                max_uncertainty_rate=qc_cfg.max_uncertainty_rate,
                median_area_min=qc_cfg.median_area_min,
                median_area_max=qc_cfg.median_area_max,
                max_type_corr=qc_cfg.max_type_corr,
                max_residual_rate=qc_cfg.max_residual_rate,
            ),
            context={
                "input": portable_path(args.input),
                "fibers_path": fibers_path.name,
                "review_required": bool(fibers.get("needs_review", pd.Series(dtype=bool)).any()),
            },
        )
        write_qc_report(postrun_qc_path, postrun_report)

        summary = {
            # Paths are portable: the input as given (or its file name) and output files
            # relative to this image's output directory.
            "input": portable_path(args.input),
            "input_sha256": run_manifest["source_image_sha256"],
            "labels_path": labels_path.name,
            "fibers_path": fibers_path.name,
            "feature_diagnostics_path": (
                diagnostics_path.name if diagnostics_path is not None else ""
            ),
            "semantic_predictions_path": (
                semantic_predictions_path.name if semantic_predictions_path is not None else ""
            ),
            "run_manifest_path": run_manifest_path.name,
            "preflight_qc_path": preflight_qc_path.name,
            "postrun_qc_path": postrun_qc_path.name,
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

    result_bundle_path = output_dir / f"{stem}_result_bundle.json"
    result_report_path = output_dir / f"{stem}_result_report.html"
    artifact_paths = {
        "fiber_labels": labels_path,
        "fiber_table": fibers_path,
        "feature_diagnostics": diagnostics_path,
        "fiber_identity_predictions": semantic_predictions_path,
        "image_summary": summary_path,
        "preflight_qc": preflight_qc_path,
        "postrun_qc": postrun_qc_path,
        "run_provenance": run_manifest_path,
        "nuclei_labels": nuclear_outputs.get("nuclei_labels"),
        "nuclei_table": nuclear_outputs.get("nuclei"),
        "nucleus_fiber_associations": nuclear_outputs.get("links"),
        "fiber_nuclei_summary": nuclear_outputs.get("fiber_nuclei"),
        "nuclear_provenance": nuclear_outputs.get("manifest"),
        "html_report": result_report_path,
    }
    bundle_kwargs = {
        "output_dir": output_dir,
        "image_id": stem,
        "retain_mode": args.retain_mode,
        "artifact_paths": artifact_paths,
        "additional_domains": (
            {"feature_diagnostics": ["regeneration"]}
            if channel_cfg.emhc_channel is not None
            else None
        ),
    }
    result_bundle = build_result_bundle(**bundle_kwargs)
    write_result_bundle(result_bundle_path, result_bundle)
    generate_result_report(result_bundle_path, result_report_path)
    result_bundle = build_result_bundle(**bundle_kwargs)
    write_result_bundle(result_bundle_path, result_bundle)
    generate_result_report(result_bundle_path, result_report_path)

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
    print("saved result bundle:", result_bundle_path)
    print("saved result report:", result_report_path)
    if removed_outputs:
        print(
            "removed retained-mode outputs:",
            ", ".join(str(path.name) for path in removed_outputs),
        )
    print(f"total runtime: {time.perf_counter() - t_all:.1f}s")
    print("summary:", summary)


if __name__ == "__main__":
    main()
