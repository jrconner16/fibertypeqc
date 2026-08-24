"""
Batch runner for fibertyping v0.

Applies the frozen v0 pipeline to a folder of .czi files and collects results
into batch_summary.csv. Failures are logged and collected but don't crash the batch.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fibertypeqc.czi_scenes import discover_czi_scenes, export_czi_scenes

# V0 frozen parameters (these define the baseline production command)
V0_PARAMS = {
    "type1_channel": 0,
    "type2_channel": 1,
    "membrane_channel": 2,
    "typing_preprocess": "tile_subtract",
    "typing_tile_size": 256,
    "typing_erode_px": 2,
    "classifier_path": "data/models/rebaseline_tile_v2_p75p90_iib_iia_iix.joblib",
    "model_confidence_threshold": 0.55,
    "model_margin_threshold": 0.15,
    "downsample_factor": 2,
}

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BatchChannelOverrides:
    channel_config: Path | None = None
    membrane_channel: int | None = None
    dapi_channel: int | None = None
    i_channel: int | None = None
    iia_channel: int | None = None
    iib_channel: int | None = None
    iix_channel: int | None = None
    emhc_channel: int | None = None
    type1_channel: int | None = None
    type2_channel: int | None = None

    def uses_nonbaseline_channel_config(self) -> bool:
        return any(
            value is not None
            for value in (
                self.channel_config,
                self.membrane_channel,
                self.dapi_channel,
                self.i_channel,
                self.iia_channel,
                self.iib_channel,
                self.iix_channel,
                self.emhc_channel,
                self.type1_channel,
                self.type2_channel,
            )
        )


def setup_logging(output_dir: Path) -> None:
    """Configure logging to both console and file."""
    log_file = output_dir / "batch_run.log"
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)
    logger.info(f"Batch run log: {log_file}")


def build_batch_command(
    input_file: Path,
    output_dir: Path,
    channel_overrides: BatchChannelOverrides,
    classifier_path: Path | None = None,
    downsample_factor: int | None = None,
    crop_auto: bool = True,
    crop_ds: int | None = None,
    crop_pad: int | None = None,
    crop_min_size: int | None = None,
    export_diagnostics: bool = False,
    retain_mode: str = "full",
    reuse_artifacts: str = "never",
) -> list[str]:
    """
    Build the frozen v0 pipeline command for a single image.

    Args:
        input_file: Path to .czi or .tiff file
        output_dir: Output directory for this image

    Returns:
        List of command arguments to pass to subprocess
    """
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "src.run_pipeline",
        "--input",
        str(input_file.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--typing-preprocess",
        V0_PARAMS["typing_preprocess"],
        "--typing-tile-size",
        str(V0_PARAMS["typing_tile_size"]),
        "--typing-erode-px",
        str(V0_PARAMS["typing_erode_px"]),
        "--classifier-path",
        str((classifier_path or (PROJECT_ROOT / V0_PARAMS["classifier_path"])).resolve()),
        "--model-confidence-threshold",
        str(V0_PARAMS["model_confidence_threshold"]),
        "--model-margin-threshold",
        str(V0_PARAMS["model_margin_threshold"]),
        "--downsample-factor",
        str(downsample_factor or V0_PARAMS["downsample_factor"]),
        "--retain-mode",
        retain_mode,
        "--reuse-artifacts",
        reuse_artifacts,
    ]
    if not crop_auto:
        cmd.append("--no-crop-auto")
    for flag, value in (
        ("--crop-ds", crop_ds),
        ("--crop-pad", crop_pad),
        ("--crop-min-size", crop_min_size),
    ):
        if value is not None:
            cmd.extend([flag, str(value)])
    if export_diagnostics:
        cmd.append("--export-diagnostics")

    if not channel_overrides.uses_nonbaseline_channel_config():
        cmd.extend(
            [
                "--type1-channel",
                str(V0_PARAMS["type1_channel"]),
                "--type2-channel",
                str(V0_PARAMS["type2_channel"]),
                "--membrane-channel",
                str(V0_PARAMS["membrane_channel"]),
            ]
        )
        return cmd

    if channel_overrides.channel_config is not None:
        cmd.extend(["--channel-config", str(channel_overrides.channel_config.resolve())])

    explicit_flags: list[tuple[str, int | Path | None]] = [
        ("--membrane-channel", channel_overrides.membrane_channel),
        ("--dapi-channel", channel_overrides.dapi_channel),
        ("--i-channel", channel_overrides.i_channel),
        ("--iia-channel", channel_overrides.iia_channel),
        ("--iib-channel", channel_overrides.iib_channel),
        ("--iix-channel", channel_overrides.iix_channel),
        ("--emhc-channel", channel_overrides.emhc_channel),
        ("--type1-channel", channel_overrides.type1_channel),
        ("--type2-channel", channel_overrides.type2_channel),
    ]
    for flag, value in explicit_flags:
        if value is not None:
            cmd.extend([flag, str(value)])

    return cmd


def output_stem(input_file: Path) -> str:
    """Match src.run_pipeline output filename normalization."""
    return input_file.resolve().stem.replace(" ", "_")


def _pipeline_timing_lines(stdout: str) -> list[str]:
    """Keep useful child-stage telemetry without logging its full summary dictionary."""
    prefixes = ("[", "Cellpose device:", "reused compatible", "total runtime:")
    return [line for line in stdout.splitlines() if line.startswith(prefixes)]


def _load_input_manifest(
    path: Path,
    *,
    input_root: Path | None = None,
) -> list[tuple[str, Path]]:
    df = pd.read_csv(path)
    if "image_id" not in df.columns:
        raise ValueError(f"{path} missing column: image_id")
    path_columns = {"input_path", "input_relpath"} & set(df.columns)
    if len(path_columns) != 1:
        raise ValueError(f"{path} must contain exactly one of input_path or input_relpath")
    path_column = next(iter(path_columns))
    if path_column == "input_relpath" and input_root is None:
        raise ValueError(f"{path} uses input_relpath; provide --input-root")

    rows: list[tuple[str, Path]] = []
    for row in df.itertuples(index=False):
        image_id = str(row.image_id).strip()
        if not image_id:
            raise ValueError(f"{path} contains blank image_id")
        raw_path = Path(str(getattr(row, path_column))).expanduser()
        if path_column == "input_relpath":
            if raw_path.is_absolute() or ".." in raw_path.parts:
                raise ValueError(f"{path} contains unsafe input_relpath: {raw_path}")
            input_path = input_root / raw_path
        else:
            input_path = raw_path
        rows.append((image_id, input_path))
    return rows


def _canonicalize_output_names(image_output_dir: Path, stem: str, canonical_image_id: str) -> None:
    if stem == canonical_image_id:
        return
    for path in sorted(image_output_dir.glob(f"{stem}*")):
        suffix = path.name[len(stem) :]
        target = image_output_dir / f"{canonical_image_id}{suffix}"
        if target.exists():
            target.unlink()
        path.rename(target)


def run_single_image(
    input_file: Path,
    output_dir: Path,
    channel_overrides: BatchChannelOverrides,
    image_name: str | None = None,
    classifier_path: Path | None = None,
    downsample_factor: int | None = None,
    crop_auto: bool = True,
    crop_ds: int | None = None,
    crop_pad: int | None = None,
    crop_min_size: int | None = None,
    export_diagnostics: bool = False,
    retain_mode: str = "full",
    reuse_artifacts: str = "never",
) -> dict:
    """
    Process a single image through the v0 pipeline.

    Args:
        input_file: Path to input image
        output_dir: Base output directory

    Returns:
        Dictionary with status, image name, error message (if any), fiber count, etc.
    """
    result = {
        "image_name": image_name or input_file.stem,
        "status": "success",
        "error": None,
        "fiber_count": None,
        "summary_path": None,
        "feature_diagnostics_path": None,
    }

    # Create per-image output directory
    image_output_dir = output_dir / result["image_name"]
    image_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing: {input_file.name}")

    # Build and run command
    cmd = build_batch_command(
        input_file,
        image_output_dir,
        channel_overrides=channel_overrides,
        classifier_path=classifier_path,
        downsample_factor=downsample_factor,
        crop_auto=crop_auto,
        crop_ds=crop_ds,
        crop_pad=crop_pad,
        crop_min_size=crop_min_size,
        export_diagnostics=export_diagnostics,
        retain_mode=retain_mode,
        reuse_artifacts=reuse_artifacts,
    )

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1-hour timeout per image
            check=True,
            cwd=PROJECT_ROOT,
        )
        logger.info(f"✓ Completed: {input_file.name}")
        for line in _pipeline_timing_lines(completed.stdout):
            logger.info(f"  pipeline: {line}")

        # Try to read fiber count from output CSV
        stem = output_stem(input_file)
        _canonicalize_output_names(image_output_dir, stem, str(result["image_name"]))
        stem = str(result["image_name"])
        fibers_csv = image_output_dir / f"{stem}_fibers.csv"
        if fibers_csv.exists():
            df = pd.read_csv(fibers_csv)
            result["fiber_count"] = len(df)
            result["summary_path"] = str(image_output_dir / f"{stem}_summary.csv")
            diagnostics_csv = image_output_dir / f"{stem}_feature_diagnostics.csv"
            if diagnostics_csv.exists():
                result["feature_diagnostics_path"] = str(diagnostics_csv)

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Pipeline exceeded 1-hour timeout"
        logger.error(f"✗ Timeout: {input_file.name}")

    except subprocess.CalledProcessError as e:
        result["status"] = "failed"
        result["error"] = f"Exit code {e.returncode}: {e.stderr[:200]}"
        logger.error(f"✗ Failed: {input_file.name} — {result['error']}")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"✗ Error: {input_file.name} — {result['error']}")

    return result


def find_input_files(input_dir: Path) -> list[Path]:
    """
    Find all .czi or .tiff files in a directory.

    Args:
        input_dir: Directory to search

    Returns:
        List of image file paths
    """
    files = list(input_dir.glob("*.czi"))
    files += list(input_dir.glob("*.tif"))
    files += list(input_dir.glob("*.tiff"))
    files = sorted(set(files))  # Remove duplicates and sort
    return files


def main() -> None:
    """Run batch pipeline."""
    parser = argparse.ArgumentParser(
        description="Batch run fibertyping v0 pipeline on a folder of images.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=False,
        default=None,
        help="Directory containing .czi/.tif/.tiff files",
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        required=False,
        default=None,
        help=(
            "CSV with image_id,input_path or portable image_id,input_relpath rows "
            "for explicit per-image inputs."
        ),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Root directory used to resolve input_relpath values in --input-manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/v0_batch)",
    )
    parser.add_argument(
        "--show-v0-params",
        action="store_true",
        help="Print v0 parameters and exit",
    )
    parser.add_argument(
        "--classifier-path",
        type=Path,
        default=None,
        help=(
            "Optional classifier override. When omitted, the frozen v0 alpha classifier is used."
        ),
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=None,
        help="Override segmentation downsample factor for this batch.",
    )
    crop_group = parser.add_mutually_exclusive_group()
    crop_group.add_argument(
        "--crop-auto",
        dest="crop_auto",
        action="store_true",
        default=True,
        help="Automatically crop to the detected tissue field (default).",
    )
    crop_group.add_argument(
        "--no-crop-auto",
        dest="crop_auto",
        action="store_false",
        help="Segment each complete image without automatic tissue cropping.",
    )
    parser.add_argument(
        "--crop-ds",
        type=int,
        default=None,
        help="Optional automatic-crop downsampling factor forwarded to the pipeline.",
    )
    parser.add_argument(
        "--crop-pad",
        type=int,
        default=None,
        help="Optional automatic-crop padding in pixels forwarded to the pipeline.",
    )
    parser.add_argument(
        "--crop-min-size",
        type=int,
        default=None,
        help="Optional minimum tissue size for automatic cropping, forwarded to the pipeline.",
    )
    parser.add_argument(
        "--channel-config",
        type=Path,
        default=None,
        help=(
            "Optional panel-aware channel config. When omitted, the frozen v0 "
            "channel defaults are used."
        ),
    )
    parser.add_argument(
        "--panel-config",
        type=Path,
        default=None,
        help="Preferred alias for --channel-config, including canonical semantic panel YAML.",
    )
    parser.add_argument("--membrane-channel", type=int, default=None)
    parser.add_argument("--dapi-channel", type=int, default=None)
    parser.add_argument("--i-channel", type=int, default=None)
    parser.add_argument("--iia-channel", type=int, default=None)
    parser.add_argument("--iib-channel", type=int, default=None)
    parser.add_argument("--iix-channel", type=int, default=None)
    parser.add_argument("--emhc-channel", type=int, default=None)
    parser.add_argument("--type1-channel", type=int, default=None)
    parser.add_argument("--type2-channel", type=int, default=None)
    parser.add_argument(
        "--export-diagnostics",
        action="store_true",
        help=(
            "Write optional per-image *_feature_diagnostics.csv files for model/feature debugging."
        ),
    )
    parser.add_argument(
        "--retain-mode",
        type=str,
        default="full",
        choices=["full", "tables", "summary"],
        help=(
            "Control retained per-image outputs after successful runs. "
            "'tables' removes label TIFFs; 'summary' keeps only summary CSVs."
        ),
    )
    parser.add_argument(
        "--reuse-artifacts",
        choices=["auto", "never", "required"],
        default="never",
        help="Reuse compatible cached fiber labels within each per-image output directory.",
    )
    parser.add_argument(
        "--split-czi-scenes",
        action="store_true",
        help=(
            "Opt in to exporting every true Zeiss scene from multi-scene CZI inputs and "
            "processing the resulting scalar section TIFFs sequentially."
        ),
    )

    args = parser.parse_args()
    if args.channel_config is not None and args.panel_config is not None:
        parser.error("use only one of --panel-config and --channel-config")

    # Show v0 params if requested
    if args.show_v0_params:
        print("V0 Frozen Parameters:")
        for k, v in V0_PARAMS.items():
            print(f"  {k}: {v}")
        return

    # Validate input
    if bool(args.input_dir) == bool(args.input_manifest):
        print("Error: provide exactly one of --input-dir or --input-manifest")
        sys.exit(1)
    if args.input_root is not None and args.input_manifest is None:
        print("Error: --input-root requires --input-manifest")
        sys.exit(1)

    manifest_rows: list[tuple[str, Path]] | None = None
    input_dir = args.input_dir
    if args.input_manifest is not None:
        if not args.input_manifest.is_file():
            print(f"Error: Input manifest does not exist: {args.input_manifest}")
            sys.exit(1)
        try:
            manifest_rows = _load_input_manifest(args.input_manifest, input_root=args.input_root)
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
    else:
        if input_dir is None or not input_dir.is_dir():
            print(f"Error: Input directory does not exist: {input_dir}")
            sys.exit(1)

    # Set output directory
    if args.output_dir is None:
        output_dir = Path("outputs") / "v0_batch"
    else:
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    channel_overrides = BatchChannelOverrides(
        channel_config=args.panel_config or args.channel_config,
        membrane_channel=args.membrane_channel,
        dapi_channel=args.dapi_channel,
        i_channel=args.i_channel,
        iia_channel=args.iia_channel,
        iib_channel=args.iib_channel,
        iix_channel=args.iix_channel,
        emhc_channel=args.emhc_channel,
        type1_channel=args.type1_channel,
        type2_channel=args.type2_channel,
    )

    if manifest_rows is not None:
        logger.info(f"Input manifest: {args.input_manifest}")
    else:
        logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Find images
    if manifest_rows is not None:
        image_rows = manifest_rows
    else:
        image_files = find_input_files(input_dir)
        if not image_files:
            logger.error(f"No .czi/.tif/.tiff files found in {input_dir}")
            sys.exit(1)
        image_rows = [(path.stem, path) for path in image_files]

    if args.split_czi_scenes:
        expanded_rows: list[tuple[str, Path]] = []
        scene_export_root = output_dir / "raw_scene_exports"
        for image_name, image_path in image_rows:
            if image_path.suffix.lower() != ".czi":
                expanded_rows.append((image_name, image_path))
                continue
            scene_dir = scene_export_root / image_name
            expected_scene_count = len(discover_czi_scenes(image_path))
            if expected_scene_count == 0:
                expanded_rows.append((image_name, image_path))
                continue
            existing_scenes = sorted(scene_dir.glob("*_section-*.tif"))
            if existing_scenes:
                if len(existing_scenes) != expected_scene_count:
                    raise ValueError(
                        f"Existing scene export is incomplete for {image_path}: expected "
                        f"{expected_scene_count}, found {len(existing_scenes)}."
                    )
                scenes = existing_scenes
            else:
                scenes = export_czi_scenes(image_path, scene_dir)
            for section_number, scene_path in enumerate(scenes, start=1):
                expanded_rows.append((f"{image_name}_section-{section_number:02d}", scene_path))
        image_rows = expanded_rows

    logger.info(f"Found {len(image_rows)} image(s) to process")
    logger.info("V0 Parameters:")
    for k, v in V0_PARAMS.items():
        logger.info(f"  {k}: {v}")
    if args.classifier_path is not None:
        logger.info(f"Override: classifier_path={args.classifier_path}")
    if args.downsample_factor is not None:
        logger.info(f"Override: downsample_factor={args.downsample_factor}")
    if not args.crop_auto:
        logger.warning("Override: automatic cropping disabled")
    for key in ("crop_ds", "crop_pad", "crop_min_size"):
        value = getattr(args, key)
        if value is not None:
            logger.info(f"Override: {key}={value}")
    if channel_overrides.uses_nonbaseline_channel_config():
        logger.warning(
            "This batch run is using channel/config overrides and is not "
            "the strict frozen v0 baseline."
        )
        if channel_overrides.channel_config is not None:
            logger.warning(f"  channel_config: {channel_overrides.channel_config}")
        for key in (
            "membrane_channel",
            "dapi_channel",
            "i_channel",
            "iia_channel",
            "iib_channel",
            "iix_channel",
            "emhc_channel",
            "type1_channel",
            "type2_channel",
        ):
            value = getattr(args, key)
            if value is not None:
                logger.warning(f"  override {key}: {value}")

    # Process each image
    results = []
    for i, (image_name, image_file) in enumerate(image_rows, 1):
        logger.info(f"\n[{i}/{len(image_rows)}] Processing image...")
        result = run_single_image(
            image_file,
            output_dir,
            channel_overrides=channel_overrides,
            image_name=image_name,
            classifier_path=args.classifier_path,
            downsample_factor=args.downsample_factor,
            crop_auto=args.crop_auto,
            crop_ds=args.crop_ds,
            crop_pad=args.crop_pad,
            crop_min_size=args.crop_min_size,
            export_diagnostics=args.export_diagnostics,
            retain_mode=args.retain_mode,
            reuse_artifacts=args.reuse_artifacts,
        )
        results.append(result)

    # Save batch summary
    summary_df = pd.DataFrame(results)
    summary_path = output_dir / "batch_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\nBatch summary saved: {summary_path}")

    # Print summary
    success_count = len([r for r in results if r["status"] == "success"])
    failed_count = len([r for r in results if r["status"] != "success"])
    logger.info("\nBatch Results:")
    logger.info(f"  Success: {success_count}/{len(results)}")
    logger.info(f"  Failed: {failed_count}/{len(results)}")

    if failed_count > 0:
        logger.warning("Failed images:")
        for r in results:
            if r["status"] != "success":
                logger.warning(f"  {r['image_name']}: {r['status']} — {r['error']}")

    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
