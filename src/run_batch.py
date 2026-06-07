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
    downsample_factor: int | None = None,
    export_diagnostics: bool = False,
    retain_mode: str = "full",
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
        str(PROJECT_ROOT / V0_PARAMS["classifier_path"]),
        "--model-confidence-threshold",
        str(V0_PARAMS["model_confidence_threshold"]),
        "--model-margin-threshold",
        str(V0_PARAMS["model_margin_threshold"]),
        "--downsample-factor",
        str(downsample_factor or V0_PARAMS["downsample_factor"]),
        "--retain-mode",
        retain_mode,
    ]
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


def run_single_image(
    input_file: Path,
    output_dir: Path,
    channel_overrides: BatchChannelOverrides,
    downsample_factor: int | None = None,
    export_diagnostics: bool = False,
    retain_mode: str = "full",
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
        "image_name": input_file.stem,
        "status": "success",
        "error": None,
        "fiber_count": None,
        "summary_path": None,
        "feature_diagnostics_path": None,
    }

    # Create per-image output directory
    image_output_dir = output_dir / input_file.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing: {input_file.name}")

    # Build and run command
    cmd = build_batch_command(
        input_file,
        image_output_dir,
        channel_overrides=channel_overrides,
        downsample_factor=downsample_factor,
        export_diagnostics=export_diagnostics,
        retain_mode=retain_mode,
    )

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1-hour timeout per image
            check=True,
            cwd=PROJECT_ROOT,
        )
        logger.info(f"✓ Completed: {input_file.name}")

        # Try to read fiber count from output CSV
        stem = output_stem(input_file)
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
        "--downsample-factor",
        type=int,
        default=None,
        help="Override segmentation downsample factor for this batch.",
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
    parser.add_argument("--membrane-channel", type=int, default=None)
    parser.add_argument("--dapi-channel", type=int, default=None)
    parser.add_argument("--i-channel", type=int, default=None)
    parser.add_argument("--iia-channel", type=int, default=None)
    parser.add_argument("--iib-channel", type=int, default=None)
    parser.add_argument("--iix-channel", type=int, default=None)
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

    args = parser.parse_args()

    # Show v0 params if requested
    if args.show_v0_params:
        print("V0 Frozen Parameters:")
        for k, v in V0_PARAMS.items():
            print(f"  {k}: {v}")
        return

    # Validate input
    if args.input_dir is None:
        print("Error: --input-dir is required (unless using --show-v0-params)")
        sys.exit(1)
    
    input_dir = args.input_dir
    if not input_dir.is_dir():
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
        channel_config=args.channel_config,
        membrane_channel=args.membrane_channel,
        dapi_channel=args.dapi_channel,
        i_channel=args.i_channel,
        iia_channel=args.iia_channel,
        iib_channel=args.iib_channel,
        iix_channel=args.iix_channel,
        type1_channel=args.type1_channel,
        type2_channel=args.type2_channel,
    )

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Find images
    image_files = find_input_files(input_dir)
    if not image_files:
        logger.error(f"No .czi/.tif/.tiff files found in {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(image_files)} image(s) to process")
    logger.info("V0 Parameters:")
    for k, v in V0_PARAMS.items():
        logger.info(f"  {k}: {v}")
    if args.downsample_factor is not None:
        logger.info(f"Override: downsample_factor={args.downsample_factor}")
    if channel_overrides.uses_nonbaseline_channel_config():
        logger.warning(
            "This batch run is using channel/config overrides and is not "
            "the strict frozen v0 baseline."
        )
        if args.channel_config is not None:
            logger.warning(f"  channel_config: {args.channel_config}")
        for key in (
            "membrane_channel",
            "dapi_channel",
            "i_channel",
            "iia_channel",
            "iib_channel",
            "iix_channel",
            "type1_channel",
            "type2_channel",
        ):
            value = getattr(args, key)
            if value is not None:
                logger.warning(f"  override {key}: {value}")

    # Process each image
    results = []
    for i, image_file in enumerate(image_files, 1):
        logger.info(f"\n[{i}/{len(image_files)}] Processing image...")
        result = run_single_image(
            image_file,
            output_dir,
            channel_overrides=channel_overrides,
            downsample_factor=args.downsample_factor,
            export_diagnostics=args.export_diagnostics,
            retain_mode=args.retain_mode,
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
