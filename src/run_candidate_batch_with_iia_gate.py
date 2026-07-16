from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

import pandas as pd

from src.analyze_iia_gate import _apply_iia_gate, _gate_mask, derive_iia_gate_thresholds
from src.run_batch import (
    PROJECT_ROOT,
    V0_PARAMS,
    BatchChannelOverrides,
    _load_input_manifest,
    find_input_files,
    output_stem,
    setup_logging,
)

logger = logging.getLogger("src.run_batch")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the weighted baseline_gb candidate in batch mode and apply the soft IIa "
            "gate to each output fiber table."
        )
    )
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument(
        "--input-manifest",
        type=Path,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classifier-path", type=Path, required=True)
    parser.add_argument("--true-iia-reviewed-glob", type=str, required=True)
    parser.add_argument("--gate-quantile", type=float, default=0.01)
    parser.add_argument("--downsample-factor", type=int, default=None)
    parser.add_argument("--channel-config", type=Path, default=None)
    parser.add_argument("--membrane-channel", type=int, default=None)
    parser.add_argument("--dapi-channel", type=int, default=None)
    parser.add_argument("--i-channel", type=int, default=None)
    parser.add_argument("--iia-channel", type=int, default=None)
    parser.add_argument("--iib-channel", type=int, default=None)
    parser.add_argument("--iix-channel", type=int, default=None)
    parser.add_argument("--type1-channel", type=int, default=None)
    parser.add_argument("--type2-channel", type=int, default=None)
    parser.add_argument("--export-diagnostics", action="store_true")
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
    return parser


def _load_true_iia_reviewed(pattern: str) -> pd.DataFrame:
    files = sorted(Path().glob(pattern))
    if not files:
        raise ValueError(f"No true_iia_hunt reviewed files matched: {pattern}")
    frames = [pd.read_csv(path, low_memory=False) for path in files]
    out = pd.concat(frames, ignore_index=True)
    out["audit_corrected_type"] = (
        out.get("audit_corrected_type", pd.Series("", index=out.index))
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )
    out = out.loc[out["audit_corrected_type"].eq("iia")].copy()
    if out.empty:
        raise ValueError("No confirmed IIa rows found in true_iia_hunt reviewed files.")
    return out


def _build_pipeline_command(
    input_file: Path,
    output_dir: Path,
    classifier_path: Path,
    channel_overrides: BatchChannelOverrides,
    downsample_factor: int | None,
    export_diagnostics: bool,
    retain_mode: str,
) -> list[str]:
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
        str(classifier_path.resolve()),
        "--model-confidence-threshold",
        str(V0_PARAMS["model_confidence_threshold"]),
        "--model-margin-threshold",
        str(V0_PARAMS["model_margin_threshold"]),
        "--downsample-factor",
        str(downsample_factor or V0_PARAMS["downsample_factor"]),
        "--retain-mode",
        retain_mode,
    ]
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
    else:
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
    if export_diagnostics:
        cmd.append("--export-diagnostics")
    return cmd


def _canonicalize_output_names(image_output_dir: Path, stem: str, canonical_image_id: str) -> None:
    if stem == canonical_image_id:
        return
    for path in sorted(image_output_dir.glob(f"{stem}*")):
        suffix = path.name[len(stem) :]
        target = image_output_dir / f"{canonical_image_id}{suffix}"
        if target.exists():
            target.unlink()
        path.rename(target)


def _apply_gate_to_fibers(
    fibers_path: Path,
    diagnostics_path: Path,
    thresholds: dict[str, float],
) -> None:
    fibers = pd.read_csv(fibers_path, low_memory=False)
    diagnostics = pd.read_csv(diagnostics_path, low_memory=False)
    required = {
        "label",
        "type1_mean",
        "type2_mean",
        "type1_coverage",
        "type2_coverage",
        "type1_cov_x_snr",
        "type1_snr_mean",
        "type2_cov_x_snr",
        "type2_snr_mean",
    }
    missing = sorted(required - set(diagnostics.columns))
    if missing:
        raise ValueError(f"{diagnostics_path} missing gate columns: {', '.join(missing)}")

    diagnostics = diagnostics.copy()
    diagnostics["label"] = diagnostics["label"].astype(int)
    pred = diagnostics["fiber_type"].astype(str).str.lower()
    gate_ok = _gate_mask(diagnostics, thresholds)
    gated = _apply_iia_gate(pred, gate_ok)
    gate_df = pd.DataFrame(
        {
            "label": diagnostics["label"].astype(int),
            "fiber_type_pre_iia_gate": pred,
            "gate_iia_ok": gate_ok,
            "fiber_type_gated": gated,
        }
    )

    fibers = fibers.copy()
    fibers["label"] = fibers["label"].astype(int)
    fibers = fibers.drop(
        columns=["fiber_type_pre_iia_gate", "gate_iia_ok"],
        errors="ignore",
    ).merge(
        gate_df,
        on="label",
        how="left",
        validate="one_to_one",
    )
    fibers["fiber_type"] = fibers["fiber_type_gated"].fillna(fibers["fiber_type"]).astype(str)
    fibers = fibers.drop(columns=["fiber_type_gated"])
    fibers.to_csv(fibers_path, index=False)

    diagnostics["fiber_type_pre_iia_gate"] = pred
    diagnostics["gate_iia_ok"] = gate_ok
    gated = _apply_iia_gate(pred, gate_ok)
    diagnostics["fiber_type"] = gated
    diagnostics.to_csv(diagnostics_path, index=False)


def main() -> None:
    args = build_parser().parse_args()
    if bool(args.input_dir) == bool(args.input_manifest):
        raise SystemExit("Provide exactly one of --input-dir or --input-manifest.")
    if args.input_root is not None and args.input_manifest is None:
        raise SystemExit("--input-root requires --input-manifest.")
    manifest_rows: list[tuple[str, Path]] | None = None
    if args.input_manifest is not None:
        if not args.input_manifest.is_file():
            raise SystemExit(f"Input manifest does not exist: {args.input_manifest}")
        manifest_rows = _load_input_manifest(args.input_manifest, input_root=args.input_root)
    else:
        if not args.input_dir.is_dir():
            raise SystemExit(f"Input directory does not exist: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir)
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

    true_iia_reviewed = _load_true_iia_reviewed(args.true_iia_reviewed_glob)
    thresholds = derive_iia_gate_thresholds(true_iia_reviewed, gate_quantile=args.gate_quantile)
    logger.info("Soft IIa gate thresholds:")
    for key, value in thresholds.items():
        logger.info("  %s=%.6f", key, value)

    if manifest_rows is not None:
        image_rows = manifest_rows
    else:
        image_files = find_input_files(args.input_dir)
        if not image_files:
            raise SystemExit(f"No .czi/.tif/.tiff files found in {args.input_dir}")
        image_rows = [(path.stem, path) for path in image_files]
    logger.info("Found %d image(s) to process", len(image_rows))
    if channel_overrides.uses_nonbaseline_channel_config():
        logger.warning(
            "This candidate batch run is using channel/config overrides and is not "
            "the strict frozen v0 baseline channel mapping."
        )
        if args.channel_config is not None:
            logger.warning("  channel_config: %s", args.channel_config)
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
                logger.warning("  override %s: %s", key, value)

    results: list[dict[str, object]] = []
    for index, (image_id, input_file) in enumerate(image_rows, start=1):
        logger.info("\n[%d/%d] Processing %s", index, len(image_rows), input_file.name)
        image_output_dir = args.output_dir / image_id
        image_output_dir.mkdir(parents=True, exist_ok=True)
        cmd = _build_pipeline_command(
            input_file=input_file,
            output_dir=image_output_dir,
            classifier_path=args.classifier_path,
            channel_overrides=channel_overrides,
            downsample_factor=args.downsample_factor,
            export_diagnostics=args.export_diagnostics,
            retain_mode=args.retain_mode,
        )
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
                check=True,
                cwd=PROJECT_ROOT,
            )
            stem = output_stem(input_file)
            _canonicalize_output_names(image_output_dir, stem, image_id)
            fibers_path = image_output_dir / f"{image_id}_fibers.csv"
            diagnostics_path = image_output_dir / f"{image_id}_feature_diagnostics.csv"
            if fibers_path.exists() and diagnostics_path.exists():
                _apply_gate_to_fibers(fibers_path, diagnostics_path, thresholds)
            results.append(
                {
                    "image_name": image_id,
                    "status": "success",
                    "fibers_path": str(fibers_path) if fibers_path.exists() else None,
                }
            )
            logger.info("✓ Completed %s", input_file.name)
        except subprocess.TimeoutExpired:
            logger.error("✗ Timeout: %s", input_file.name)
            results.append({"image_name": input_file.stem, "status": "timeout"})
        except subprocess.CalledProcessError as exc:
            logger.error("✗ Failed: %s", input_file.name)
            if exc.stdout:
                logger.error("stdout:\n%s", exc.stdout[-2000:])
            if exc.stderr:
                logger.error("stderr:\n%s", exc.stderr[-2000:])
            results.append(
                {
                    "image_name": input_file.stem,
                    "status": "failed",
                    "error": f"exit {exc.returncode}",
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("✗ Error: %s — %s", input_file.name, exc)
            results.append(
                {
                    "image_name": input_file.stem,
                    "status": "error",
                    "error": str(exc),
                }
            )

    summary_path = args.output_dir / "batch_summary.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False)
    logger.info("Saved batch summary: %s", summary_path)


if __name__ == "__main__":
    main()
