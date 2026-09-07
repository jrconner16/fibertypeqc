"""Recompute diagnostics and ROI matches for one immutable-mask transfer image."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TransferRow:
    image_id: str
    input_path: Path
    labels_path: Path
    myosight_results_dir: Path


REQUIRED_SEMANTIC_FEATURES = tuple(
    f"type_{marker}.{suffix}"
    for marker in ("iia", "iib")
    for suffix in (
        "mean",
        "p90",
        "coverage_high",
        "snr_mean",
        "snr_p90",
        "center_mean",
        "edge_mean",
    )
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use an explicit cached label TIFF to regenerate current diagnostics, then create "
            "exhaustive MyoSight-to-pipeline match candidates for the same image."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--matching-root", type=Path, required=True)
    parser.add_argument("--channel-config", type=Path, required=True)
    parser.add_argument("--typing-preprocess", required=True)
    parser.add_argument("--typing-bg-quantile", type=float, required=True)
    parser.add_argument("--typing-tile-size", type=int, required=True)
    parser.add_argument("--typing-bg-sigma", type=float, required=True)
    parser.add_argument("--typing-smooth-sigma", type=float, required=True)
    parser.add_argument("--typing-erode-px", type=int, required=True)
    parser.add_argument("--coverage-quantile", type=float, required=True)
    parser.add_argument("--sensitivity", type=float, required=True)
    parser.add_argument("--mixed-strictness", type=float, required=True)
    parser.add_argument("--max-centroid-distance", type=float, default=25.0)
    return parser


def _resolve_manifest_path(manifest_path: Path, value: object) -> Path:
    path = Path(str(value).strip()).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def load_manifest(path: Path) -> list[TransferRow]:
    table = pd.read_csv(path)
    required = {"image_id", "input_path", "labels_path", "myosight_results_dir"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Transfer manifest missing required columns: {missing}")
    if table["image_id"].astype(str).str.strip().duplicated().any():
        raise ValueError("Transfer manifest image_id values must be unique.")

    rows: list[TransferRow] = []
    for _, record in table.iterrows():
        image_id = str(record["image_id"]).strip()
        if not image_id:
            raise ValueError("Transfer manifest contains a blank image_id.")
        input_path = _resolve_manifest_path(path, record["input_path"])
        labels_path = _resolve_manifest_path(path, record["labels_path"])
        results_dir = _resolve_manifest_path(path, record["myosight_results_dir"])
        for candidate, description in [
            (input_path, "source image"),
            (labels_path, "cached labels"),
            (results_dir / "Results.txt", "MyoSight Results.txt"),
            (results_dir / "ROISet.zip", "MyoSight ROISet.zip"),
        ]:
            if not candidate.exists():
                raise FileNotFoundError(f"{description} not found for {image_id}: {candidate}")
        rows.append(
            TransferRow(
                image_id=image_id,
                input_path=input_path,
                labels_path=labels_path,
                myosight_results_dir=results_dir,
            )
        )
    if not rows:
        raise ValueError("Transfer manifest contains no rows.")
    return rows


def build_diagnostics_command(
    row: TransferRow,
    *,
    output_dir: Path,
    channel_config: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.run_pipeline",
        "--input",
        str(row.input_path),
        "--labels-path",
        str(row.labels_path),
        "--output-dir",
        str(output_dir),
        "--channel-config",
        str(channel_config),
        "--no-crop-auto",
        "--export-diagnostics",
        "--retain-mode",
        "tables",
        "--typing-preprocess",
        str(args.typing_preprocess),
        "--typing-bg-quantile",
        str(args.typing_bg_quantile),
        "--typing-tile-size",
        str(args.typing_tile_size),
        "--typing-bg-sigma",
        str(args.typing_bg_sigma),
        "--typing-smooth-sigma",
        str(args.typing_smooth_sigma),
        "--typing-erode-px",
        str(args.typing_erode_px),
        "--coverage-quantile",
        str(args.coverage_quantile),
        "--sensitivity",
        str(args.sensitivity),
        "--mixed-strictness",
        str(args.mixed_strictness),
    ]


def build_matching_command(
    row: TransferRow,
    *,
    output_dir: Path,
    max_centroid_distance: float,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.compare_roi_boundaries",
        "--myosight-results-dir",
        str(row.myosight_results_dir),
        "--myosight-image",
        str(row.input_path),
        "--pipeline-labels",
        str(row.labels_path),
        "--output-dir",
        str(output_dir),
        "--name",
        row.image_id,
        "--no-panel",
        "--max-centroid-distance",
        str(max_centroid_distance),
    ]


def _canonicalize_diagnostics(output_dir: Path, image_id: str) -> Path:
    paths = sorted(output_dir.glob("*_feature_diagnostics.csv"))
    if len(paths) != 1:
        raise ValueError(f"Expected one diagnostics CSV in {output_dir}; found {len(paths)}.")
    original_stem = paths[0].name.removesuffix("_feature_diagnostics.csv")
    for path in sorted(output_dir.glob(f"{original_stem}*")):
        target = output_dir / f"{image_id}{path.name[len(original_stem) :]}"
        if target != path:
            if target.exists():
                if not target.samefile(path):
                    raise FileExistsError(f"Refusing to overwrite canonical output: {target}")
                # APFS is typically case-insensitive. A case-only canonical rename therefore
                # appears to target an existing file even though it is the source itself.
                temporary = path.with_name(f".{path.name}.canonicalizing-{uuid.uuid4().hex}")
                path.rename(temporary)
                temporary.rename(target)
                continue
            path.rename(target)
    diagnostics_path = output_dir / f"{image_id}_feature_diagnostics.csv"
    diagnostics = pd.read_csv(diagnostics_path, nrows=1)
    missing = sorted(set(REQUIRED_SEMANTIC_FEATURES).difference(diagnostics.columns))
    if missing:
        raise ValueError(f"Current diagnostics missing required transfer features: {missing}")
    return diagnostics_path


def _run(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = build_parser().parse_args()
    rows = load_manifest(args.manifest.resolve())
    if args.task_index < 0 or args.task_index >= len(rows):
        raise ValueError(f"--task-index must be in [0, {len(rows) - 1}].")
    row = rows[args.task_index]
    diagnostics_dir = args.diagnostics_root.resolve() / row.image_id
    matching_dir = args.matching_root.resolve() / row.image_id
    if diagnostics_dir.exists() or matching_dir.exists():
        raise FileExistsError(f"Refusing to overwrite outputs for {row.image_id}.")
    diagnostics_dir.mkdir(parents=True)
    matching_dir.mkdir(parents=True)

    diagnostics_command = build_diagnostics_command(
        row,
        output_dir=diagnostics_dir,
        channel_config=args.channel_config.resolve(),
        args=args,
    )
    matching_command = build_matching_command(
        row,
        output_dir=matching_dir,
        max_centroid_distance=args.max_centroid_distance,
    )
    _run(diagnostics_command)
    diagnostics_path = _canonicalize_diagnostics(diagnostics_dir, row.image_id)
    _run(matching_command)

    provenance = {
        "image_id": row.image_id,
        "input_path": str(row.input_path),
        "labels_path": str(row.labels_path),
        "myosight_results_dir": str(row.myosight_results_dir),
        "diagnostics_path": str(diagnostics_path),
        "diagnostics_command": diagnostics_command,
        "matching_command": matching_command,
        "segmentation_rerun": False,
        "myosight_labels_modified": False,
        "input_sha256": _sha256(row.input_path),
        "labels_sha256": _sha256(row.labels_path),
        "myosight_results_sha256": _sha256(row.myosight_results_dir / "Results.txt"),
        "myosight_roiset_sha256": _sha256(row.myosight_results_dir / "ROISet.zip"),
        "diagnostics_sha256": _sha256(diagnostics_path),
    }
    provenance_path = diagnostics_dir / f"{row.image_id}_fixed_mask_transfer_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"completed image: {row.image_id}")
    print(f"saved diagnostics: {diagnostics_path}")
    print(f"saved matching outputs: {matching_dir}")
    print(f"saved provenance: {provenance_path}")


if __name__ == "__main__":
    main()
