"""Run one protected Notch3 ROI-to-frozen-mask matching task."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-centroid-distance", type=float, default=25.0)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    required = {"image_id", "raw_czi", "pipeline_labels", "myosight_results_dir"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    if not 0 <= args.task_index < len(manifest):
        raise ValueError(f"--task-index must be in [0, {len(manifest) - 1}].")
    row = manifest.iloc[args.task_index]
    image_id = str(row["image_id"])
    output_dir = args.output_root / image_id
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite matching output: {output_dir}")
    output_dir.mkdir(parents=True)
    command = [
        sys.executable,
        "-m",
        "src.compare_roi_boundaries",
        "--myosight-results-dir",
        str(row["myosight_results_dir"]),
        "--myosight-image",
        str(row["raw_czi"]),
        "--pipeline-labels",
        str(row["pipeline_labels"]),
        "--output-dir",
        str(output_dir),
        "--name",
        image_id,
        "--no-panel",
        "--max-centroid-distance",
        str(args.max_centroid_distance),
    ]
    subprocess.run(command, check=True)
    expected = output_dir / f"{image_id}_match_candidates.csv"
    if not expected.is_file():
        raise FileNotFoundError(f"Matching task did not create candidates: {expected}")
    print(f"completed protected matching: {image_id}")


if __name__ == "__main__":
    main()
