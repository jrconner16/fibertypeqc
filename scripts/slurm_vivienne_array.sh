#!/bin/bash
# Submit with an explicit array range, for example: sbatch --array=0-17 scripts/slurm_vivienne_array.sh
# Set FIBERTYPEQC_ROOT, VIVIENNE_INPUT_ROOT, VIVIENNE_OUTPUT_ROOT, and VIVIENNE_MANIFEST first.
# Add your cluster-specific account, partition, and module setup at submission time or below.
#SBATCH --job-name=fibertypeqc-vivienne
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-vivienne-%A_%a.out

set -euo pipefail

: "${FIBERTYPEQC_ROOT:?Set the repository path on the cluster.}"
: "${VIVIENNE_INPUT_ROOT:?Set the root containing the image files.}"
: "${VIVIENNE_OUTPUT_ROOT:?Set the output root.}"
: "${VIVIENNE_MANIFEST:?Set the private CSV manifest path.}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array.}"

cd "$FIBERTYPEQC_ROOT"

uv run python -m src.run_panel_array_task \
  --manifest "$VIVIENNE_MANIFEST" \
  --input-root "$VIVIENNE_INPUT_ROOT" \
  --output-root "$VIVIENNE_OUTPUT_ROOT" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --panel-config manifests/vivienne_i_iia_dapi_panel.yaml \
  --fiber-downsample-factor 2 \
  --fiber-diameter 30 \
  --nuclei-downsample-factor 2 \
  --nuclei-diameter 15 \
  --nuclei-min-size 30
