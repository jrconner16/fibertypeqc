#!/bin/bash
# Submit with an explicit array range, for example: sbatch --array=0-17 scripts/slurm_panel_array.sh
# Set FIBERTYPEQC_ROOT, PANEL_INPUT_ROOT, PANEL_OUTPUT_ROOT, and PANEL_MANIFEST first.
# Add your cluster-specific account, partition, and module setup at submission time or below.
#SBATCH --job-name=fibertypeqc-panel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-panel-%A_%a.out

set -euo pipefail

: "${FIBERTYPEQC_ROOT:?Set the repository path on the cluster.}"
: "${PANEL_INPUT_ROOT:?Set the root containing the image files.}"
: "${PANEL_OUTPUT_ROOT:?Set the output root.}"
: "${PANEL_MANIFEST:?Set the private CSV manifest path.}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array.}"

cd "$FIBERTYPEQC_ROOT"

uv run python -m research.run_panel_array_task \
  --manifest "$PANEL_MANIFEST" \
  --input-root "$PANEL_INPUT_ROOT" \
  --output-root "$PANEL_OUTPUT_ROOT" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --panel-config manifests/i_iia_laminin_dapi_panel.yaml \
  --fiber-downsample-factor 2 \
  --fiber-diameter 30 \
  --nuclei-downsample-factor 2 \
  --nuclei-diameter 15 \
  --nuclei-min-size 30
