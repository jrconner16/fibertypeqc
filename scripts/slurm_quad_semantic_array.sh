#!/bin/bash
# Submit with: sbatch --array=0-$((N-1)) scripts/slurm_quad_semantic_array.sh
# Set FIBERTYPEQC_ROOT, QUAD_INPUT_ROOT, QUAD_OUTPUT_ROOT, and QUAD_MANIFEST.
# The manifest must contain only canonical raw CZI files.
#SBATCH --job-name=fibertypeqc-quad-semantic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-quad-semantic-%A_%a.out

set -euo pipefail

: "${FIBERTYPEQC_ROOT:?Set the repository path on the cluster.}"
: "${QUAD_INPUT_ROOT:?Set the root containing raw canonical CZI files.}"
: "${QUAD_OUTPUT_ROOT:?Set the output root.}"
: "${QUAD_MANIFEST:?Set the canonical raw-CZI CSV manifest path.}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array.}"

cd "$FIBERTYPEQC_ROOT"

uv run --frozen python -m src.run_quad_semantic_array_task \
  --manifest "$QUAD_MANIFEST" \
  --input-root "$QUAD_INPUT_ROOT" \
  --output-root "$QUAD_OUTPUT_ROOT" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --panel-config manifests/quad_351545_panel.yaml \
  --classifier-path data/models/quad_351545_logistic_semantic_v1.joblib \
  --model-manifest manifests/quad_351545_logistic_semantic_v1.yaml \
  --fiber-downsample-factor 2 \
  --fiber-diameter 30
