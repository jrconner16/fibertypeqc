#!/usr/bin/env bash
# Required environment: FIBERTYPEQC_ROOT, LOCK_DIR, INPUT_MANIFEST, SCORED_DIR
# Set the array range to the zero-based rows in INPUT_MANIFEST.
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail
: "${FIBERTYPEQC_ROOT:?}"
: "${LOCK_DIR:?}"
: "${INPUT_MANIFEST:?}"
: "${SCORED_DIR:?}"
cd "$FIBERTYPEQC_ROOT"
uv run --frozen python -m src.score_locked_candidate \
  --lock-dir "$LOCK_DIR" --manifest "$INPUT_MANIFEST" --output-dir "$SCORED_DIR" \
  --task-index "${SLURM_ARRAY_TASK_ID:?}"
