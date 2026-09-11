#!/usr/bin/env bash
# Required environment: FIBERTYPEQC_ROOT, QUAD_LOCK_DIR, QUAD_MANIFEST, QUAD_SCORED_DIR
# Submit 27 tasks: sbatch --array=0-26 scripts/slurm_score_jag1_quad_locked_candidate.sh
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail
: "${FIBERTYPEQC_ROOT:?}"
: "${QUAD_LOCK_DIR:?}"
: "${QUAD_MANIFEST:?}"
: "${QUAD_SCORED_DIR:?}"
cd "$FIBERTYPEQC_ROOT"
/home/ch197590/projects/fibertypeqc/.venv/bin/python -m src.score_jag1_quad_locked_candidate \
  --lock-dir "$QUAD_LOCK_DIR" --manifest "$QUAD_MANIFEST" --output-dir "$QUAD_SCORED_DIR" \
  --task-index "${SLURM_ARRAY_TASK_ID:?}"
