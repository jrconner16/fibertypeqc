#!/bin/bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage:"
  echo "  $0 INPUT_TIF OUTPUT_DIR"
  exit 1
fi

INPUT="$1"
OUTPUT="$2"

REPO="$HOME/projects/fibertypeqc"
PANEL="$REPO/manifests/ta_28dpi_emhc_panel.yaml"

export PATH="$HOME/.local/bin:$PATH"

cd "$REPO"

mkdir -p "$OUTPUT"

echo "TA eMHC nuclei baseline: v1"
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "Git commit: $(git rev-parse HEAD)"

uv run --frozen python -m src.run_pipeline \
  --input "$INPUT" \
  --output-dir "$OUTPUT" \
  --panel-config "$PANEL" \
  --downsample-factor 2 \
  --diameter 30 \
  --nuclei-downsample-factor 1 \
  --nuclei-diameter 12 \
  --nuclei-min-size 30 \
  --nuclei-cellprob-threshold -1 \
  --nuclei-flow-threshold 0.6 \
  --dapi-preprocess tile_subtract \
  --export-diagnostics
