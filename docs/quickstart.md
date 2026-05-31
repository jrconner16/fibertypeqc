# Quickstart

## Install

```bash
uv sync
```

## Run One Image

```bash
uv run python -m scripts.run_pipeline \
  --input path/to/image.czi \
  --output-dir outputs/v0_run/image_name \
  --type1-channel 0 \
  --type2-channel 1 \
  --membrane-channel 2 \
  --typing-preprocess tile_subtract \
  --typing-tile-size 256 \
  --typing-erode-px 2 \
  --classifier-path data/models/rebaseline_tile_v2_p75p90_iib_iia_iix.joblib
```

Channel meanings in v0.1-alpha:

- `--type1-channel`: IIb marker
- `--type2-channel`: IIa marker
- `--membrane-channel`: membrane/laminin segmentation channel

IIx is inferred as the unstained class relative to the IIb and IIa marker channels.

## Run A Batch

```bash
uv run python -m scripts.run_batch \
  --input-dir path/to/images \
  --output-dir outputs/v0_batch
```

## Review Flags

```bash
uv run python -m scripts.review_labels_napari \
  --image path/to/image.czi \
  --labels outputs/v0_run/image_name/image_name_cellpose_labels.tif \
  --fibers outputs/v0_run/image_name/image_name_fibers.csv \
  --output outputs/v0_run/image_name/image_name_fibers_manual_review.csv
```

## Merge Review

```bash
uv run python -m scripts.merge_reviewed_labels \
  --fibers outputs/v0_run/image_name/image_name_fibers.csv \
  --review outputs/v0_run/image_name/image_name_fibers_manual_review.csv \
  --output outputs/v0_run/image_name/image_name_fibers_final.csv
```
