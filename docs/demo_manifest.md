# Demo Manifest (v0.1.1-alpha)

This document defines the public demo artifacts included with FiberTypeQC v0.1.1-alpha.

## Included Demo Files

Under `examples/demo_outputs/`:

- `demo_fiber_table.csv`:
  Synthetic/sanitized per-fiber-style table for schema and QC-flag communication.
- `demo_batch_summary.csv`:
  Synthetic/sanitized batch summary table with representative status/count fields.
- `demo_review_corrections.csv`:
  Synthetic/sanitized manual review correction table matching merge schema.
- `demo_segmentation.png`:
  Segmentation view screenshot (raw membrane context with label overlay).
- `demo_fibertype_overlay.png`:
  Fiber-type overlay screenshot for full-section visual context.
- `demo_napari_review_ui_overview.png`:
  Napari review UI screenshot showing full overlay/workflow context.
- `demo_napari_review_zoom.png`:
  Napari review UI zoomed screenshot showing per-fiber confidence/probability context.
- `demo_batch_summary_plot.png`:
  Clipped batch/validation summary figure suitable for README display.

## Data Status

- CSV demo files in `examples/demo_outputs/` are synthetic/sanitized examples.
- PNG demo images are public-safe screenshots prepared for workflow communication.
- Raw microscopy images are not included in this demo package.
- Screenshots derived from published/cleared JAG images should remain de-identified in naming and
  should avoid exposing private paths in UI captures.

## Intended Commands For Demo Reproduction

Single-image pipeline:

```bash
uv run python -m scripts.run_pipeline \
  --input test_inputs/demo_screenshots/demo_image_a.czi \
  --output-dir outputs/demo_run/demo_image_a \
  --type1-channel 0 \
  --type2-channel 1 \
  --membrane-channel 2 \
  --typing-preprocess tile_subtract \
  --typing-tile-size 256 \
  --typing-erode-px 2 \
  --classifier-path data/models/rebaseline_tile_v2_p75p90_iib_iia_iix.joblib
```

Batch pipeline:

```bash
uv run python -m scripts.run_batch \
  --input-dir path/to/demo_inputs \
  --output-dir outputs/demo_batch
```

Review UI:

```bash
uv run python -m scripts.review_labels_napari \
  --image test_inputs/demo_screenshots/demo_image_a.czi \
  --labels outputs/demo_run/demo_image_a/demo_image_a_cellpose_labels.tif \
  --fibers outputs/demo_run/demo_image_a/demo_image_a_fibers.csv \
  --output outputs/demo_run/demo_image_a/demo_image_a_fibers_manual_review.csv
```

Merge review labels:

```bash
uv run python -m scripts.merge_reviewed_labels \
  --fibers outputs/demo_run/demo_image_a/demo_image_a_fibers.csv \
  --review outputs/demo_run/demo_image_a/demo_image_a_fibers_manual_review.csv \
  --output outputs/demo_run/demo_image_a/demo_image_a_fibers_final.csv
```

## What The Demo Validates

- Public workflow documentation completeness.
- Expected output file schema and review-table integration.
- Example QC field interpretation for an alpha review-assisted pipeline.

## What The Demo Does Not Validate

- Full biological equivalence to MyoSight across cohorts.
- Final clinical/research conclusions without image-level review and validation.
- Generalization to arbitrary staining/channel schemas.

FiberTypeQC v0.1.1-alpha remains an alpha workflow release and should be treated as such in
manuscript or archive language.
