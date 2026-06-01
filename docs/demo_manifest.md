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
- `demo_segmentation_overlay_placeholder.md`:
  Placeholder for a publication-safe segmentation overlay screenshot.
- `demo_fibertype_overlay_placeholder.md`:
  Placeholder for a publication-safe fiber-type overlay screenshot.
- `demo_napari_review_ui_placeholder.md`:
  Placeholder for a publication-safe Napari review UI screenshot.
- `demo_batch_validation_summary_placeholder.md`:
  Placeholder for batch/validation summary screenshot(s).

## Data Status

- CSV demo files in `examples/demo_outputs/` are synthetic/sanitized examples.
- Placeholder markdown files are not analysis outputs; they indicate intended screenshot locations.
- Raw microscopy images are not included in this demo package.
- If screenshots are added from published/cleared JAG images, they should be exported separately and
  verified as safe for public release before commit.

## Intended Commands For Demo Reproduction

Single-image pipeline:

```bash
uv run python -m scripts.run_pipeline \
  --input path/to/demo_image.czi \
  --output-dir outputs/demo_run/demo_image \
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
  --image path/to/demo_image.czi \
  --labels outputs/demo_run/demo_image/demo_image_cellpose_labels.tif \
  --fibers outputs/demo_run/demo_image/demo_image_fibers.csv \
  --output outputs/demo_run/demo_image/demo_image_fibers_manual_review.csv
```

Merge review labels:

```bash
uv run python -m scripts.merge_reviewed_labels \
  --fibers outputs/demo_run/demo_image/demo_image_fibers.csv \
  --review outputs/demo_run/demo_image/demo_image_fibers_manual_review.csv \
  --output outputs/demo_run/demo_image/demo_image_fibers_final.csv
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
