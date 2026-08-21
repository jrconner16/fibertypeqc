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

## Executable Reference Command

From the repository root, run the public-safe synthetic fixture through the frozen classifier,
merge the supplied review corrections, and validate the declared outputs:

```bash
uv run python -m scripts.run_reference
```

The command writes to `outputs/reference/`. To rerun only the validator:

```bash
uv run python -m scripts.validate_reference_outputs \
  --output-dir outputs/reference
```

The synthetic reference uses a supplied exact label mask to keep golden tables independent of
Cellpose device behavior. It therefore validates deterministic image loading, quantification,
frozen-model prediction, QC summary, review merge, artifact schemas, and digests. It does not claim
to validate segmentation or biology.

The review UI can be opened against the generated reference artifacts:

```bash
uv run python -m scripts.review_labels_napari \
  --image examples/reference/synthetic_reference.tif \
  --labels outputs/reference/synthetic_reference_cellpose_labels.tif \
  --fibers outputs/reference/synthetic_reference_fibers.csv \
  --channel-config examples/reference/panel.yaml \
  --output outputs/reference/synthetic_reference_fibers_manual_review.csv
```

To merge an interactively reviewed table instead of the supplied deterministic corrections:

```bash
uv run python -m scripts.merge_reviewed_labels \
  --fibers outputs/reference/synthetic_reference_fibers.csv \
  --review outputs/reference/synthetic_reference_fibers_manual_review.csv \
  --panel-config examples/reference/panel.yaml \
  --output outputs/reference/synthetic_reference_fibers_reviewed.csv
```

## What The Demo Validates

- Executable public mechanics path from tracked inputs.
- Frozen-model artifact and fixture digest verification.
- Expected output file schema and review-table integration.
- Example QC field interpretation for an alpha review-assisted pipeline.

## What The Demo Does Not Validate

- Full biological equivalence to MyoSight across cohorts.
- Cellpose segmentation repeatability or quality.
- Final clinical/research conclusions without image-level review and validation.
- Generalization to arbitrary staining/channel schemas.

FiberTypeQC v0.1.1-alpha remains an alpha workflow release and should be treated as such in
manuscript or archive language.
