# Review Workflow

The v0.1-alpha workflow is designed around batch pre-labeling followed by targeted manual review.

## Review Priority

Start with fibers where:

- `needs_review` is `True`,
- `model_confidence` is low,
- `model_margin` is low,
- `typing_signal_qc_flags` is not empty,
- the whole image has obvious staining, channel, or segmentation problems.

## Napari Controls

Run:

```bash
uv run python -m scripts.review_labels_napari \
  --image path/to/image.czi \
  --labels outputs/v0_run/image_name/image_name_cellpose_labels.tif \
  --fibers outputs/v0_run/image_name/image_name_fibers.csv \
  --output outputs/v0_run/image_name/image_name_fibers_manual_review.csv
```

Click a labeled fiber, then use the review keys shown in the terminal. The review table can be
merged back into the fiber table with `scripts.merge_reviewed_labels`.

## Display Versus Measurement

The review UI may show contrast-boosted channels to make weak signal easier to inspect. Reviewers
should compare against raw channels when deciding whether signal is biological or background.

## Alpha Caveats

- Review is focused on fiber-type classification, not exhaustive segmentation correction.
- Current channel arguments assume IIb, IIa, and membrane/laminin channels.
- Manual review output is a correction table; it does not rewrite the label mask.
