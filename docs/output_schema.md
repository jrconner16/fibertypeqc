# Output Schema

FiberTypeQC writes one output folder per image.

## Main Files

- `*_cellpose_labels.tif`: integer label mask; `0` is background.
- `*_fibers.csv`: one row per segmented fiber.
- `*_feature_diagnostics.csv`: optional model-development/debugging table written only when
  diagnostics export is enabled.
- `*_summary.csv`: one row with image-level processing settings, counts, QC metrics, and summaries.
- `*_fibers_manual_review.csv`: manual review table written by the Napari review UI.
- `*_weak_labels.csv`: fibers prioritized for review.
- `batch_summary.csv`: batch-level status table from `scripts.run_batch`.

## Fiber Table Columns

Core columns:

- `label`: segmentation label ID.
- `area`: raw label area in pixels.
- `area_um2`: raw label area in square microns when image pixel-size metadata is available.
- `feret_max_px`, `feret_min_px`: maximum and minimum Feret diameters in pixel units.
- `feret_max_um`, `feret_min_um`: Feret diameters in microns when pixel-size metadata is available.
- `area_erode_<N>px`: area after inward label erosion by `N` pixels, when requested.
- `area_erode_<N>px_um2`: eroded area in square microns when pixel-size metadata is available.

The stable `*_fibers.csv` is the conservative biological/review output. Experimental modeling
features are intentionally kept out of this table by default.

Typing features:

- `type1_mean`, `type1_p75`, `type1_p90`, `type1_pctl`, `type1_coverage`: IIb channel features.
- `type2_mean`, `type2_p75`, `type2_p90`, `type2_pctl`, `type2_coverage`: IIa channel features.
- `typing_interior_area`: pixels used for typing after `typing_erode_px`.
- `typing_erode_px`: inward erosion used for typing features.
- `typing_preprocess`: typing channel preprocessing mode.

Classification columns:

- `fiber_type`: predicted class, usually `iib`, `iia`, or `iix`.
- `fiber_type_source`: provenance for the current call. In `v0.2` this is one of
  `direct_marker`, `hybrid_marker`, `residual_inference`, or `model_prediction`.
- `available_markers`: pipe-delimited marker channels available for that run, for
  example `iib|iia` or `iib|iia|i|iix`.
- `classification_method`: classifier/rule source.
- `prob_iib`, `prob_iia`, `prob_iix`: model class probabilities when available.
- `model_confidence`: highest model probability.
- `model_margin`: gap between the highest and second-highest probabilities.
- `needs_review`: `True` when confidence or margin thresholds flag the fiber.
- `typing_signal_qc_flags`: signal/model consistency flags separated by `|`.
- `classifier_path`: classifier file used for the run.

## Optional Diagnostics Table

When `--export-diagnostics` is enabled, FiberTypeQC also writes `*_feature_diagnostics.csv`.

This file is intended for model-development/debugging work, not for stable biological reporting.
It may include:

- stable metadata columns such as `label`, `fiber_type`, `fiber_type_source`, and `needs_review`;
- the frozen alpha baseline model features;
- experimental coverage/SNR/extra-marker features used for diagnostics.

This optional file does not change classifier predictions or expand the stable `*_fibers.csv`
schema by default.

## Review Table Columns

The Napari review UI writes:

- `fiber_id`: fiber label ID.
- `corrected_type`: manual label, normalized during merge.
- `is_uncertain`: reviewer marked uncertain.
- `is_hybrid`: reviewer marked hybrid.
- `is_excluded`: reviewer excluded the fiber.
- `label_source`: `manual_gold` for manually corrected labels.

## Merged Review Columns

`scripts.merge_reviewed_labels` preserves the original fiber columns and adds:

- `fiber_id`: integer fiber ID used for merge.
- `predicted_type`: original automatic prediction.
- `predicted_internal_type`: internal prediction label.
- `predicted_biological_type`: display/biological label mapping.
- `final_type`: prediction replaced by manual correction, uncertainty, hybrid, or exclusion status.

## Batch Summary Columns

`batch_summary.csv` includes:

- `image_name`: image stem.
- `status`: `success` or `failed`.
- `error`: failure message when applicable.
- `fiber_count`: number of segmented fibers when successful.
- `summary_path`: per-image summary CSV path.
