# Validation Summary

FiberTypeQC v0.1-alpha includes validation utilities for image-level comparison against MyoSight
exports.

## Included Tools

- `validation.compare_myosight_pipeline`: compare image-level MyoSight and FiberTypeQC summaries.
- `validation.plot_validation_summary`: generate summary plots for validation slide decks.
- `validation.plot_confidence_diagnostics`: inspect confidence, margin, and review flags.
- `validation.compare_roi_boundaries`: compare ROI boundaries on selected examples.
- `validation.sweep_measurement_mask_erosion`: test how typing erosion changes classification.

## Validation Position

The alpha validation target is group-level and image-level agreement with an existing lab workflow.
Exact per-fiber ROI matching is a stronger future validation step.

## Measurement Definition

MyoSight and Cellpose masks may define fiber boundaries differently. FiberTypeQC can report raw and
eroded area measurements so comparisons can separate biological disagreement from measurement
definition differences.

## Interpreting Differences

Differences between methods can come from:

- channel order or marker mismatch,
- staining/background differences,
- segmentation mask definition,
- IIx being inferred from absent IIb/IIa signal,
- classifier uncertainty,
- MyoSight/manual thresholding choices.

Validation figures should show both hard class proportions and review/confidence diagnostics.
