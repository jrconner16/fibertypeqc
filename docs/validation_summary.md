# Validation Summary

FiberTypeQC `v0.2.0` is the current published release. It ships a conservative frozen public
baseline together with validation utilities for comparing FiberTypeQC outputs against historical
MyoSight workflow summaries and manually reviewed subsets.

This document summarizes the current public validation position and the standing experimental
candidate state for ongoing `v0.3` work.

## Published Baseline vs Experimental Candidate

### Frozen public baseline

The current public default remains the frozen baseline model:

- `data/models/rebaseline_tile_v2_p75p90_iib_iia_iix.joblib`

This baseline is the supported public typing path for `v0.2.0`. It remains the reference point for:

- release comparisons,
- regression checks,
- CLI/default behavior,
- documentation examples.

### Standing experimental candidate

The current leading experimental candidate for `v0.3` evaluation is:

- `manual8_myo3_base0p1`

This candidate is being evaluated as a **candidate-pipeline**, not as a public default and not as a
release-ready replacement for the frozen baseline.

## Current Validation Position

FiberTypeQC is evaluated through three complementary lenses:

1. protected manually reviewed holdout data;
2. image-level and group-level comparison against the historical MyoSight workflow;
3. targeted review of disagreement and uncertainty patterns.

Current interpretation remains conservative:

- the frozen public baseline remains the supported public release path;
- the standing experimental candidate appears meaningfully stronger than the frozen baseline on the
  current protected manual benchmark;
- the experimental candidate also appears closer to the historical MyoSight biological story at the
  cohort level;
- remaining disagreement is concentrated rather than uniform, with the `IIb`/`IIx` boundary still
  driving most of the important residual error.

FiberTypeQC should therefore be presented as a reproducible, review-assisted workflow with an
improving experimental candidate path, not as a final replacement for MyoSight.

## Protected Manual Holdout Status

The current `v0.3` candidate benchmark is based on a protected manually reviewed holdout derived
from the broader reviewed benchmark set.

High-level standing result for the leading experimental candidate:

- protected holdout accuracy: about `0.814`
- protected holdout balanced accuracy: about `0.847`

These results are stronger than the older frozen and earlier candidate references, but they are
still part of an internal candidate-evaluation process rather than a finalized promotion decision.

## MyoSight Comparison Status

The current repaired full-cohort comparison suggests that the standing experimental candidate
broadly matches the same biological story seen in the historical MyoSight workflow better than the
frozen public baseline.

At a high level:

- `IIa` behavior is much improved relative to the frozen baseline;
- the older pattern of excessive `IIx` inflation has been reduced;
- `IIb` recovery is improved;
- remaining mismatch is concentrated in a smaller set of `IIb`/`IIx` boundary groups and hotspot
  images.

This is encouraging, but it is still a descriptive comparison rather than a claim of final
methodological equivalence.

## Main Remaining Validation Issue

The main remaining issue is the `IIb`/`IIx` boundary.

In the current panel setup, `IIx` is inferred as the residual unstained class relative to direct
IIb and IIa evidence. That makes the `IIb`/`IIx` boundary especially sensitive to:

- weak or patchy positive marker signal,
- slide/background variation,
- historical thresholding differences,
- classifier uncertainty near the residual class boundary.

Current validation work indicates that most biologically important remaining disagreement is
concentrated here rather than in broad failure across all classes.

## Review and Triage Status

The older global `needs_review` framing from the frozen baseline does not transfer cleanly to the
current experimental candidate.

Current review-policy work therefore remains experimental. The most promising direction so far is:

- conservative handling of direct `IIa` through gating,
- broader trust in routine predicted `IIb`,
- targeted ranking of risky inferred `IIx` fibers for focused review.

A targeted `IIx` review ranker has shown promising internal validation results, but it should still
be treated as an experimental triage aid rather than a release-stable public review policy.

## Included Validation Tools

- `validation.compare_myosight_pipeline`: compare image-level MyoSight and FiberTypeQC summaries.
- `validation.plot_validation_summary`: generate summary plots for validation slide decks.
- `validation.plot_confidence_diagnostics`: inspect confidence, margin, and review flags.
- `validation.compare_roi_boundaries`: compare ROI boundaries on selected examples.
- `validation.sweep_measurement_mask_erosion`: test how typing erosion changes classification.

## Measurement Definition Caveat

MyoSight and Cellpose-derived masks may define fiber boundaries differently. FiberTypeQC can report
raw and eroded area measurements so that biological disagreement can be separated from measurement
definition differences where possible.

Typing conclusions should be interpreted only after confirming that segmentation provenance and
count comparisons are on comparable footing.

## Summary

As of `v0.2.0` and the current `v0.3` candidate evaluation state:

- the frozen baseline remains the public default;
- `manual8_myo3_base0p1` is the standing experimental candidate-pipeline;
- protected holdout and cohort-level MyoSight comparison both favor the experimental candidate over
  older references;
- the `IIb`/`IIx` boundary remains the main unresolved validation issue;
- targeted `IIx` review ranking is promising, but still experimental.
