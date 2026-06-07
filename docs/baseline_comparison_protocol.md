# Baseline Comparison Protocol

This document defines how candidate typing models should be compared against the frozen
FiberTypeQC alpha baseline before any default-model decision is made.

The goal is not to optimize one metric in isolation. The goal is to compare candidate behavior
against the current frozen default in a way that is reproducible, conservative, and interpretable.

## Scope

This protocol applies to candidate-model evaluation for `v0.3` and later work.

Current frozen baseline:

- model: `data/models/rebaseline_tile_v2_p75p90_iib_iia_iix.joblib`
- default panel:
  - membrane
  - direct IIb
  - direct IIa
  - residual inferred IIx
- default output contract:
  - stable `*_fibers.csv`
  - optional `*_feature_diagnostics.csv` only when explicitly requested

This protocol does not itself change:

- classifier defaults
- thresholds
- channel mappings
- preprocessing defaults
- erosion defaults
- QC flag logic
- review merge behavior

## Baseline Policy

The frozen alpha baseline is the comparator, not the candidate.

Candidate work must answer:

1. Does the candidate improve behavior on declared evaluation data?
2. Does the candidate preserve or improve interpretability?
3. Does the candidate reduce review burden without introducing obvious biological drift?
4. Are tradeoffs explicit if some metrics improve while others worsen?

No candidate should become the default because it "looks better" on one image or one cohort.

## Comparison Targets

Every candidate should be compared against the frozen baseline on the same dataset split.

Minimum comparison targets:

1. frozen baseline feature contract / current default model
2. candidate feature set
3. candidate model artifact
4. held-out evaluation outputs

Minimum artifacts to retain:

- candidate feature-generation command
- candidate training command
- candidate model artifact path
- held-out evaluation summary
- notes on known tradeoffs or failure modes

## Required Dataset Split

At minimum, keep three buckets distinct:

1. training set
2. development/validation set
3. held-out evaluation set

Rules:

- held-out images must not be used to tune candidate features or thresholds
- if manual labels exist, keep image-level separation strict
- do not mix fibers from the same image across train and held-out buckets
- if protocol-specific panels are mixed, record panel type explicitly

Recommended split metadata:

- image ID
- cohort / experiment group
- panel definition
- source of manual labels, if any
- whether image was previously used for baseline development

Use the seeded manifest generator to pre-populate filename-derived biology and baseline-derived
technical columns before manual curation:

```bash
uv run python -m validation.build_candidate_split_manifest \
  --output outputs/validation/candidate_split_manifest.csv
```

The generated manifest is intended to be edited by hand for:

- final `split`
- manual quality notes
- saturation/difficulty notes
- any explicit inclusion/exclusion decisions

## Evaluation Outputs

Each candidate comparison should produce these outputs.

### 1. Feature-contract comparison

Use:

```bash
uv run python -m validation.compare_feature_sets
```

Purpose:

- confirm what the frozen baseline actually uses
- confirm which candidate features are experimental additions
- document whether candidate features include extra markers, coverage terms, or SNR terms

### 2. Held-out image summary

At minimum, summarize by image:

- total fibers
- per-class counts
- per-class proportions
- `needs_review` rate
- QC flag counts
- confidence / margin summaries

If manual labels exist, also summarize:

- per-class precision/recall or equivalent
- confusion matrix counts
- mismatch counts by image

### 3. Review-burden summary

Report:

- review rate (`needs_review / total`)
- flagged-fiber composition by predicted class
- confidence or margin distribution for correct vs incorrect calls where labels exist

### 4. Biological story check

For candidate models evaluated on the JAG1/MyoSight-style comparison data, check whether the
candidate changes the biological interpretation relative to:

- the frozen FiberTypeQC baseline
- the MyoSight comparator summaries

This is not a requirement that the candidate exactly match MyoSight. It is a requirement that any
meaningful biological shift be noticed, documented, and justified.

## Minimum Acceptance Criteria for a Candidate

Before a candidate can be considered for default use, it should satisfy all of the following:

1. Reproducibility
   - training/evaluation commands are recorded
   - model artifact is versioned
   - feature set is explicit

2. Baseline comparison completeness
   - frozen baseline results are reported side by side
   - candidate-vs-baseline differences are summarized by image and by class

3. No silent default drift
   - the frozen baseline tests still pass unchanged
   - the candidate runs through a separate path or explicit selection mechanism

4. Utility, not just novelty
   - candidate either improves held-out behavior, reduces review burden, improves calibration, or
     provides another clearly justified benefit

5. Explicit tradeoff accounting
   - if one metric improves while another worsens, the change is called out directly

## Suggested Candidate Comparison Sequence

Follow this sequence for each candidate iteration.

1. Freeze the candidate feature table definition.
2. Generate candidate feature tables on the declared split.
3. Train and version a candidate artifact.
4. Run held-out evaluation.
5. Compare against the frozen baseline.
6. Write a short comparison note:
   - what changed
   - what improved
   - what worsened
   - whether the candidate should advance

## Recommended Commands

These are the current repo-level commands already available for baseline-aware work.

Feature contract comparison:

```bash
uv run python -m validation.compare_feature_sets
```

Fast tests:

```bash
uv run python -m pytest -m "not integration"
```

Optional integration tests:

```bash
uv run python -m pytest -m integration
```

If diagnostics are needed for feature inspection:

```bash
uv run python -m scripts.run_pipeline \
  --image path/to/image.czi \
  --output-dir outputs/candidate_debug \
  --export-diagnostics
```

The diagnostics export is for model-development/debugging only. It does not replace the stable
fiber table and should not be treated as the routine biological output contract.

To assemble per-image diagnostics into one candidate-model feature table:

```bash
uv run python -m validation.build_candidate_feature_table \
  --input-root outputs/candidate_debug \
  --output outputs/candidate_feature_table.csv
```

If you have image-level split metadata:

```bash
uv run python -m validation.build_candidate_feature_table \
  --input-root outputs/candidate_debug \
  --manifest outputs/candidate_split_manifest.csv \
  --output outputs/candidate_feature_table.csv
```

## What This Protocol Does Not Require

This protocol does not require:

- full per-fiber ROI agreement with MyoSight
- immediate support for every possible panel
- immediate promotion of experimental features into the stable fibers CSV
- changing the default model in `v0.3`

It does require that candidate-model work be explicit, reproducible, and judged against the frozen
baseline rather than intuition alone.

## Relationship to MyoSight

MyoSight remains a historical comparator and lab-process reference point.

It is useful for:

- image-level trend comparison
- biological story checks
- sanity checks during validation

It is not treated as literal per-fiber ground truth in this protocol.

Where FiberTypeQC and MyoSight disagree, the comparison should separate:

- segmentation/measurement-definition differences
- panel/stain limitations
- model uncertainty
- likely biological disagreement

## Decision Rule for Default-Model Changes

Default-model changes are a separate release decision.

A candidate should not replace the frozen alpha baseline unless:

- the candidate artifact is versioned
- held-out comparison outputs are saved
- tradeoffs are documented
- release notes/changelog explicitly state the scientific behavior change

Until then:

- the frozen alpha model remains the public default
- candidate models remain opt-in evaluation artifacts
