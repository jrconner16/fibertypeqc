# Type I Panel and Curated-Label Audit

## Decision record

Phase 3 began with a panel/label audit before any Type I model work. This is an
eligibility record, not a classifier specification and not evidence of model
performance.

| Cohort or source | Panel status | Curated-label status | Type I training eligibility |
| --- | --- | --- | --- |
| Historical TA baseline | No direct Type I channel | Reviewed labels use the legacy IIa/IIb/residual-IIx policy | Excluded |
| Jag1 regeneration | Type I channel was present but omitted from the prior analysis configuration | No registered Type I-reviewed labels | Excluded until labels are reviewed |
| Jag1 quadriceps | Manually verified: Type I=0, IIa=1, laminin=2, IIb=3 | No registered Type I-reviewed labels | Panel-compatible; labels still required |
| Vivienne | Manual channel verification pending | Not audited | Excluded pending verification and labels |

The historical residual/IIx call must never be relabeled as a Type I negative
solely because it is not IIa or IIb. Likewise, an image with a Type I channel
is not training-eligible until fibers are manually reviewed for the requested
Type I task.

## Admission requirements

A source can enter a Type I candidate training or evaluation manifest only when
all of the following are recorded outside the repository with its private data:

1. Manual verification of the semantic channel mapping for that acquisition
   cohort.
2. A reviewed fiber-label table produced through the supported workflow:
   `run_pipeline` or `run_batch` -> `review_labels_napari` ->
   `merge_reviewed_labels`.
3. Explicit Type I-positive labels and task-appropriate non-Type-I labels;
   legacy residual labels do not satisfy the latter requirement by themselves.
4. Image-level source grouping and an intended `train`, development, or
   held-out role, so scenes from one biological source cannot leak across an
   evaluation split.
5. A record of exclusions (uncertain fibers, artifact, missing channels, or
   unverified stains).

## Next data-collection action

Create a small, manually reviewed Jag1 quadriceps label set before training.
Reserve at least one whole image or biological source for the pilot evaluation;
do not use it to choose features or thresholds. Keep raw images, review CSVs,
and any derived outputs outside version control.

## Consequence for Phase 3

No Type I candidate model or model manifest may be created from the current
legacy review inventory. The next Phase 3 slice starts only after compatible
quadriceps reviewed labels are available.
