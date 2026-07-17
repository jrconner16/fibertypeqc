# FiberTypeQC Roadmap (2026 H3)

Planning horizon: repository stabilization followed by the multi-panel skeletal-muscle expansion.  
Execution model: gate-based rather than date-driven. Feature implementation does not begin until
the clean-baseline gate is complete.  
Priority order: reproducible public workflow, scientifically defensible outputs, reusable analysis
stages, then external usability and broader model adaptation.

## Progress

- [x] Phase 0 — clean repository baseline
- [x] Phase 1 — architecture and contract specification
- [x] Phase 2 — panel and restartable artifact foundation
- [ ] Phase 3 — Type I and semantic myosin features (in progress)
- [ ] Phase 4 — independent eMHC regeneration domain
- [ ] Phase 5 — DAPI and nuclear association
- [ ] Phase 6 — project-specific adaptation
- [ ] Phase 7 — external usability pilot

## Objective

Establish a clean and reproducible repository baseline, then expand FiberTypeQC into a standardized
skeletal-muscle fiber analysis workflow that supports common four-channel immunofluorescence panels.

The analysis engine should generalize across supported panels. Classifiers may remain panel-specific
or be adapted to a project from reviewed labels. The existing frozen classifier and public workflow
must remain available and unchanged until a replacement independently satisfies its promotion gates.

The stable public workflow remains:

```text
pipeline -> review -> merge
```

The expanded workflow introduces restartable stages beneath that surface:

```text
panel config
  -> fiber segmentation cache
  -> optional nuclear segmentation cache
  -> semantic feature extraction
  -> task-specific classification
  -> review
  -> optional project adaptation
  -> grouped validation
  -> explicit model selection
```

## Guiding Decisions

1. Support a fixed skeletal-muscle marker vocabulary, not arbitrary microscopy markers:
   `laminin`, `dapi`, `type_i`, `type_iia`, `type_iib`, `type_iix`, and `emhc`.
2. Permit no more than four active image channels. A biological class may additionally be inferred
   from marker absence only when the panel and model manifest explicitly allow that inference.
3. Keep observed signals, derived biological calls, and review/provenance metadata separate.
4. Do not force all measurements into one multiclass label. Fiber identity, regeneration, nuclear
   pathology, and geometry are parallel output domains.
5. Cache segmentation independently from classification. Model changes must not rerun Cellpose.
6. Treat starter models as review accelerators, not universal classifiers.
7. Require grouped validation by image, mouse, batch, or experiment; never use a random fiber split
   when related fibers can cross the train/test boundary.
8. Never silently replace a released model with an adapted model.
9. Use cautious DAPI terminology. DAPI objects are nuclei, not automatically myonuclei.
10. Preserve the frozen baseline as the explicit comparator throughout the expansion.

## Current Baseline and Entry Gate

The current public implementation is split across:

- `scripts/`: stable command wrappers.
- `src/`: core implementation plus experimental model-development and validation code.
- `fibertypeqc/`: public API facade, currently delegating substantially to `src/`.
- `validation/`: optional candidate-model, MyoSight, audit, and comparison commands.
- `data/models/`: released model artifacts and model documentation.
- `outputs/`, local images, labels, notebooks, and most of `data/`: private or generated workspace.

At roadmap creation, the repository is not yet at the clean-baseline gate. Known work includes pending
Jag1 analysis/reporting changes, Feret/backfill and candidate-batch work, manifests under generated
output paths, inconsistent version identifiers, and Ruff findings. The fast non-integration test path
passes, but both tests and lint must be green on the final cleanup commit.

## Phase 0 — Clean Repository Baseline

No major feature implementation is allowed during this phase.

### Work

1. Commit the pending Jag1 analysis, report, manifests, and tests as one reviewable unit.
2. Separate independent Feret/backfill and candidate-batch changes into their own commits.
3. Move tracked validation and holdout manifests out of `outputs/` into `manifests/` or a documented
   validation-manifest directory; update every reference and test.
4. Reconcile the README, package metadata, runtime version, changelog, and roadmap around one current
   development version.
5. Resolve all Ruff findings without intentionally changing scientific behavior.
6. Add `ARCHITECTURE.md` documenting:
   - stable public entry points and the `pipeline -> review -> merge` contract;
   - production versus experimental modules;
   - data, manifest, model, and generated-artifact boundaries;
   - the current runnable-research-application package identity;
   - the files appropriate for future GPT/Codex planning handoffs.
7. Document experimental modules in place. Move them under an `experiments/` namespace only through
   later, incremental changes with import and CLI compatibility tests.
8. After all work is committed or safely stashed, inspect the reported temporary Git pack artifact,
   perform safe Git maintenance, and verify repository integrity.

### Required verification

```bash
uv run python -m pytest -m "not integration" -q
uv run ruff check .
git status --short
git fsck --full
```

### Exit criteria

- Working tree is clean.
- Pending work is represented by understandable, scoped commits.
- No tracked manifests live under generated output paths.
- User-facing and package version identifiers agree.
- Fast tests and Ruff pass.
- `ARCHITECTURE.md` exists and identifies stable and experimental areas.
- Git integrity passes after safe maintenance.
- A tagged commit or recorded baseline commit SHA is selected for feature comparisons.

## Phase 1 — Architecture and Contract Specification

This is the planning and contract phase. It precedes behavior changes.

### Deliverables

1. A feature architecture plan identifying:
   - proposed modules and responsibilities;
   - reusable existing functions;
   - experimental modules eligible for promotion;
   - schema and CLI changes;
   - compatibility risks;
   - artifact-directory structure;
   - migration behavior for old outputs and models;
   - the smallest end-to-end implementation slice.
2. A versioned panel-schema specification for the fixed marker vocabulary.
3. A versioned feature-schema specification using semantic names such as
   `type_i.p90`, `type_i.coverage_high`, and `emhc.center_mean`.
4. A model-manifest specification declaring:
   - compatible panels;
   - required feature-schema version;
   - target task and output labels;
   - permitted residual inference;
   - training-data provenance;
   - software/model versions.
5. An output-contract proposal separating geometry, fiber identity, regeneration, nuclear pathology,
   and provenance.
6. A compatibility plan showing how the current TA workflow runs unchanged.

### Design gate

The specifications must answer these questions before implementation begins:

- What can be measured directly from this panel?
- What may be inferred, and under which explicit policy?
- Which outputs must be suppressed or marked unresolved?
- Which cached artifacts can be reused after a config, classifier, or association-rule change?
- How will an old frozen-baseline run remain reproducible?

## Phase 2 — Panel and Restartable Artifact Foundation

### Status: complete (2026-07-16)

Implementation is recorded in `b8b5417`, `c888ad4`, and `8138209`. The frozen baseline was
validated on two local one-month images: labels and every shared numeric fiber-table value matched
the trusted outputs exactly. The only added fiber columns were the planned Feret measurements.

### Compute posture

Use CUDA-enabled Cellpose for HPC segmentation and MPS-enabled Cellpose for local macOS runs.
Keep feature extraction and scikit-learn inference CPU-based, optimize and profile their algorithms,
and parallelize independent images through the scheduler. A GPU feature-extraction backend is
deferred unless a post-optimization profile demonstrates a material throughput need.

### Scope

- Implement the fixed marker vocabulary and semantic channel mapping.
- Enforce no more than four active channels.
- Reject duplicate channel indices and indices absent from the source image.
- Require laminin when fiber segmentation is requested.
- Validate requested outputs against available markers and model compatibility.
- Introduce feature-schema and model-manifest validation.
- Define stable, separately cached stage artifacts.
- Preserve current CLI behavior and frozen defaults when new configuration is not used.

### Initial artifact contract

```text
<run>/<image_id>/
  run.json
  fiber_labels.tif
  nuclei_labels.tif                 # only when DAPI segmentation is requested
  fibers.csv
  nuclei.csv                        # optional
  nucleus_fiber_links.csv           # optional
  diagnostics/
```

Names may be refined during specification, but the following invalidation rules are required:

| Change | Rerun fiber segmentation | Rerun nuclear segmentation | Recompute links/features |
|---|---:|---:|---:|
| Classifier or threshold | No | No | Classification only |
| Marker feature recipe | No | No | Yes |
| Nucleus-association rule | No | No | Links and summaries only |
| Fiber Cellpose parameters | Yes | No | Yes |
| Nuclear Cellpose parameters | No | Yes | Yes |

### Acceptance criteria

- Valid and invalid panel fixtures cover every schema rule.
- A legacy TA run produces the expected frozen outputs under existing defaults.
- A classifier cannot run with missing required channels or the wrong feature schema.
- Run metadata records Git commit, FiberTypeQC/Python/Cellpose/PyTorch versions, device, panel,
  segmentation parameters, preprocessing parameters, model manifest, and source-image identifier.
- One representative full image completes with restartable artifacts.

## Phase 3 — Type I and Semantic Myosin Features

### Status: in progress

The first slice remains diagnostics-only: semantic observed-marker features may be added without
changing legacy fiber calls, residual inference, or the stable `*_fibers.csv` schema.

### Slice checklist

- [x] Versioned semantic mean/percentile/coverage/SNR diagnostics for observed markers.
- [x] Center/edge semantic diagnostics, emitted only with optional diagnostics export.
- [x] Type I manual-review support (display and labels only; no automatic Type I calls).
- [x] Type I-compatible curated-label inventory and panel audit (see
  `docs/type_i_panel_audit.md`).
- [x] Development-only QUAD smoke experiment with a manual-IIa evidence gate;
  one-image fiber-level cross-validation only, not a generalization result.
- [ ] Type I candidate model and manifest; no automatic model selection.
- [ ] Held-out TA false-positive and quadriceps pilot evaluation.

### Data roles

- Current TA/Jag1 cohort: Type I-negative distribution, false-positive testing, backward compatibility.
- Quadriceps pilot: Type I-positive fibers and large-image processing.
- External-laboratory images: cross-laboratory panel behavior after manual labels exist.

### Channel-verification record

- Jag1 quadriceps images: manually verified panel mapping is Type I = channel 0,
  IIa = channel 1, laminin = channel 2, and IIb = channel 3. The prior Jag
  regeneration analysis configured only IIa/laminin/IIb, so it did not quantify
  the available Type I channel.
- Vivienne images: require separate manual channel verification before they are
  included in any panel inventory, training set, or validation cohort.

### Scope

- Apply a consistent semantic feature family to available myosin markers:
  mean, median, upper percentiles, high-intensity coverage, center/edge signal, signal-to-background,
  heterogeneity, and biologically appropriate channel contrasts.
- Add Type I fields without changing the meaning of historical `IIa`, `IIb`, and inferred `IIx`.
- Support Type I-aware panels and explicit residual-class policies.
- Assemble training data from compatible curated labels only.
- Produce a starter Type I-capable candidate and manifest.
- Complete at least one full quadriceps image locally or on E3.

### Scientific gates

- Held-out TA images show an acceptably low Type I false-positive rate.
- Quadriceps validation holds out whole images and reports the pilot sample-size limitation.
- Historical residual/IIx labels are not reused as Type I negatives without a panel-compatibility audit.
- The workflow does not claim `IIb` versus `IIx` separation for panels that lack supporting evidence.

## Phase 4 — Independent eMHC Regeneration Domain

### Scope

- Implement semantic eMHC marker features.
- Model eMHC as an independent binary/probabilistic task:
  positive, negative, uncertain, score/probability, and coverage.
- Allow a fiber to have both a fiber-type label and an eMHC status.
- Add eMHC review context to Napari.
- Add image- and mouse-aware validation reports.

### Acceptance criteria

- eMHC is never inserted as an exclusive fiber-type class.
- Outputs retain both identity and regeneration domains.
- Model and threshold provenance are recorded.
- Validation for the DAPI/eMHC cohort is grouped by mouse.
- Missing eMHC channels suppress the domain cleanly rather than creating inferred values.

## Phase 5 — Core DAPI and Nuclear Association

### Segmentation sequence

```text
load laminin -> segment fibers -> save fiber labels -> release memory
load DAPI -> segment nuclei -> save nuclear labels -> release memory
load cached labels -> associate nuclei -> calculate features and summaries
```

Cellpose remains the segmentation engine for both passes. The new work is integration,
postprocessing, association, output contracts, and review—not training a new segmentation network.

### Initial features

- Nuclear object label, centroid, area, and filtering status.
- Fiber overlap and unambiguous fiber assignment.
- Distance to fiber boundary and normalized radial position.
- `central_interior`, `boundary_associated`, `peripheral_associated`, and
  `unassigned_or_interstitial` categories.
- Associated and central nuclei counts per fiber.
- Centrally nucleated fiber status.
- Whole-image nuclear density and assignment summaries.
- Fiber, nuclear, and association overlays in Napari.

### Scientific boundaries

- Use “fiber-associated nucleus,” not “myonucleus,” unless additional evidence justifies it.
- Do not classify inflammatory, endothelial, fibroblast, or satellite cells from DAPI alone.
- Make ambiguous assignment explicit rather than forcing one nucleus to one fiber.
- Version centrality and association rules independently from segmentation.

### Acceptance criteria

- Fiber and nuclear segmentation can be resumed independently from cache.
- Changing association thresholds does not rerun Cellpose.
- Synthetic geometry tests cover central, peripheral, boundary, ambiguous, and unassigned nuclei.
- A full DAPI/eMHC section completes without retaining both segmentation images in GPU memory.
- Review overlays make segmentation and assignment failures inspectable.

## Phase 6 — Supported Project-Specific Adaptation

Promote only the necessary, tested parts of the existing experimental candidate-model machinery.

### Supported workflow

```text
starter model
  -> stratified review pool
  -> reviewed labels
  -> candidate project model
  -> grouped validation
  -> starter-versus-candidate report
  -> explicit user acceptance
  -> explicit candidate selection for later runs
```

### Review-pool requirements

Include a controlled mixture of:

- low-confidence and boundary cases;
- random fibers;
- high-confidence examples from every predicted class;
- rare classes and artifacts;
- images/animals across the dataset;
- diverse intensity and staining ranges.

Uncertainty-only sampling is not sufficient because a shifted model may be confidently wrong.

### Model requirements

- Manually reviewed labels are trusted.
- Existing labels are reused only when panel and biological definitions are compatible.
- Unreviewed predictions are not training labels by default.
- Grouping supports image, mouse, staining batch, and experiment.
- Every candidate produces a model artifact, manifest, comparison report, limitations, and exact
  training/evaluation split.
- Candidate creation never changes the released default or reclassifies prior results automatically.

### Acceptance criteria

- Image-held-out adaptation works for quadriceps and external panels.
- Mouse-held-out adaptation works for the eMHC study.
- Leakage checks fail loudly when a group crosses train and evaluation sets.
- Base-versus-adapted reports include per-class/task metrics, group-level summaries, sample sizes,
  missing-class warnings, and known limitations.
- Selecting an adapted model requires an explicit CLI/config value.

## Phase 7 — External Usability Pilot

Apply the supported workflow to approximately ten external-laboratory images after their panels are
confirmed and a review set is labeled.

Evaluate:

- whether channel configuration is understandable;
- whether unsupported output requests fail clearly;
- how much labeling is required before adaptation is useful;
- whether adaptation improves held-out images;
- whether a user other than the primary developer can complete setup, review, training, and rerun;
- documentation, CLI, Napari, and artifact-management friction.

This pilot is evidence about usability and adaptation, not proof of universal cross-laboratory
generalization.

## Local and E3 Execution Contract

### Local development subset

Use a deliberately small set:

- one typical TA image;
- one difficult/noisy TA image;
- one Type I-positive quadriceps image;
- one DAPI/eMHC section;
- one external-laboratory image when available.

Use fixed representative crops for rapid iteration, followed by at least one full-image run before a
stage is declared stable.

### E3 production sequence

1. Commit the exact code and record the configuration.
2. Pull that commit on E3.
3. Run one GPU smoke-test image and confirm CUDA execution.
4. Compare object counts, boundary quality, nuclear counts, major artifacts, and downstream feature
   stability; exact MPS/CUDA pixel equality is not required.
5. Run the remaining cohort on GPU nodes.
6. Review full-resolution results in Napari on the E3 interactive desktop.

Keep full raw cohorts on E3 whenever possible. Transfer code and configuration through Git and copy a
small representative local subset only once.

## Testing and Validation Strategy

| Layer | Required evidence |
|---|---|
| Panel/config | Unit tests for vocabulary, channel count, uniqueness, bounds, required markers, and model compatibility |
| Artifact cache | Invalidation/resume tests and run-manifest provenance tests |
| Legacy workflow | Frozen snapshot/regression tests for default TA behavior |
| Marker features | Synthetic masks and intensities with semantic-schema snapshots |
| Nuclear association | Synthetic geometry plus reviewed full-image overlays |
| Classification | Group-held-out task metrics and comparison with the frozen/starter model |
| CLI | Smoke tests for legacy and panel-aware commands, including clear failure cases |
| Scale/device | One full large image and one E3 CUDA smoke test per segmentation milestone |
| Review | Napari load/save round trip for all enabled output domains |

Any change to the frozen model, feature calculations used by frozen behavior, class order, channel
defaults, erosion, thresholds, review flags, or merge logic remains baseline-sensitive and requires a
versioned comparison.

## Release and Migration Policy

- Existing frozen outputs remain readable and mergeable.
- New fields are introduced through a documented schema version.
- New optional domains do not silently alter the stable legacy fibers CSV until a versioned migration
  decision is made.
- Old models without manifests are treated through an explicit legacy compatibility adapter.
- New models fail clearly when panel or feature requirements are unmet.
- Cache reuse requires compatible stage versions and recorded parameters.
- No migration rewrites raw images or old output directories in place.

## Major Milestone Dependency Map

```text
Phase 0 clean baseline
  -> Phase 1 contracts
      -> Phase 2 panel/artifact foundation
          -> Phase 3 Type I -----------+
          -> Phase 4 eMHC ---------+    |
          -> Phase 5 DAPI ---------+----+-> Phase 7 external pilot
          -> Phase 6 adaptation ---+
```

Phase 6 may begin with existing myosin features after Phase 2. Its eMHC and nuclear variants depend on
the relevant later phases.

## Initial End-to-End Definition of Success

The first complete major-feature milestone is achieved when:

1. A four-channel image can be configured using only the supported marker vocabulary.
2. Laminin and DAPI are segmented sequentially and cached separately.
3. Type I and eMHC features are quantified when their markers are present.
4. Fiber identity, regeneration, geometry, and nuclear pathology are represented as separate domains.
5. Fiber, nuclear, and association outputs can be reviewed in Napari.
6. Reviewed labels can train a candidate project-specific model.
7. The candidate is evaluated on held-out images or mice with no grouping leakage.
8. The candidate has a manifest and comparison report and must be selected explicitly.
9. The original frozen classifier and legacy workflow remain available and unchanged.

## Explicitly Deferred

- Arbitrary antibodies or unrestricted marker names.
- Automatic interpretation of unknown channels.
- A generic microscopy machine-learning platform.
- Automatic retraining after every Napari edit.
- Automatic promotion or replacement of released models.
- Deep-learning classifier fine-tuning.
- Unreviewed pseudo-label training by default.
- Claims that one classifier generalizes across all laboratories.
- Myonuclear, inflammatory, endothelial, or fibroblast classification from DAPI alone.
- Pax7 and satellite-cell analysis.
- Advanced nuclear morphology models.
- Pooled cross-laboratory model development.
- Broad installable-package restructuring without a demonstrated need.

## Issue and PR Operating Model

- Create one issue per acceptance-testable slice and assign exactly one roadmap phase.
- Use one branch per issue: `feat/<issue-id>-short-name`, `fix/<issue-id>-short-name`, or
  `chore/<issue-id>-short-name`.
- Keep cleanup commits separate from scientific feature commits.
- Label scientific-behavior changes `baseline-sensitive` and attach before/after evidence.
- Every feature PR reports files changed, tests run, schema impact, cache invalidation impact, and
  whether frozen behavior changed.
- Record blockers explicitly; do not bypass a phase gate to keep implementation moving.

Before closing a phase, confirm its acceptance criteria, documentation, changelog/schema notes,
reproduction commands, and unresolved work moved deliberately to a later phase.
