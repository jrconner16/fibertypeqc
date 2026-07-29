# QC and Manual Review System Plan

Status: Phase 1 complete; repository audit and shared headless foundation implemented.

This document is the controlling product and implementation plan for turning
FiberTypeQC into a project-based, human-in-the-loop review system. It records the
requested product specification, the repository audit performed on
`feature/napari-review-workflow`, and every meaningful design adjustment discovered
during that audit.

## 1. Product goal

Build FiberTypeQC into one coherent human-in-the-loop image-analysis system that
combines:

1. reproducible CLI/HPC prediction;
2. automated technical QC and section triage;
3. progressive manual review at image, region, and object scale;
4. final, analysis-ready reviewed outputs; and
5. structured corrections that can later support model retraining.

The user begins at cohort or mouse level and descends into an image, region, or
object only when QC or the selected review mode requires it:

```text
CLI predictions
  -> precomputed QC
  -> cohort/mouse dashboard
  -> review-intensity choice
  -> section selection
  -> image/region/object review
  -> finalization and reports
```

The target experience is to run FiberTypeQC predictions through CLI/HPC, generate
QC, open one verified project, understand which mice and domains need attention,
choose a review mode and section strategy, make traceable corrections, resume
after closing, and finalize without overwriting any prediction.

The central rule is:

> Scope says where the problem is. Domain says what is wrong.

Technical QC is decision support, not biological ground truth. Unusual biology
must not be silently rejected.

## 2. User workflow

### 2.1 Prediction and project creation

1. Run the existing public prediction workflow:
   `src/run_pipeline.py` or `src/run_batch.py`.
2. Create or load a YAML project manifest that points to existing raw images and
   prediction directories; do not copy large prediction artifacts.
3. If the manifest is inferred from an output directory, present inferred
   mouse/section metadata for explicit user verification.
4. Generate versioned QC tables before launching the interactive application.

### 2.2 Cohort and section triage

The Cohort Dashboard shows:

- mouse and section counts;
- per-domain PASS / REVIEW / FAIL counts;
- mice with at least one acceptable section;
- mice needing targeted review;
- mice with no acceptable section;
- review progress and correction burden; and
- each section's domain-specific status and explicit QC reasons.

Selection is domain-specific. A section may be usable for fiber morphology but
not nuclei. Fiber-nucleus analyses require all relevant domains to pass.

Strategies:

- `all_passing`: aggregate all technically passing sections;
- `best_passing`: select the highest-priority technically passing section; and
- `manual`: let the reviewer choose.

If no section passes for a mouse/domain, route that mouse/domain to manual review.

### 2.3 Progressive review

The current image, mouse, section, scope, domain, queue, review mode, and progress
remain visible. The reviewer may:

- accept, flag, fail, exclude, or mark a domain not applicable at image scale;
- inspect or annotate a flagged tile/region and choose an explicit region action;
- navigate a reasoned or random object queue;
- correct fiber type without overwriting model type;
- edit reviewed fiber or nucleus masks while predictions remain read-only;
- correct nucleus-to-fiber association independently of nucleus segmentation; and
- see downstream results marked stale whenever an edit invalidates them.

Only one domain is editable at a time. Drawing a region alone never applies an
irreversible action.

### 2.4 Save, resume, and finalize

- Persist both a current snapshot and appendable audit events.
- Restore image, queue, queue position, statuses, regions, object decisions,
  reviewed masks, and stale markers.
- Materialize a reviewed mask only after the first edit (copy-on-write).
- At finalization, apply section selection, domain and region exclusions,
  reviewed values, and recomputation state.
- Clearly distinguish predicted, reviewed, excluded, unresolved, and stale values.
- Report selection, exclusions, corrections, random-audit results, unresolved
  work, review burden, and all relevant versions.

## 3. Scope/domain model

### 3.1 Scope

Controlled values:

- `image`
- `region`
- `object`

### 3.2 Domain

Top-level controlled values:

- `fiber_segmentation`
- `fiber_typing`
- `nuclei`

Internal nucleus subdomains:

- `nuclei_segmentation`
- `nucleus_association`

Nucleus segmentation and association remain independently addressable internally,
but the application exposes three comprehensible top-level domain cards rather
than nine separate tools.

### 3.3 Status and action vocabulary

Domain status:

- `not_reviewed`
- `pass`
- `review`
- `fail`
- `excluded`
- `not_applicable`

Fiber-type review actions:

- accept model prediction;
- Type I, IIa, IIb, or IIx;
- uncertain;
- exclude; and
- unresolved.

Fiber and nucleus segmentation actions:

- accept;
- add;
- delete;
- split;
- merge;
- boundary paint/erase where applicable; and
- unresolved.

Nucleus association actions:

- accept;
- reassign to fiber;
- mark unassigned;
- mark ambiguous; and
- unresolved.

Region actions:

- exclude the selected domain in the region;
- queue all objects in the region;
- mark for detailed review;
- ignore nuclei;
- ignore fiber typing;
- exclude all analysis;
- recommend regional rerun; and
- unresolved.

### 3.4 Project and session entities

A `Project` contains:

- project ID/name;
- panel manifest reference;
- model version;
- verified image records;
- project-root-relative or absolute artifact references; and
- project, QC, review, finalized, and report locations.

Each image record contains:

- `image_id`, `mouse_id`, and `section_id`;
- condition metadata;
- `raw_image_path`;
- `prediction_directory`; and
- optional explicit output paths.

A shared `ReviewSession` tracks:

- project and current image;
- active scope, domain, review mode, and queue;
- queue position;
- per-image/per-domain statuses;
- region annotations;
- object decisions;
- reviewed-mask paths;
- stale/recomputation state;
- model, QC, and review schema versions;
- reviewer and timestamps; and
- save/resume position.

## 4. Architecture

### 4.1 Target boundaries

GUI-independent logic belongs under `src/review/` and must not import Qt, Napari,
or magicgui. Proposed Phase 1 modules:

```text
src/review/
├── __init__.py
├── schemas.py       # enums and validated snapshot records
├── project.py       # YAML project manifest loading and path resolution
├── session.py       # ReviewSession state transitions
├── storage.py       # atomic snapshots/events and mask copy-on-write
└── invalidation.py  # centralized dependency graph
```

Later phases may add:

```text
qc.py
qc_rules.py
queues.py
section_selection.py
mask_editing.py
lineage.py
finalization.py
dashboard_widget.py
review_hub_widget.py
object_inspector_widget.py
```

Thin module entry points should follow the repository's current non-packaged
style:

```text
uv run python -m src.generate_review_qc --project project.yaml
uv run python -m src.review_project_napari --project project.yaml
uv run python -m src.finalize_review_project --project project.yaml
uv run python -m src.report_review_project --project project.yaml
```

Do not change `[tool.uv] package = false` solely to add console scripts.

### 4.2 Application workspaces

One Napari application will coordinate shared backend state across:

1. Cohort Dashboard
2. Image Review
3. Region Review
4. Object Inspector
5. Finalization / Review Summary

These may be coordinated docks or stacked Qt pages. They must not become
independent applications with incompatible state.

### 4.3 Existing components to reuse

Repository audit completed against both current reviewers.

From `src/review_labels_napari.py`:

- multichannel loading through `src.io_utils.load_multichannel_image`;
- channel resolution through `fibertypeqc.config.resolve_channel_config`;
- typing composites and signal display from `src.typing_display`;
- downsampled display while preserving original fiber IDs;
- label and optional nuclei TIFF loading;
- click-to-select callbacks and selected-object outline;
- predicted/reviewed fiber-type overlays and existing color conventions;
- probability, confidence, margin, and QC-flag display;
- manual type/eMHC assignment and keyboard shortcuts;
- needs-review point overlay;
- add-by-polygon, add-by-brush, delete, and boundary paint/erase behaviors; and
- compatibility loading and saving of the existing manual-review CSV.

From `src/review_audit_napari.py`:

- audit-subset and manifest-row loading;
- queue-style next/previous navigation;
- centroid mapping, target highlighting, and zoom-to-object behavior;
- dense object context text;
- raw/enhanced/composite layer presets and display controls;
- keyboard navigation and classification shortcuts; and
- compatibility loading and saving of audit-reviewed CSVs.

From other repository modules:

- `src/merge_reviewed_labels.py` preserves the public reviewed-label merge;
- `src/fiber_type_labels.py` provides normalized fiber-type vocabulary;
- `src/io_utils.py` provides image loading;
- `src/typing_display.py` and `src/label_masks.py` provide display preparation;
- `fibertypeqc/artifacts.py` contains existing version/provenance conventions and
  a related pipeline-stage reuse matrix;
- `src/run_pipeline.py` already writes fiber and nuclear artifact paths and run
  provenance; and
- `src/run_batch.py` already supports explicit image IDs and portable relative
  paths in a batch manifest.

### 4.4 Audit findings and architectural conflicts

1. Both reviewers are large, independent Napari applications
   (`src/review_labels_napari.py`, 1,076 lines;
   `src/review_audit_napari.py`, 722 lines) with no shared session.
2. Both import Napari/Qt at module import time, so shared logic extracted from
   them would not remain safely headless.
3. Image-channel selection, display downsampling, review-table merging, and CSV
   saving are duplicated.
4. `ReviewState` in `src/review_labels_napari.py` is an unvalidated, fiber-only,
   in-memory object; the audit reviewer uses an ad hoc dictionary containing
   only an index.
5. Their review schemas are incompatible:
   `corrected_type`/`label_source` versus `audit_corrected_type` and audit flags.
6. Both reviewers save CSVs directly and non-atomically after decisions.
7. The fiber reviewer makes the displayed predicted labels layer editable and
   derives a corrected TIFF beside the prediction. It does not centrally record
   stale downstream products or lineage.
8. The audit reviewer derives prediction files from
   `<output_root>/<image_id>/<image_id>_*` and reads source paths from a CSV
   manifest, which conflicts with the proposed explicit, YAML project artifact
   model.
9. The audit reviewer has queue navigation and zoom, while the general fiber
   reviewer has click selection and segmentation editing; neither supplies both.
10. Neither reviewer stores project-wide status, regions, review events, review
    mode, random-audit provenance, versioned review state, or resume position.
11. Existing pipeline manifests and artifact names are useful but do not form a
    review-project schema.
12. Existing `fibertypeqc.artifacts.decide_artifact_reuse` is pipeline-oriented:
    it can inform but cannot replace object-edit invalidation rules.
13. Existing mask editing supports add/delete/paint only. Split, merge, undo,
    stable lineage, and nucleus edits are absent.
14. Existing corrected-mask saving is eager and manual rather than managed
    copy-on-write.

### 4.5 Consolidation approach

- Preserve both reviewer CLIs and existing output compatibility.
- First create the shared headless project/session/storage/invalidation contract.
- Integrate existing reviewer behavior incrementally in later phases rather than
  rewriting working interaction code.
- Keep old CSV compatibility at adapters/wrappers; use versioned canonical
  schemas internally.
- Keep prediction references immutable and place reviewed artifacts under the
  project's `review/` tree.

## 5. Data and output schemas

### 5.1 Project manifest

Preferred YAML shape:

```yaml
schema_version: review_project.v1
project_id: cohort_2026_01
project_name: Example cohort
panel_manifest: manifests/example_panel.yaml
model_version: model-id-or-version
images:
  - image_id: mouse_1_section_1
    mouse_id: mouse_1
    section_id: section_1
    condition:
      genotype: example
    raw_image_path: inputs/mouse_1_section_1.czi
    prediction_directory: predictions/mouse_1_section_1
    outputs:
      fiber_labels: mouse_1_section_1_cellpose_labels.tif
      fiber_table: mouse_1_section_1_fibers.csv
      nuclei_labels: mouse_1_section_1_nuclei_labels.tif
```

Relative paths resolve against the manifest directory. Explicit metadata takes
precedence over filename inference. IDs must be non-empty and image IDs unique.
Missing or corrupt manifests and paths produce contextual errors.

### 5.2 Project directory

```text
project/
├── project.yaml
├── predictions/
├── qc/
│   ├── image_qc.csv
│   ├── region_qc.csv
│   ├── fiber_qc.csv
│   └── nucleus_qc.csv
├── review/
│   ├── review_state.json
│   ├── review_events.csv
│   ├── review_regions.geojson
│   ├── reviewed_fiber_types.csv
│   ├── reviewed_fiber_labels/
│   ├── reviewed_nuclei_labels/
│   └── object_lineage.csv
├── finalized/
└── reports/
```

The manifest may reference predictions elsewhere; it must not move or duplicate
them.

### 5.3 Review snapshot

`review_state.json` is a versioned snapshot containing the `ReviewSession` fields
listed in section 3.4. It is saved atomically and treats absent optional
collections as empty. Unknown schema versions fail clearly until a migration is
implemented.

Image/domain status keys must preserve domain separation. A nuclei exclusion must
not imply a fiber-typing exclusion.

### 5.4 Review event

Each appendable event contains:

- `event_id`
- `image_id`
- `scope`
- `domain`
- optional `subdomain`
- `target_id`
- `action`
- `reason_code`
- `old_value`
- `new_value`
- `reviewer`
- `timestamp`
- `model_version`
- `qc_version`

Events explain review history but are not used to reconstruct masks.

### 5.5 Region annotation

GeoJSON feature properties contain:

- `region_id`
- geometry
- domain
- action
- reason code
- notes
- reviewer
- timestamp

### 5.6 Fiber-type review

Canonical rows keep separate:

- `fiber_id`
- `model_fiber_type`
- `reviewed_fiber_type`
- `review_status`
- queue/reason provenance
- reviewer/timestamp/version fields

Legacy `predicted_type`, `corrected_type`, and `label_source` remain supported by
compatibility loaders during migration.

### 5.7 Mask and lineage outputs

Predicted masks are immutable. The reviewed TIFF is created on first edit and is
authoritative thereafter. New object IDs are greater than the current maximum.

Lineage rows contain:

- `event_id`
- domain
- operation
- `parent_id`
- `child_id`
- timestamp

Splits may retain the parent ID for one child and allocate new IDs for others.
Merges retain one existing target ID and record every parent-to-child mapping.

### 5.8 QC tables and provenance

QC outputs retain component metrics and explicit reason codes, plus:

- `qc_version`
- rule/configuration version
- model version
- computation timestamp
- domain-specific status
- `hard_fail`
- optional ranking score/priority

A ranking score may order work but never replaces component values or reasons.

### 5.9 Finalized outputs

Final tables explicitly distinguish:

- model prediction;
- reviewed value;
- excluded value;
- unresolved value; and
- stale/not-recomputed value.

Reports include sections used per mouse/domain, exclusions and reasons, reviewed
and corrected object counts, random-audit results, unresolved items, review
burden, and schema/model/QC versions.

## 6. QC metrics and rules

QC is precomputed outside GUI click handlers. Rules live in a versioned,
human-readable YAML configuration and prefer cohort percentiles, median/MAD
robust z-scores, and within-image rankings over unexplained global thresholds.

### 6.1 Image-level metrics

Fiber segmentation:

- fiber count;
- tissue and segmented coverage;
- median fiber area;
- tiny/giant object fraction;
- border-touching fraction;
- solidity/eccentricity outlier fraction;
- shape plausibility; and
- membrane support when available.

Fiber typing:

- unknown/uncertain fraction;
- mean maximum class probability;
- low-margin and high-entropy fractions;
- probability conflict rate;
- dim/saturated marker metrics; and
- class uniformity/composition as informational only.

Nuclei:

- nuclei per tissue area and per fiber;
- unassigned and ambiguous association fractions;
- nuclear area distribution;
- weak DAPI support;
- DAPI saturation; and
- DAPI low-signal/dropout.

### 6.2 Region/tile metrics

On a configurable grid:

- local signal and saturation;
- tissue coverage;
- fiber density and fiber-size outliers;
- typing uncertainty;
- nuclei density;
- unassigned-nucleus fraction;
- DAPI support; and
- technical artifact flags.

The GUI exposes a heatmap/shapes overlay and navigation from flagged tile to
region.

### 6.3 Object metrics

Fiber QC:

- fiber ID, predicted type, class probabilities;
- maximum probability, margin, and entropy;
- area, eccentricity, and solidity;
- available neighborhood/context values;
- nucleus count;
- technical reasons; and
- queue priority.

Nucleus QC:

- nucleus ID and area;
- DAPI support;
- assigned fiber ID;
- association overlap and distance;
- morphology outliers;
- technical reasons; and
- queue priority.

### 6.4 Technical versus biological rules

Technical hard failures may include missing tissue, extreme coverage loss,
focus/signal failure, channel dropout, saturation, folds/debris, implausible
segmentation, gross image/label mismatch, unusable DAPI support, and corrupt or
missing outputs.

The following biological endpoints must not independently hard-fail or select a
section:

- eMHC-positive fraction;
- fiber-type composition;
- central nuclei rate;
- average fiber size;
- treatment endpoints; and
- unexpectedly severe pathology.

They may be informational consistency checks. When pathology makes segmentation
uncertain, retain the reason and route to review rather than silently exclude.

### 6.5 Section selection rule

For each mouse and selected domain:

```text
passing = sections without a technical hard failure for that domain

if passing is empty:
    route mouse/domain to manual review
elif one section passes:
    use it provisionally
elif multiple sections pass:
    default to all passing
    permit best-passing or manual selection
```

## 7. Review modes

### 7.1 QC-gated automatic

- Accept technically passing output provisionally.
- Reject hard technical failures.
- Use all passing sections or the best passing section.
- Route unresolved mice to review.
- Include a configurable random audit sample.

The professional user-facing name is **QC-gated automatic**, never “lazy.”

### 7.2 Flagged review

- Review images with REVIEW status.
- Review flagged regions.
- Review low-confidence and rule-based object outliers.
- Include a random audit sample.
- Accept unflagged output provisionally.

### 7.3 Domain-focused review

- Let the user choose domains requiring detailed review.
- Permit image-QC-only treatment in other domains.
- Permit domains to be marked not applicable.

### 7.4 Full audit

- Review every image.
- Review every flagged region.
- Review every object or a configured structured sample.
- Use for validation and benchmark creation.

### 7.5 Queue types

- low confidence;
- high entropy;
- low probability margin;
- probability conflicts;
- morphology outliers;
- weak membrane support;
- weak DAPI support;
- ambiguous nucleus association;
- unusual nuclei per fiber;
- manually selected region;
- random audit.

Random auditing is mandatory because confidence queues cannot expose confidently
wrong predictions, missed objects, or QC blind spots. Sampling is configurable by
image, mouse, domain, or cohort and reproducible with a seed. Every decision
records the queue source.

## 8. Invalidation dependencies

Rules are centralized and stale state is visible from Phase 1 onward.
Recomputation may initially occur at save/finalize rather than after every edit.

| Edit | Invalidated products |
|---|---|
| Fiber mask | fiber geometry/features; fiber-type prediction/features; nucleus-to-fiber associations; fiber-level nucleus counts |
| Nucleus mask | nucleus features; nucleus associations; fiber-level nucleus counts |
| Nucleus reassignment | association output; affected fiber nucleus counts |
| Fiber-type correction | none in segmentation |
| Region exclusion | only selected domains in that region, unless “exclude all” |

The interface must state what changed and which outputs await recomputation.
Stale downstream output must never be silently presented as current.

## 9. Implementation phases

Each phase must end as a tested, usable vertical slice and retain old reviewer
CLI compatibility.

### Phase 1: repository audit and shared foundation

- complete this repository/reviewer audit;
- add controlled schemas and validation;
- add YAML project-manifest loading and path resolution;
- add shared `ReviewSession`;
- add atomic snapshot/event storage and save/resume;
- enforce predicted/reviewed path separation;
- add mask copy-on-write;
- centralize stale-state dependencies; and
- add headless tests.

No Qt/Napari UI integration and no scientific QC thresholds are introduced in
Phase 1.

### Phase 2: QC engine and project dashboard

- image/fiber/nucleus QC tables;
- versioned configurable rules and explicit reasons;
- mouse/section grouping;
- domain-specific section recommendations;
- cohort dashboard; and
- tests.

### Phase 3: image-level review

- domain cards and status actions;
- four review modes;
- image navigation;
- project save/resume integration; and
- tests.

### Phase 4: fiber-type object review integration

- reuse current per-fiber reviewer;
- queue navigation and reasons;
- reproducible random audit;
- canonical model/reviewed/status fields; and
- tests.

### Phase 5: region review

- shapes layer;
- explicit region action form;
- regional heatmaps;
- region queues and exclusions; and
- tests.

### Phase 6: nuclear review

- reviewed-mask editing;
- association editing;
- queues;
- reviewed outputs;
- stale propagation; and
- tests.

### Phase 7: fiber segmentation review

- add/delete/split/merge/boundary editing;
- stable IDs;
- lineage;
- downstream invalidation; and
- tests.

### Phase 8: finalization and reports

- domain-specific selected sections;
- exclusions;
- recomputation hooks;
- final tables;
- review burden and audit summaries; and
- tests.

## 10. Required tests

Most tests use small synthetic arrays and temporary directories and remain
headless. Add a small number of GUI tests only where feasible.

- [x] Project manifest validation.
- [x] Missing/corrupt paths produce clear errors.
- [x] Save then reload preserves review state.
- [x] Predicted files are never overwritten.
- [x] Reviewed masks use copy-on-write.
- [x] Image statuses persist.
- [x] Domain-specific exclusions remain separate.
- [x] Region geometry/action/domain persist.
- [x] Fiber-type correction preserves model prediction.
- [ ] Random queue is reproducible with a seed.
- [ ] Split assigns valid stable IDs.
- [ ] Merge preserves lineage.
- [x] Fiber edit marks downstream outputs stale.
- [x] Nucleus edit does not invalidate fiber typing.
- [ ] Nucleus reassignment updates affected associations.
- [ ] Region exclusion affects only selected domains.
- [ ] `all_passing` selection works.
- [ ] `best_passing` selection works.
- [ ] No-passing selection routes to review.
- [ ] Biological endpoint values do not trigger technical hard fails.
- [ ] Finalization respects section and region exclusions.
- [x] Resume restores queue position.
- [ ] Old reviewer output remains loadable where applicable.

After each phase:

```text
uv run python -m pytest -q
uv run ruff check <changed Python files and tests>
```

## 11. Current progress checklist

### Planning and audit

- [x] Confirmed branch is `feature/napari-review-workflow`.
- [x] Confirmed the starting worktree is clean.
- [x] Read the complete product specification.
- [x] Read repository `AGENTS.md`.
- [x] Inspected repository structure and current CLI/module conventions.
- [x] Audited `src/review_labels_napari.py`.
- [x] Audited `src/review_audit_napari.py`.
- [x] Identified reusable loaders, selection, zoom, shortcuts, colors,
  probability display, and CSV compatibility behavior.
- [x] Recorded architectural conflicts and decisions.
- [x] Ran baseline full suite: 131 passed, 36 warnings.
- [x] Created this plan before Phase 1 implementation.
- [x] Committed plan separately as `660ff28`.

### Phase 1

- [x] Added `src/review/` headless package.
- [x] Added controlled enums and versioned schemas.
- [x] Added validated YAML project loader and resolved artifact paths.
- [x] Added `ReviewSession` with domain-separated status and queue position.
- [x] Added atomic save/resume and appendable event storage.
- [x] Added immutable prediction path checks and reviewed mask copy-on-write.
- [x] Added centralized invalidation/stale tracking.
- [x] Added 15 Phase 1 headless tests.
- [x] Ran full repository lint: all checks passed.
- [x] Ran full suite: 146 passed, 36 existing warnings.
- [x] Committed Phase 1 separately from the planning commit.

### Later phases

- [ ] Phase 2 QC engine/dashboard.
- [ ] Phase 3 image review.
- [ ] Phase 4 fiber-type reviewer integration.
- [ ] Phase 5 region review.
- [ ] Phase 6 nuclear review.
- [ ] Phase 7 fiber segmentation review.
- [ ] Phase 8 finalization/reporting.

## 12. Decisions and deviations

Every meaningful change from the proposed design is recorded here.

1. **Keep modules under `src/review/` in Phase 1.** This matches the requested
   architecture and the repository's existing `python -m src...` execution
   convention. No packaging change is needed.
2. **Use standard-library dataclasses plus explicit validation, not a new schema
   dependency.** The repository has no Pydantic dependency. Dataclasses, enums,
   and PyYAML are sufficient for Phase 1 and avoid adding a large dependency.
3. **Resolve manifest-relative paths against `project.yaml`.** Existing batch
   manifests often need a separate input root; a project file should be portable
   as a unit. Absolute paths remain accepted but are not emitted in public demo
   material.
4. **Require explicit unique image/mouse/section IDs in the canonical manifest.**
   Starter-manifest inference is deferred to a later CLI slice and must be
   verified by the user. Phase 1 will not silently infer identity from filenames.
5. **Treat prediction directories as potentially external and read-only.**
   Reviewed masks are rooted under the project review directory rather than saved
   beside predictions, unlike the current fiber reviewer.
6. **Do not refactor either Napari reviewer in Phase 1.** The specification says
   to reuse them rather than rewrite them, but connecting GUI behavior before the
   shared state contract exists would create disconnected scaffolding. The two
   CLIs remain unchanged and will become adapters in later phases.
7. **Use atomic replacement for snapshot and newly written event-table files.**
   Existing reviewer CSV writes are non-atomic. Phase 1 storage establishes the
   safer contract; legacy reviewers keep current behavior until integration.
8. **Store appendable logical events as a CSV audit table, but rewrite it
   atomically when adding an event.** This preserves the requested flat-file
   schema and crash safety at current expected scale. If event volume later makes
   this impractical, the decision will be revisited explicitly rather than
   quietly adding a database.
9. **Make the reviewed TIFF authoritative but never event-reconstructed.**
   Events describe mask changes; they do not replace mask persistence.
10. **Define invalidation at product-family granularity in Phase 1.** Regional
    target IDs and recomputation execution arrive with later editing/finalization
    phases, but stale dependency tracking exists immediately.
11. **Do not reuse `fibertypeqc.artifacts.decide_artifact_reuse` directly.** Its
    inputs describe pipeline configuration changes, not manual object edits.
    Phase 1 adds a review-specific dependency map while keeping naming aligned.
12. **Keep nuclei as a top-level UI domain with explicit internal subdomains.**
    This follows the product model while allowing association-only invalidation.
13. **Do not add QC thresholds or section-selection behavior in Phase 1.** Those
    could affect scientific acceptance and belong to Phase 2 with dedicated
    rules, tests, and review.
14. **Treat old reviewer schemas as compatibility formats, not the canonical new
    schema.** Canonical state separates model value, reviewed value, and status;
    migrations/adapters will preserve old output loading in the integration
    phase.
15. **No product-goal deviation.** The project-based progressive workflow,
    mandatory random audit, technical-versus-biological separation, immutable
    predictions, and finalization requirements remain unchanged.

## 13. Known limitations

- The current reviewers remain separate applications until Phases 3–4.
- Current reviewer CSV saves are non-atomic until those UIs use shared storage.
- Current fiber-mask edits do not record lineage or stale dependencies.
- Split, merge, undo, nuclear mask editing, and association editing are not
  implemented.
- The current audit reviewer depends on a CSV audit set and inferred output
  paths; it does not yet open a project YAML.
- Phase 1 has no QC computation, dashboard, section selection, queues, region UI,
  finalization, or reporting.
- Starter-manifest generation and verification UI are not part of the initial
  foundation.
- Flat-file event storage is appropriate for the planned scale but may require
  measurement before very large multi-reviewer cohorts.
- Concurrent multi-user editing and conflict resolution are not included.
- Copy-on-write protects prediction masks only when callers use shared storage;
  legacy reviewers are not yet routed through it.
- The provisional nuclear baseline remains intentionally unchanged.
- GUI tests may require a display-capable CI environment; core behavior will be
  covered headlessly.
