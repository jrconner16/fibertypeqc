# QC and Manual Review System Plan

Status: Phase 5 complete; mask-level review remains deferred.

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
    applicable_domains:
      - fiber_segmentation
      - fiber_typing
      - nuclei
    outputs:
      fiber_labels: mouse_1_section_1_cellpose_labels.tif
      fiber_table: mouse_1_section_1_fibers.csv
      nuclei_labels: mouse_1_section_1_nuclei_labels.tif
      nuclei_table: mouse_1_section_1_nuclei.csv
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

### 5.10 Phase 2A QC output schemas

Phase 2A writes four CSV files under `qc/`. JSON-valued columns use compact,
sorted JSON rather than Python representations.

`image_qc.csv` has one row per image and applicable top-level domain, plus an
explicit `not_applicable` row for other domains:

- provenance: `schema_version`, `qc_version`, `rules_version`, `model_version`,
  `computed_at`;
- identity: `project_id`, `image_id`, `mouse_id`, `section_id`, `domain`;
- disposition: `applicable`, `status`, `hard_fail`,
  `technical_quality_score`, `review_priority`;
- explanations: `reason_codes` (pipe-delimited), `reason_details_json`;
- inputs: `artifact_paths_json`;
- common availability/consistency metrics:
  `fiber_labels_available`, `fiber_labels_valid`, `fiber_table_available`,
  `nuclei_labels_available`, `nuclei_labels_valid`,
  `nuclei_table_available`, `label_shape_match`;
- fiber-segmentation metrics: `fiber_count`, `image_pixel_count`,
  `segmented_pixel_count`, `segmented_image_fraction`,
  `median_fiber_area_px`, `border_touching_fiber_count`,
  `border_touching_fiber_fraction`, `fiber_id_mismatch_fraction`;
- fiber-typing metrics: `typing_row_count`, `prediction_available`,
  `unknown_fraction`, `needs_review_fraction`, `probability_row_count`,
  `probability_coverage`, `mean_max_probability`, `mean_probability_margin`,
  `mean_normalized_entropy`, `type_counts_json`;
- nuclei metrics: `nucleus_count`, `nucleus_pixel_count`,
  `nucleus_image_fraction`, `median_nucleus_area_px`,
  `nucleus_id_mismatch_fraction`, `unassigned_nucleus_fraction`,
  `ambiguous_nucleus_fraction`, `mean_association_overlap`,
  `assigned_nuclei_per_fiber`.

Metrics not applicable to a row are null, never zero-filled. Absence therefore
cannot be confused with a measured zero.

`fiber_qc.csv` has one row per positive fiber-mask ID:

- provenance (including `model_version`) and image identity fields;
- `fiber_id`, `area_px`, `touches_image_border`;
- `predicted_type`, `prob_i`, `prob_iia`, `prob_iib`, `prob_iix`;
- `max_probability`, `probability_margin`, `normalized_entropy`,
  `needs_review`;
- `technical_reason_codes`, `review_priority`.

Typing fields are null when the fiber table or probabilities are unavailable.
Mask IDs remain authoritative for object rows; table-only IDs are represented by
the image-level mismatch metric rather than invented mask objects.

`nucleus_qc.csv` has one row per positive nucleus-mask ID:

- provenance (including `model_version`) and image identity fields;
- `nucleus_id`, `area_px`;
- `assigned_fiber_id`, `assignment_status`, `association_category`,
  `overlap_fraction`, `distance_to_boundary_px`,
  `normalized_radial_position`;
- `technical_reason_codes`, `review_priority`.

Association fields are null when unavailable. Mask IDs remain authoritative.

`section_selection.csv` has one row per mouse/domain for the requested strategy:

- provenance and grouping: `schema_version`, `qc_version`, `rules_version`,
  `model_version`, `computed_at`, `project_id`, `mouse_id`, `domain`,
  `strategy`;
- result: `selected_image_ids` (pipe-delimited),
  `eligible_image_ids` (pipe-delimited), `requires_manual_review`,
  `reason_code`.

The selected IDs always refer to explicit manifest image IDs, never inferred
filenames.

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

### 6.6 Exact Phase 2A metric design

Phase 2A intentionally does not load raw microscopy channels. It measures the
integrity and review burden of prediction artifacts already produced by the
pipeline. Raw-signal, tissue-mask, focus, saturation, membrane-support, and DAPI
support metrics remain deferred until their input contracts and calibration
datasets exist.

#### Artifact contract and domain applicability

Canonical `images[].outputs` keys are:

| Artifact key | Domain use | Required when applicable |
|---|---|---|
| `fiber_labels` | fiber segmentation; fiber object rows; nuclei shape comparison | yes for fiber segmentation |
| `fiber_table` | fiber typing; typing fields in fiber object rows | yes for fiber typing |
| `nuclei_labels` | nuclei segmentation; nucleus object rows | yes for nuclei |
| `nuclei_table` | nucleus association metrics and fields | yes for nuclei |
| `nucleus_fiber_links` | future association audit cross-check | no in Phase 2A |
| `fiber_nuclei` | future per-fiber nuclear cross-check | no in Phase 2A |

Each image may declare `applicable_domains` as a list of controlled domain
values. If omitted:

- fiber segmentation is applicable;
- fiber typing is applicable only when `fiber_table` is declared; and
- nuclei is applicable only when `nuclei_labels` or `nuclei_table` is declared.

An absent artifact for an applicable domain produces a QC row and an explicit
hard-fail reason; QC generation continues for other images/domains. An absent
artifact for a non-applicable domain produces `status=not_applicable` and no
failure. Unreadable, non-2D, non-finite, negative, or non-integral label masks are
structurally invalid. A fiber/nucleus shape mismatch is a nuclei-domain hard
failure. Missing optional columns or probability vectors produce null metrics
and informational reasons rather than fabricated values.

#### Fiber-segmentation formulas

For a valid 2D fiber label mask \(L_f\), let \(P\) be the number of image pixels,
\(S = \{p : L_f(p) > 0\}\), and \(F\) the set of unique positive IDs.

- `fiber_count = |F|`.
- `image_pixel_count = P`.
- `segmented_pixel_count = |S|`.
- `segmented_image_fraction = |S| / P`.
- `area_px(f) = count(L_f == f)`.
- `median_fiber_area_px = median(area_px(f) for f in F)`.
- a fiber touches the image border when its ID appears in the first/last row or
  first/last column.
- `border_touching_fiber_fraction = border_touching_fiber_count / |F|`.
- if the table has IDs \(T\),
  `fiber_id_mismatch_fraction = |F symmetric_difference T| / |F union T|`.

Zero-denominator fractions are null except `segmented_image_fraction`, whose
denominator is a valid non-empty image. An all-background fiber mask has
`fiber_count=0` and triggers the unmistakable `fiber_segmentation.no_objects`
hard failure. Area and border fractions remain informational in the default
configuration because pathology, cropping, and acquisition geometry affect
their distributions.

#### Fiber-typing formulas

The denominator is the number of unique positive integer fiber IDs after keeping
the first row for each duplicated ID; duplicate or invalid IDs trigger REVIEW.
The prediction column is the first available
of `predicted_type`, `fiber_type`, or `model_prediction`.

- `typing_row_count = number of valid table rows`.
- `unknown_fraction = count(prediction is blank, "unknown", or "uncertain") /
  typing_row_count`. `iix_candidate` is not counted as unknown.
- `needs_review_fraction = count(needs_review is true) / typing_row_count` when
  that column exists.
- probability columns are the available members of
  `prob_i`, `prob_iia`, `prob_iib`, `prob_iix`.
- a probability row is usable when at least two values are finite,
  non-negative, and have a positive sum. Values are normalized to sum to one.
- `probability_coverage = probability_row_count / typing_row_count`.
- per usable row, `max_probability = max(p)`.
- per usable row, `probability_margin = largest(p) - second_largest(p)`.
- per usable row with \(K\) probabilities,
  `normalized_entropy = -sum(p * ln(p)) / ln(K)`, treating `0 ln 0` as zero.
- image means use only usable probability rows.
- `type_counts_json` records exact predicted-label counts and is informational.

No default confidence, margin, entropy, unknown-rate, `needs_review`-rate, or
composition threshold is enabled. Those distributions require model/panel
calibration. Missing probabilities trigger an informational reason only.

#### Nuclei formulas

For a valid 2D nucleus mask \(L_n\), use the same mask definitions as fibers:

- `nucleus_count = number of unique positive IDs`.
- `nucleus_pixel_count = count(L_n > 0)`.
- `nucleus_image_fraction = nucleus_pixel_count / image_pixel_count`.
- `area_px(n) = count(L_n == n)`.
- `median_nucleus_area_px = median(area_px(n))`.
- `nucleus_id_mismatch_fraction` uses the same symmetric-difference/union
  formula between mask and table IDs.

For a nuclei table with \(N\) unique valid positive nucleus IDs after keeping the
first duplicate:

- `unassigned_nucleus_fraction = count(assignment_status ==
  "unassigned_or_interstitial" or assigned_fiber_id == 0) / N`.
- `ambiguous_nucleus_fraction = count(assignment_status == "ambiguous") / N`.
- `mean_association_overlap = mean(overlap_fraction)` over finite values.
- `assigned_nuclei_per_fiber = count(assignment_status == "assigned") /
  fiber_count` when a valid nonempty fiber mask exists.

Nuclear density, central nuclei, nuclei per fiber, and area distributions are
biologically sensitive and remain informational. An empty but structurally valid
nucleus mask triggers REVIEW, not hard failure. Uncalibrated association
fractions do not trigger review by default.

#### Missing-input behavior

| Condition | Result |
|---|---|
| Missing/corrupt required label mask | applicable domain hard fails; dependent metrics null |
| Missing/corrupt required table | applicable domain hard fails; mask-only metrics still emitted |
| Missing prediction column in an applicable typing table | typing hard fails |
| Empty applicable typing table | typing hard fails |
| Empty fiber mask | fiber segmentation hard fails |
| Empty nucleus mask | nuclei REVIEW |
| Fiber/nucleus mask shape mismatch | nuclei hard fails |
| Missing probabilities/optional association columns | metrics null; informational reason |
| Missing artifact for non-applicable domain | `not_applicable`; no reason penalty |

#### Technical quality versus review priority

These fields are deliberately separate and transparent:

- `technical_quality_score` is a coarse disposition summary, not a biological
  quality model: `1.0` for PASS, `0.5` for REVIEW, `0.0` for hard FAIL, and null
  when not applicable.
- `review_priority` sorts work: `100 * hard_fail_reason_count +
  10 * review_reason_count`. Informational reasons add zero. Higher values are
  reviewed first; stable manifest order breaks ties.

Neither field contains biological endpoints. The score must not be interpreted
as a calibrated probability, and priority must not be used as an acceptance
threshold.

#### Rule severities and default rules

Controlled severities:

- `informational`: retain a reason without changing status or priority;
- `review`: set REVIEW unless a hard failure is present and add 10 priority;
- `hard_fail`: set FAIL, set `hard_fail=true`, and add 100 priority.

The versioned YAML rule configuration contains `rules_version`, `qc_version`,
and rules with `reason_code`, `domain`, `metric`, `operator`, `threshold`,
`severity`, `enabled`, and `description`.

Enabled default hard failures are restricted to:

- missing/unreadable/invalid required fiber labels;
- no positive fiber objects;
- missing/unreadable/empty applicable fiber table;
- missing typing prediction column;
- missing/unreadable/invalid required nuclei labels;
- missing/unreadable required nuclei table; and
- fiber/nucleus label-shape mismatch.

Enabled default REVIEW rules are:

- fiber mask/table ID mismatch greater than zero;
- duplicate/invalid fiber-table IDs;
- empty nucleus mask;
- nucleus mask/table ID mismatch greater than zero; and
- duplicate/invalid nucleus-table IDs.

Enabled informational rules report unavailable typing probabilities and optional
nucleus association fields. Rules for segmented fraction, border fraction,
fiber/nuclear area, typing confidence/margin/entropy, unknown fraction,
composition, nuclei density, unassigned/ambiguous fractions, and association
overlap are present but disabled or informational with no default threshold.

### 6.7 Phase 2A test fixtures and expected results

All fixtures use temporary project directories and synthetic TIFF/CSV artifacts.

1. **Two-fiber geometry fixture:** a `4 x 5` mask contains fiber 1 in four
   interior pixels and fiber 2 in four pixels touching the right border.
   Expected: `fiber_count=2`, `segmented_image_fraction=8/20`,
   `median_fiber_area_px=4`, and `border_touching_fiber_fraction=1/2`.
2. **Typing fixture:** two rows have probabilities `[0.8, 0.2]` and
   `[0.5, 0.5]`. Expected mean maximum probability `0.65`, mean margin `0.30`,
   probability coverage `1.0`, and normalized entropies calculated with the
   two-class formula. No default confidence/entropy rule changes PASS status.
3. **Nucleus fixture:** four nuclei contain two assigned, one ambiguous, and one
   unassigned nucleus. Expected unassigned fraction `1/4`, ambiguous fraction
   `1/4`, and assigned nuclei per fiber `2/fiber_count`.
4. **Missing typing table fixture:** typing is explicitly applicable but its
   declared table is absent. Expected typing FAIL with
   `fiber_typing.missing_table`; fiber segmentation remains independently
   evaluable.
5. **Biological-extreme fixture:** all fibers have the same predicted type and
   all nuclei are assigned/central. Expected no technical hard failure because
   composition and biological outcomes are informational.
6. **Selection fixture:** one mouse has PASS, REVIEW, and hard-FAIL sections.
   `all_passing` selects PASS and REVIEW sections; `best_passing` selects PASS;
   manual selection accepts an explicitly chosen eligible section. A mouse with
   only hard failures selects none and routes to review.
7. **CLI fixture:** generation writes all four versioned CSVs, respects an
   alternate rules file and selection strategy, and does not import Napari/Qt.

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

### Phase 2A: headless QC engine and section selection

- artifact-derived image/fiber/nucleus QC tables;
- versioned configurable rules and explicit reasons;
- mouse/section grouping;
- domain-specific `all_passing`, `best_passing`, and `manual` selection;
- headless CLI;
- tests and schema/workflow documentation.

### Phase 2B: project dashboard

- validate and load the versioned Phase 2A QC tables;
- build a GUI-independent dashboard model;
- summarize domain PASS / REVIEW / FAIL / not-applicable counts;
- classify mouse/domain readiness as complete, targeted review, or no acceptable
  section;
- show section-level status, technical score, review priority, selection, and
  explicit reason codes;
- show review progress and correction burden from an existing `ReviewSession`;
- allow in-memory switching between `all_passing` and `best_passing`, plus a
  supplied manual selection;
- add a read-only Napari/Qt cohort dashboard dock;
- add a project dashboard CLI; and
- add headless tests plus a minimal offscreen widget smoke test.

Phase 2B does not load raw images, edit review state, persist changed section
selections, or implement image/region/object review. Those actions begin in
Phase 3.

#### Phase 2B data validation

The dashboard requires `image_qc.csv`; `fiber_qc.csv`, `nucleus_qc.csv`, and
`section_selection.csv` are loaded when present. It rejects:

- unsupported schema versions;
- missing identity/status/reason columns;
- project IDs that do not match `project.yaml`;
- QC image IDs absent from the manifest;
- duplicate image/domain rows;
- missing image/domain rows for the three controlled domains; and
- mixed QC/rules/model provenance within one table.

Boolean CSV fields are parsed explicitly rather than with Python truthiness, so
the string `"False"` cannot become true.

#### Phase 2B dashboard model

The GUI-independent model contains:

- cohort counts: mice, sections, applicable image/domain rows, reviewed
  image/domain rows, and correction burden;
- per-domain PASS / REVIEW / FAIL / not-applicable counts;
- one mouse/domain readiness row with section counts, acceptable and selected
  image IDs, `requires_manual_review`, and explicit selection reason;
- one section/domain display row with score, priority, selection, and QC reasons.

An acceptable section is applicable and has no technical hard failure.
Mouse/domain readiness is:

- `no_acceptable_section` when no acceptable section exists;
- `targeted_review` when selection requires review, no PASS section exists, or a
  selected section has REVIEW status; and
- `complete` otherwise.

A mouse is `no_acceptable_section` if any applicable domain has that readiness;
it is `targeted_review` if none are unacceptable and at least one domain needs
targeted review; otherwise it is complete.

Review progress is:

```text
reviewed applicable image/domain statuses /
applicable image/domain rows
```

where a stored status other than `not_reviewed` counts as reviewed. Correction
burden is reported as separate object-decision, region, and reviewed-mask counts;
the dashboard does not combine them into a scientific score.

#### Phase 2B widget and launcher

The dashboard dock contains:

- a persistent context header showing project, scope (`cohort`), domain filter,
  and section-selection strategy;
- cohort summary labels;
- a domain-status count table;
- a mouse → section → domain tree;
- status/domain filters;
- a reason/details panel; and
- an in-memory strategy selector.

The launcher is:

```text
uv run python -m src.review_project_napari --project project.yaml
```

It accepts an optional QC directory, initial strategy, and manual-selection YAML.
It opens the dashboard only; image loading and edit actions remain Phase 3.

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

### Phase 4.5: guided-review UX consolidation

Refine the completed Phase 2B–4 capabilities into a focused reviewer workflow
without removing advanced controls or changing review-state semantics:

- make the cohort dashboard a launch point rather than a permanently competing
  workspace;
- present a clear review-plan choice first (for example, flagged fibers,
  selected section, or cohort QC), with **Review flagged fibers** as the
  default entry point when applicable;
- show one active domain and one current object at a time, while representing
  other domain states as compact context/status chips;
- prioritize plain-language actions such as **Keep model call** and **Needs
  follow-up**, retaining technical status names and QC evidence in expandable
  details;
- group uncommon actions (uncertain, exclude, unresolved, queue configuration,
  random-audit settings) under an explicit advanced/details affordance;
- automatically advance after a saved object decision and provide an immediate,
  reversible Undo affordance;
- retain a short persistent context header with mouse, section, domain, queue,
  and queue position; and
- add reviewer-oriented usability tests, including keyboard navigation and
  resume/undo behavior.

This is a presentation and interaction milestone only. It must preserve the
existing project manifest, QC tables, ReviewSession fields, audit events,
prediction immutability, and advanced controls.

### Phase 4.6: reviewer navigation, recovery, and stain-aware display

Improve confidence and recoverability while retaining the focused Phase 4.5
workflow:

- add a persistent, clickable cohort → section → domain → object navigator;
- provide a globally reachable Workspace menu to restore Guided Review, cohort
  QC, image controls, and the channel map after any dock is closed;
- show a concise first-run guide per project, with an always-available replay
  control rather than a blocking tutorial;
- load panel-aware display mappings where available, with a display-only legacy
  fallback for incomplete manifests;
- show a stain-aware additive composite by default, plus individually named
  color layers such as `IIb (ch 3)` and a compact channel-map legend; and
- add tests for mapping, navigation callbacks, workspace recovery boundaries,
  and onboarding dismissal.

Display colors and onboarding preferences must never affect predictions, QC,
review decisions, or project artifacts.

### Phase 4.7: responsive display and review ergonomics

Make large-section review practical without changing the scientific workflow:

- add a `--display-downsample` launcher option that applies the same
  display-only stride to raw and label layers while preserving label IDs;
- replace the filled selected-fiber overlay with a high-contrast outline that
  leaves raw stain signal visible;
- provide a button and shortcut that re-centers the current reviewed fiber, and
  avoid reloading a section while moving between its fibers;
- surface and implement reviewer shortcuts for navigation, keeping the model
  call, type choices, and immediate Undo; and
- document an explicit normal-flow smoke-test walkthrough, including recovery
  from closed docks and the boundary between fiber-object and other domains.

The full-resolution prediction artifacts remain immutable; display resolution
must not affect decisions or persisted outputs.

### Phase 5: region review

- shapes layer;
- explicit region action form;
- regional heatmaps;
- region queues and exclusions; and
- tests.

### Phase 5.1: named analysis ROIs and subregions

Extend region review with a separate, non-destructive **analysis ROI** workflow
for anatomical subregions such as the four quadrants of a quadriceps section.
This is distinct from artifact/exclusion annotations:

- draw and name reusable anatomical ROIs with a role (for example, `quadrant`);
- preserve ROI geometry and metadata in review GeoJSON without changing
  predictions or masks;
- define and display overlap, outside-ROI, and boundary-assignment policy;
- assign objects by centroid during finalization, recording `region_id`,
  `region_name`, and `region_role` in reviewed/final CSV outputs; and
- retain ambiguous or boundary cases explicitly rather than silently forcing a
  membership.

The default intended policy is non-overlapping named ROIs with centroid-based
assignment. Finalization/reporting owns the output columns so region drawing
does not change scientific results until that step.

### Phase 6: nuclear review

- reviewed-mask editing;
- association editing;
- queues;
- reviewed outputs;
- stale propagation; and
- tests.

### Phase 6.1: reviewed nucleus addition

- [x] Add painted nuclei only to a copy-on-write reviewed mask.
- [x] Allocate reviewed nucleus IDs that remain stable across deletion and resume.
- [x] Include manually added nuclei in reviewed association output as unresolved
  until independently associated.
- [x] Mark nucleus features, associations, and fiber-level nucleus counts stale.
- [x] Add headless tests for addition, stable IDs, and invalid geometry.

### Phase 7: fiber segmentation review

- add/delete/split/merge/boundary editing;
- stable IDs;
- lineage;
- downstream invalidation; and
- tests.

### Phase 8:blinded primary review and model - auidit comparison 
Add a blinded mode to the existing guided reviewer so the primary reviewer can make independent decisions without being biased by the model prediction or, when desired, sample identity.

Support two review presets:

Model-blinded: hide the model classification, class probabilities, confidence metrics, model-colored overlays, QC reasons, needs_review status, and queue source.
Fully blinded: additionally hide filenames, image/mouse/section IDs, condition, genotype, treatment, time point, and other identifying metadata. Display stable neutral aliases instead.

Blinding is presentation-only. The application may continue using the real project IDs and paths internally, and prediction artifacts remain unchanged.

The reviewer must still see:

raw stain channels;
stain/channel identities;
the selected-object outline;
enough surrounding tissue context to make a valid decision;
neutral queue progress;
the normal review actions, shortcuts, automatic advance, Undo, save, and resume controls.

Use reproducible random or stratified audit queues. Do not reveal whether an object was randomly sampled, model-flagged, or selected because of a particular confidence or QC result while blinded review is active.

Save the blinded human decision separately from the model prediction. After the audit queue is completed, provide an explicit action to reveal the model results and compare:

human decision;
model prediction;
model confidence/probabilities;
agreement or disagreement;
queue and QC reasons.

Once model results have been revealed, preserve the original blinded decision. Any later change must be recorded separately as a post-reveal decision rather than silently replacing the blinded call.

Prevent identifying or model information from leaking through layer names, headers, details panels, tooltips, window titles, status text, or model-colored overlays.

Phase 8 should export blinded-review decisions and human-versus-model comparison results for Phase 9 reporting.

Phase 8 does not include:

a required second reviewer;
inter-reviewer adjudication;
collaborative review;
model retraining;
final scientific-output selection;
finalization or reporting.

Add tests confirming that hidden model and metadata fields are not exposed during the applicable blinded mode, save/resume preserves the blinded queue, model reveal is explicit, and model predictions remain immutable.

### Phase 9: finalization and reports

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
- [x] Nucleus reassignment updates affected associations.
- [ ] Region exclusion affects only selected domains.
- [x] `all_passing` selection works.
- [x] `best_passing` selection works.
- [x] No-passing selection routes to review.
- [x] Biological endpoint values do not trigger technical hard fails.
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

### Phase 2A

- [x] Documented exact artifact inputs, formulas, denominators, and missing-input
  behavior before implementation.
- [x] Documented `technical_quality_score` separately from `review_priority`.
- [x] Added explicit and inferred per-image domain applicability.
- [x] Added versioned YAML QC rules with controlled severities.
- [x] Restricted default hard failures to structurally unusable artifacts.
- [x] Left uncalibrated distribution and biological rules informational/disabled.
- [x] Added headless image, fiber, and nucleus QC generation.
- [x] Added explicit reason codes and version/model provenance.
- [x] Added mouse/section grouping and all-passing, best-passing, and manual
  selection.
- [x] Added the `src.generate_review_qc` CLI and four atomic CSV outputs.
- [x] Added operator-facing Phase 2A documentation and output-schema links.
- [x] Added 10 Phase 2A tests; focused review tests pass.
- [x] Ran changed-file Ruff checks: all checks passed.
- [x] Ran full suite: 156 passed, 36 existing warnings.
- [x] Committed Phase 2A separately from Phases 1 and 2B.

### Phase 2B

- [x] Defined the exact read-only dashboard boundary before implementation.
- [x] Recorded the authoritative frozen eMHC/DAPI baseline and excluded Vivienne
  outputs from QC validation.
- [x] Added validated Phase 2A QC-table loading and cross-table provenance checks.
- [x] Added GUI-independent dashboard summaries and readiness classifications.
- [x] Added ReviewSession progress and correction-burden summaries.
- [x] Added Napari/Qt dashboard dock with filters and explicit reasons.
- [x] Added `src.review_project_napari` launcher.
- [x] Added six dashboard tests and operator documentation.
- [x] Ran changed-file lint: all checks passed.
- [x] Ran full suite: 162 passed, 36 existing warnings.
- [x] Committed Phase 2B separately from Phase 2A.

### Later phases

- [x] Phase 2A headless QC engine/section selection.
- [x] Phase 2B project dashboard.
- [x] Phase 3 image review.
- [x] Phase 4 fiber-type reviewer integration.
- [x] Phase 4.5 guided-review UX consolidation.
- [x] Phase 4.6 reviewer navigation, recovery, and stain-aware display.
- [x] Phase 4.7 responsive display and review ergonomics.
- [x] Phase 5 region review.
- [x] Phase 5.1 named analysis ROIs and subregions.
- [x] Phase 6 nuclear review.
- [x] Phase 6.1 add nuclei
- [ ] Phase 7 fiber segmentation review.
- [ ] Phase 8 blinded review 
- [ ] Phase 9 finalization/reporting.

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
16. **Split Phase 2 into 2A headless generation and 2B dashboard.** This keeps
    expensive cohort computation out of interactive handlers and gives the later
    GUI a versioned, tested file contract.
17. **Do not reuse legacy `QCConfig` thresholds as project-review defaults.**
    `src.quantify_classify.QCConfig` contains fixed development-era thresholds
    for label count, unknown rate, median area, and marker correlation. Phase 2A
    emits corresponding evidence where possible but does not silently promote
    those thresholds into cohort acceptance rules.
18. **Add optional explicit per-image `applicable_domains`.** Artifact presence
    supplies a backward-compatible default, while explicit applicability
    distinguishes “not run by design” from “expected output missing.”
19. **Use mask-derived area and identity as the object-QC authority.** Prediction
    tables enrich those rows. Table-only IDs are reported through mismatch
    reasons rather than represented as nonexistent mask objects.
20. **Use coarse rule-derived scores, not opaque learned QC scores.** Phase 2A
    separates disposition (`technical_quality_score`) from work ordering
    (`review_priority`) and documents both formulas.
21. **Defer region/tile and raw-signal metrics to a calibrated follow-up.**
    Phase 2A does not have a stable tissue-mask or channel-quality artifact
    contract, so inventing those values would be scientifically misleading.
22. **Use `e3_exports/ta_emhc_baseline_v1_2026-07-29` as the authoritative
    frozen eMHC/DAPI nuclei baseline.** Do not use
    `phase5_vivienne_dapi`; those artifacts may originate from a different
    pipeline. The dashboard consumes project/QC tables and does not encode either
    user-specific SSD path.
23. **Keep Phase 2B read-only.** Strategy changes are provisional in memory.
    Persisted review decisions, manual section selection, and image navigation
    require the Phase 3 session/action contract.
24. **Keep the dashboard model Qt-free.** Qt/Napari imports live only in the
    widget/launcher boundary so cohort logic remains headlessly testable.

## 13. Known limitations

- The current reviewers remain separate applications until Phases 3–4.
- Current reviewer CSV saves are non-atomic until those UIs use shared storage.
- Current fiber-mask edits do not record lineage or stale dependencies.
- Split, merge, and undo for fiber segmentation are not implemented.
- Nuclear review currently supports copy-on-write addition/deletion and
  independent association decisions; split/merge nucleus-mask editing remains
  deferred.
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


# Bug fixes / UI fixes

- [x] Channel map initializes compactly and keeps a fixed maximum width.
- [x] Clarify automatic saving with a visible autosave/decision-count status.
- [x] Make the navigator a labeled, prominent review-navigation bar.
- [x] Show hotkeys on their corresponding primary action buttons.
- [x] Label the collapsed review-strategy and uncommon-action controls as
  **Advanced review options**.
- [x] Restore the saved standard queue and its position when Guided Review
  reopens; cover that behavior with a widget smoke test.
