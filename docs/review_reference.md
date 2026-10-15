# Review Reference

Reference for the project reviewer's components: project QC, the cohort dashboard, image
review, guided fiber review, regions, and nuclei. For the workflow itself see the
[review guide](review_guide.md). Section titles name the development phase in which each part
was built; the design record is in [history/qc_manual_review_system.md](history/qc_manual_review_system.md).

## Headless Review QC (Phase 2A)

Phase 2A generates versioned technical-QC evidence and domain-specific section
selections without importing Napari or Qt. It does not implement the dashboard
and does not change prediction outputs.

The detailed formulas, denominators, missing-input behavior, and design decisions
are controlled by
[`docs/history/qc_manual_review_system.md`](history/qc_manual_review_system.md).

### Project artifact contract

Each image declares its expected domains and prediction artifacts:

```yaml
schema_version: review_project.v1
project_id: example
project_name: Example review project
panel_manifest: panel.yaml
model_version: model.v1
images:
  - image_id: mouse_1_section_1
    mouse_id: mouse_1
    section_id: section_1
    raw_image_path: inputs/mouse_1_section_1.czi
    prediction_directory: predictions/mouse_1_section_1
    applicable_domains:
      - fiber_segmentation
      - fiber_typing
      - nuclei
    outputs:
      fiber_labels: mouse_1_section_1_cellpose_labels.tif
      fiber_table: mouse_1_section_1_fibers.csv
      nuclei_labels: nuclear/mouse_1_section_1_nuclei_labels.tif
      nuclei_table: nuclear/mouse_1_section_1_nuclei.csv
```

Output paths are relative to `prediction_directory`. If
`applicable_domains` is omitted, fiber segmentation is applicable by default;
fiber typing is added when `fiber_table` is declared, and nuclei is added when a
nuclear label or table artifact is declared.

This distinction matters:

- an expected but missing artifact produces an explicit technical failure;
- an artifact omitted because its domain is not applicable produces
  `not_applicable`, not a failure.

### Generate QC

```bash
uv run python -m src.generate_review_qc \
  --project project.yaml \
  --selection-strategy all_passing
```

Optional arguments:

- `--rules configs/review_qc_rules.v1.yaml`
- `--output-dir path/to/qc`
- `--selection-strategy all_passing|best_passing|manual`
- `--manual-selection manual_selection.yaml`

Manual-selection YAML uses explicit manifest IDs:

```yaml
mouse_1:
  fiber_segmentation:
    - mouse_1_section_1
  fiber_typing:
    - mouse_1_section_2
  nuclei:
    - mouse_1_section_1
```

A manually selected section must be applicable and free of technical hard
failures. Empty manual selections are retained as
`manual_selection_required`.

### Outputs

The command writes:

- `qc/image_qc.csv`: one image/domain row with metrics, status, scores, explicit
  reasons, and provenance;
- `qc/fiber_qc.csv`: one row per positive fiber-mask object. When the panel's laminin channel and
  the raw image are available it includes `laminin_rim_mean` (laminin on the fiber's outline) and
  `laminin_inner_band_mean` (laminin 3–7 px inside the outline), and `technical_reason_codes` may
  contain `fiber.weak_laminin_rim` (outline below the section median by more than 3 MADs) and
  `fiber.thick_laminin` (inner-band laminin more than 4 robust SDs above the section median).
  These are review prompts only: they queue the fiber for review and never exclude it or change
  its type. Both are relative to the section, so a small share of ordinary fibers is always
  flagged;
- `qc/nucleus_qc.csv`: one row per positive nucleus-mask object;
- `qc/section_selection.csv`: one mouse/domain result for the selected strategy.

`all_passing` means all applicable sections without a hard technical failure.
This intentionally includes sections with REVIEW findings. `best_passing`
prefers the highest `technical_quality_score`, then the lowest
`review_priority`, then manifest order. If no section is eligible, the result
contains no selected ID and `requires_manual_review=true`.

### Score interpretation

`technical_quality_score` is a transparent disposition:

- PASS: `1.0`
- REVIEW: `0.5`
- hard FAIL: `0.0`
- not applicable: null

It is not a calibrated probability and contains no biological outcome.

`review_priority` only orders work:

```text
100 * hard-fail reason count + 10 * review reason count
```

Informational reasons do not change either status or priority.

### Default rules

The default configuration is
[`configs/review_qc_rules.v1.yaml`](../configs/review_qc_rules.v1.yaml).

Enabled hard failures are restricted to structurally unusable output:

- required artifact missing or unreadable;
- invalid label-mask dimensionality/type/value;
- no positive fiber objects;
- empty applicable typing table;
- missing typing prediction column;
- fiber/nucleus label-shape mismatch.

Enabled REVIEW rules cover object-ID inconsistencies, invalid/duplicate IDs, and
an empty nucleus mask. Missing probability or optional association fields are
informational.

No default threshold is enabled for:

- segmented or border-touching fractions;
- fiber or nucleus area;
- confidence, margin, or entropy;
- unknown or `needs_review` fractions;
- fiber-type composition;
- nucleus density or nuclei-per-fiber;
- unassigned or ambiguous association fractions;
- central nuclei or other biological outcomes.

These values require panel/model/cohort calibration. To activate one, create a
new versioned rule file, supply an explicit threshold and rationale, set
`enabled: true`, and retain the new `rules_version` in exported QC.

### Current metric boundary

Phase 2A reads label TIFFs and object CSVs only. It does not currently compute:

- tissue-normalized coverage;
- focus or channel dropout;
- saturation;
- membrane support;
- DAPI support;
- tile/region heatmaps;
- cohort percentile or MAD rankings.

Those require stable raw-signal/tissue-mask input contracts and calibration.
They must not be approximated from biological endpoint values.

The Phase 2B cohort dashboard that consumes these outputs is documented in
[Cohort Dashboard](#cohort-dashboard-phase-2b).

## Cohort Dashboard (Phase 2B)

Phase 2B is the read-only cohort/mouse dashboard for precomputed Phase 2A QC.
It does not rerun prediction, recompute QC in click handlers, or edit review
state.

### Prerequisite

Generate Phase 2A outputs first:

```bash
uv run python -m src.generate_review_qc \
  --project project.yaml \
  --selection-strategy all_passing
```

Then open the dashboard:

```bash
uv run python -m src.review_project_napari \
  --project project.yaml
```

Optional arguments:

- `--qc-dir path/to/qc`
- `--selection-strategy all_passing|best_passing|manual`
- `--manual-selection manual_selection.yaml`

The authoritative frozen eMHC/DAPI baseline is
`e3_exports/ta_emhc_baseline_v1_2026-07-29`.
`phase5 Type I/IIa/DAPI output` is not an accepted baseline and must not be used for
dashboard validation or QC calibration.

No user-specific SSD path is stored in source, configuration, or public
documentation.

### Dashboard contents

The dock always displays:

- project, cohort scope, current domain filter, strategy, and model version;
- mouse and section counts;
- complete, targeted-review, and no-acceptable-section mouse counts;
- review progress from an existing `review_state.json`;
- object-decision, region, and reviewed-mask burden;
- PASS / REVIEW / FAIL / not-applicable counts by domain;
- a mouse → section → domain QC tree;
- technical score, review priority, provisional selection, and reason codes;
- a details pane with rule evidence and available metrics.

The domain and status filters affect only the visible tree. They do not alter QC
or selection.

### Readiness definitions

An acceptable section is applicable and has no hard technical failure.

Mouse/domain readiness:

- `complete`: at least one PASS section and the selected section set contains no
  REVIEW section;
- `targeted_review`: selection requires review, no PASS section exists, or the
  provisional selection includes a REVIEW section;
- `no_acceptable_section`: every applicable section hard-failed;
- `not_applicable`: the domain is not expected for that mouse.

A mouse inherits `no_acceptable_section` if any applicable domain has no
acceptable section. Otherwise it inherits `targeted_review` if any applicable
domain requires it.

### Strategy switching

Changing strategy in the dashboard recomputes recommendations in memory:

- `all_passing` includes all sections without hard failures;
- `best_passing` prefers technical score, then review priority, then manifest
  order;
- `manual` displays the supplied manual selection or marks selection required.

Phase 2B does not persist strategy changes. Persisted section decisions and
opening an image for review belong to Phase 3.

### Input validation

The dashboard rejects incompatible schema versions, project/model mismatches,
unknown image IDs, duplicate or missing image/domain rows, and malformed boolean
fields. This prevents silently displaying stale or unrelated QC as current.

`image_qc.csv` is required. Object QC and stored section-selection tables are
optional because the cohort summary can be reconstructed from image QC.

### Current limitations

- No raw image or label layers are opened.
- No status, exclusion, or selection action is saved.
- No image, region, or object review controls are exposed.
- No signal-aware Laminin/DAPI/eMHC metrics are added by the dashboard.
- Multi-user review and live filesystem refresh are not supported.

## Image-Level Review (Phase 3)

`src.review_project_napari` now opens the existing cohort dashboard alongside an
image-review dock and the selected raw-image channels. Phase 3 records only
image/domain decisions; it does not edit masks, regions, or individual fibers.

```bash
uv run python -m src.review_project_napari --project project.yaml --reviewer "name"
```

Generate and validate Phase 2A QC before launching. The project manifest paths
are validated when the launcher starts because raw images are opened in this
slice.

The image-review dock retains the current image, active domain, mode, and
position in `review/review_state.json`. Selecting Pass, Review, Fail, or Exclude
immediately saves that snapshot and appends a `set_domain_status` event to
`review/review_events.csv`. Prediction artifacts remain read-only.

The modes control the navigation set for the active domain:

- `qc_gated_automatic`: images with QC REVIEW or FAIL;
- `flagged_review`: QC REVIEW or positive review priority;
- `domain_focused`: every applicable image in the chosen domain; and
- `full_audit`: every applicable image in the chosen domain.

The difference between the last two modes is intentional at this phase: both
provide complete image navigation. Their object-level sampling policies arrive
with Phase 4. A domain marked not applicable by the project manifest cannot be
given a conflicting image-level status.

## Fiber-Type Object Review (Phase 4)

The project Napari launcher opens a focused **Guided Review** dock for
fiber-type adjudication. It reads the project manifest's `fiber_table` artifacts
and never modifies them. The cohort dashboard and full image controls remain
available on demand as advanced docks.

```bash
uv run python -m src.review_project_napari --project project.yaml --reviewer "name"
```

For large sections, use display-only downsampling:

```bash
uv run python -m src.review_project_napari \
  --project project.yaml \
  --display-downsample 2
```

`1` is full resolution; `2` is the recommended first smoke-test setting, and
`4` is useful for very large sections. Raw and label layers are sampled
together, preserving fiber IDs; the original prediction artifacts are never
rewritten.

Available queue sources are model/QC flags, low confidence, high entropy, low
probability margin, near-tied probability conflicts, a deterministic random
audit, and full audit. Each decision stores the model type separately from the
reviewed type, decision status, queue source, reason, reviewer, and timestamp
in `review/review_state.json`; its audit event is appended atomically to
`review/review_events.csv`.

Guided Review visibly reports its autosave state and decision count. It resumes
the last saved standard queue and position when the project is reopened; starting
a different plan deliberately replaces that position. A new session simply shows
the flagged-fiber count without starting a queue.

The default random audit is cohort-scoped, uses seed `0`, and selects up to 25
fibers. The headless queue API also supports image- and mouse-scoped samples,
explicit seeds, and sample sizes for reproducible study protocols.

### Guided review workflow

Start by choosing **Review flagged fibers**, **Review this section**, or **View
cohort QC**. The prominent **Navigate review** bar switches to the cohort,
current section, and other domains. After selecting a plan, the dock shows one
fiber at a time with a short context line and plain-language primary action
**Keep model call (K)**. Type corrections and navigation buttons show their
hotkeys directly. Advanced queue controls and uncommon actions are grouped under
the labeled **Advanced review options** control. Saved decisions automatically
advance to the next fiber; **Undo last decision** restores the immediately
preceding value and records a corresponding audit event.

The selected fiber is a thick cyan outline only, so it does not obscure stain signal.
The visible shortcut legend is: Left/Right to navigate, `K` to keep the model
call, `1`/`2`/`3`/`4` for I/IIa/IIb/IIx, `F` to center the current fiber, and
`U` to undo the immediately prior decision.

The persistent navigator moves between cohort QC, the current section, and
domain-specific review. If a dock is closed accidentally, reopen it from the
Napari **Workspace** menu; **Restore review workspace** returns the Guided
Review dock and channel map without restarting. Raw data opens as individually
toggleable, color-named stain channels (a fixed composite layer is available, hidden),
with fibers outlined in the color of their current call.
choices do not modify predictions or review data.
The compact, fixed-width Channel Map identifies each role and source channel.
These display choices do not modify predictions or review data.
choices do not modify predictions or review data.

The legacy `review_labels_napari` CSV layout remains readable through the
`load_legacy_fiber_type_decisions` adapter. It is a migration input only; the
project session remains the canonical record. Region review, segmentation edits,
and masks are not part of this phase.

## Region Review (Phase 5)

Open the project workspace and choose **Regions** in the persistent navigator,
or use **Workspace → Show Region Review**. The image receives a yellow
`review_region_shapes` layer and a translucent `review_region_coverage` layer.
The coverage layer counts only saved reviewer polygons; it is not a QC score or
biological measurement.

1. Draw a polygon in the Region shapes layer and select it.
2. In Region Review, select the domain and an explicit action.
3. Optionally add a reason and notes, then choose **Apply action to selected
   shape**.

The action is autosaved to both `review/review_state.json` and
`review/review_regions.geojson`, and an audit event is appended to
`review/review_events.csv`. Use the saved-region list to remove a mistaken
annotation; removal is also recorded and rewrites the GeoJSON atomically.

The available actions record domain exclusion, object-queue, detailed-review,
ignore, rerun recommendation, and unresolved intent. They do not alter masks,
predictions, or finalized analysis in this phase; finalization applies their
meaning later. Region geometry is stored at full image coordinates even when
`--display-downsample` is active.

### Named analysis ROIs (Phase 5.1)

To draw anatomical subregions, such as four quadrants of a QUAD section, draw
the polygon in the editable yellow **Region shapes** layer, choose **Analysis
ROI**, and provide both an ROI name (for example, `quad_1`) and role (for
example, `quadrant`). Applying it autosaves the ROI, then renders it in the
separate orange `review_analysis_rois` layer.

ROIs remain geometry and metadata until finalization. The headless assignment
contract uses fiber/object centroids and reports one of `assigned`, `outside`,
`boundary`, or `ambiguous`; it never silently resolves overlapping ROIs or a
centroid on an ROI boundary. Finalization can use these values to emit
`region_id`, `region_name`, and `region_role` columns in reviewed/final CSVs.

## Nuclei Review (Phase 6.2)

Open a validated project in the shared Napari review workspace:

```bash
uv run python -m src.review_project_napari \
  --project project.yaml \
  --reviewer "name" \
  --display-downsample 1
```

Choose **Nuclei** in Guided Review, or use **Workspace → Show Nuclei Review**.
The selected image shows its configured raw-image channels (including DAPI/eMHC
when present), fiber labels, reviewed nuclei labels, and an empty **Draft new
nucleus** labels layer.

Use the dock as follows:

- Select a label in **reviewed nuclei labels**, click **Use selected nucleus**,
  then delete it or set its association.
- Paint only in **Draft new nucleus** and click **Save painted draft nucleus**.
  The draft must not overlap an existing nucleus. It receives a stable reviewed
  ID and begins with an unresolved association.
- Set an association status and, for `assigned`, a fiber ID; then click **Save
  association**.

Every action autosaves the review state, event log, reviewed nucleus mask, and
reviewed association CSV. Model artifacts are never modified. Nucleus-mask
edits mark nucleus features, associations, and fiber-level nucleus counts stale;
association-only edits mark the latter two stale. Paint additions require
`--display-downsample 1` so the draft coordinates match the saved label mask.
