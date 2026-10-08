# Changelog

## Unreleased

### Added

- **Model improver:** `python -m fibertypeqc improve PROJECT` turns your review decisions into
  candidate models. It recomputes the current model's features on the reviewed sections' existing
  masks, evaluates a logistic, a random-forest, and a gradient-boosting candidate on held-out mice
  beside the current model's own calls, and writes each candidate with a manifest under
  `PROJECT/models/improve_<time>/`. A declared recipe (`manifests/improver/recipe.v1.yaml`) fixes
  the candidates and the promotion rule; the command recommends a candidate only if the rule is
  met and never changes the active model.
- **Explicit promotion and rollback:** `python -m fibertypeqc promote PROJECT CANDIDATE` switches a
  project to a candidate from `improve`. It refuses a candidate the improver did not recommend
  unless `--override` is given (recorded), archives the current outputs and review state,
  re-types every section on its existing mask, restates review decisions against the new calls
  without changing any reviewer label, rebuilds QC, and logs the change in
  `models/model_history.csv`. `promote PROJECT --rollback` restores the previous model's outputs,
  QC, and review state exactly.
- `--model` accepts an unregistered model file when a manifest of the same name beside it records
  the file's SHA-256 (how candidates are run).
- **Blind reference labelling:** `python -m fibertypeqc review PROJECT --blind --reviewer NAME`
  opens the reviewer with model calls, confidence, QC reasons, the call overlay, and the cohort
  dashboard withheld. Labels are saved per reviewer under `review/reference/<reviewer>/`, apart
  from review decisions, and are never applied at finalization. Plans: every fiber in the current
  section, or a reproducible random sample.
- **Reference fields:** in blind mode, draw shapes in the `review_analysis_rois` layer and choose
  **Label every fiber in my drawn fields**; the shapes are saved as reference fields and every
  fiber whose center is inside one is queued. Random-sample settings are saved, so reopening
  rebuilds the same sample.
- **Reference export:** `python -m fibertypeqc reference PROJECT` writes `reference_export/`
  with one table of all reviewers' blind labels (model output kept in separate `model_*`
  columns), field coverage (which drawn fields are exhaustively labelled), reviewer agreement
  (percent and Cohen's kappa on shared fibers), review time, and a manifest with file digests.
- **Review time:** every review session (blind or guided) writes a row to `review_sessions.csv`
  beside its review state with decision count and active seconds; gaps longer than two minutes
  are not counted, and the cutoff is recorded.
- The reviewer only offers the classes the project's model produces (a three-class project no
  longer shows a Type I button).

## v0.3.0

### Release summary

- Project-folder workflow (`python -m fibertypeqc`) from images to a cohort report, with
  project-based review and finalization that never overwrites model calls.
- Models are panel-specific and distributed as release downloads: `quad_four_class_rf_v1`
  (default, four-marker panels) and `ta_three_class_logistic_v1` (three-marker panels). Neither
  has been validated on a random hold-out sample; their model cards make no accuracy claims.
- A demo data set and walkthrough (`docs/demo.md`).
- Research and study tooling in `research/` remains experimental and outside the supported
  surface.

### Added

- **Demo:** `docs/demo.md` walks through run, review, finalize, and the report on four cropped
  fields distributed as a release download, with a finished example project and a reference
  snapshot.
- **Reviewer workspace:** panels can be hidden but no longer deleted, so the Workspace menu always
  brings them back; "Restore review workspace" returns to the starting panels, layer toggles, and
  zoom; opening Region or Nuclei review no longer reloads the image; the cohort dashboard opens
  as a tab; the advanced-options toggle shows its label and the color legend sits with the
  decision buttons.
- **Reviewer:** an **Exclude (X)** button beside the type buttons; the view opens on the raw stain
  channels with every fiber outlined in the color of its current call (model call or review
  decision), with a legend in the dock. Fibers cut off by the image edge are left out of the
  flagged queue and labelled in section review. "Finalize and build report" stays visible
  after a review is started or resumed (it was hidden with the plan box).
- **Opt-in edge exclusion:** `exclude_image_border_fibers: true` in `fibertypeqc_project.yaml` (or
  `finalize_review_project --exclude-image-border-fibers`) excludes fibers touching the image
  edge at finalization (`exclusion_reason` `edge_of_image`, counted as `n_excluded_edge_of_image`
  and shown in the report). Off by default; intended for cropped fields.
- **Three-marker model:** `ta_three_class_logistic_v1` (IIa, IIb, residual IIx) is registered for
  laminin/IIa/IIb panels, with pinned feature extraction (tile background subtraction). The file
  is distributed outside Git and verified by SHA-256; the model card
  (`docs/model_cards/ta_three_class_logistic_v1.md`) makes no accuracy claims. The pipeline now
  measures center/edge marker features whenever a model declares them (previously only with
  `--export-diagnostics`).
- **Reference snapshots:** `python -m fibertypeqc snapshot PROJECT --freeze|--check` records
  per-section fiber counts, class counts, and digests of masks and fiber calls, and checks a later
  run against them: identical masks must give identical calls; re-segmented sections must agree
  within stated tolerances.
- **Finalize from the reviewer:** a "Finalize and build report" button in the Guided Review dock
  and Workspace menu applies the current review state, writes finalized tables and summaries, and
  opens the report (outputs follow the project-folder layout).
- **`python -m fibertypeqc init`** creates a project folder: it asks for (or takes as options) the
  images folder, a panel preset or per-stain channels, and the model, and writes the config,
  `panel.yaml`, and a `samples.csv` template (`--mouse-id-pattern` fills mouse IDs from file names).
  `prepare` and `status` say plainly when `mouse_id` still needs filling in.
- **Panel presets** in `manifests/panels/` (`four_marker_i_iia_laminin_iib`,
  `three_marker_iib_iia_laminin`); a project config may name a preset directly. The example
  four-marker panel now enables IIx inference, matching its paired example model.
- **Project folders and one command:** `python -m fibertypeqc {status,run,prepare,review,finalize}
  PROJECT/` runs the workflow from a folder containing `fibertypeqc_project.yaml` (images folder,
  panel, sample sheet, model). Outputs go to `batch/`, `review/`, `final/`, and `results/` inside
  it; `status` reports the current stage and the next command. The `scripts.*` commands are
  unchanged underneath.
- Research scripts, options, and labels that named a specific private cohort now use the neutral
  name "cohort B" (`cohort_b`, `--cohort-b-…`). Research tooling only; the supported pipeline is
  unaffected.
- README rewritten around the release workflow (requirements, install check, inputs, end-to-end
  steps, outputs, models, QC, troubleshooting); `docs/quickstart.md` now covers single-image and
  historical commands; `ARCHITECTURE.md` and `AGENTS.md` describe the supported workflow; the
  historical model card is marked retired; `manual8_myo3_base0p1` is retired in the registry.
- Project QC measures laminin on and just inside each fiber outline and adds two review-only
  reasons, `fiber.weak_laminin_rim` and `fiber.thick_laminin`. They join the flagged review queue
  and "Why shown" now names the specific reason. No fiber is excluded and no call changes.
- `--model` accepts a path to a registered model file as well as a model ID; the file is identified
  by SHA-256, so a private model needs no `FIBERTYPEQC_MODEL_ROOT` and may have any file name.
  `run_batch` resolves the model once before processing (one clear error instead of one per image),
  logs the model in use instead of the legacy v0 parameters, reports the scene count per CZI, and
  shows the child's error line rather than its usage text. `--show-v0-params` works without a
  model or panel.
- Finalization and the cohort report label sections with no review activity `not_reviewed`
  (no warning); `unverified` is reserved for sections with decisions but no recorded fingerprints.
- Cohort dashboard rows are readable in dark themes (dark text on status colors).
- `generate_review_qc` prints per-image progress and is faster on large sections (vectorized
  probability metrics and dictionary lookups; outputs unchanged).
- **One command path:** `scripts.make_review_project` builds a review project from `run_batch`
  output and a sample sheet (split CZI sections expanded, conditions carried, failed images
  skipped, model version from run records, existing projects never overwritten). Public wrappers
  `scripts.generate_review_qc` and `scripts.review_project_napari` added. Finalization works before
  any review (all fibers keep predictions, with a warning). README documents the end-to-end path;
  `README_review_workflow.md` documents project-based review, with the per-image reviewer marked
  legacy.
- **Results layer:** `scripts.summarize_results` writes image, mouse (pooled sections), ROI, and
  cohort (mouse as the unit, grouped by project conditions) tables from finalized outputs, a
  results manifest, and a self-contained `cohort_report.html` built only from those tables. It
  shows predicted versus finalized composition per mouse, review and exclusion burden, and source
  table digests. Unresolved fibers are reported separately and excluded from composition
  denominators.
- **Review finalization:** `scripts.finalize_review_project` turns a reviewed project into
  `<image_id>_fibers_finalized.csv`, `final_fiber_table.csv`, and `finalization_manifest.json`.
  It applies image/section, region (by centroid), and fiber decisions with exclusion precedence;
  keeps every model column beside `final_type` and `value_source`; tags analysis ROIs; and refuses
  decisions whose label mask or fiber table changed since review (review sessions now record input
  fingerprints when an image is opened). Legacy per-image review CSVs import through
  `--legacy-review`, including files without a model-prediction column. `merge_reviewed_labels` is
  deprecated. Previously project-review decisions and region/section exclusions were saved but
  never applied to any output.
- **Default model change:** `run_batch` without a model option now uses the registry default,
  `quad_four_class_rf_v1` (four-class I/IIa/IIb/residual IIx), and requires `--panel-config`.
  The historical three-class run is `--model rebaseline_tile_v2_p75p90_iib_iia_iix`. Before the
  switch, the QUAD model passed software reproducibility checks (model card:
  `docs/model_cards/quad_four_class_rf_v1.md`). Its biological accuracy has not been validated on
  a random hold-out sample; the model card makes no accuracy claims.
- Model selection by ID: `run_pipeline --model` / `run_batch --model` resolve a registered model's
  manifest and artifact. Privately distributed artifacts are read from `FIBERTYPEQC_MODEL_ROOT` and
  verified by the digest recorded in the public registry (`artifact_location: private`).
- Model manifests may declare ordered `features` and pin every feature-affecting setting in
  `feature_extraction`. With such a model the pipeline reproduces training-time extraction, skips
  the `--sensitivity` profile, and refuses conflicting typing flags.
- Semantic models may be bare fitted estimators; feature count, feature order (for estimators fitted
  with names), classes, and finite inputs are checked. For `fiber_identity` models their predictions
  are now the fiber calls in `*_fibers.csv` (previously a sidecar only).
- Registry entries `quad_four_class_rf_v1` (private artifact, neutral ID, digest) and
  `synthetic_four_class_reference_v1`; the historical three-class model is marked `retired`.
- Public synthetic four-class reference (`examples/reference_four_class/`, generated by
  `scripts.generate_four_class_reference`) run by `scripts.run_reference` and CI.
- `ROADMAP.md` as the single public execution plan (release overhaul, Stages 0–7); the 2026 H2
  roadmap moved to `docs/history/`.
- Tracked `AGENTS.md` contributor rules.
- Supported release surface map in `ARCHITECTURE.md`.
- Private-data guardrails in `scripts.check_repository`: unapproved model artifacts, per-fiber tables
  outside `examples/`/`tests/`, external-drive and cloud-backup paths, and an optional private
  identifier denylist kept outside Git.
- `.pre-commit-config.yaml` running repository checks and ruff before each commit.

### Changed

- Research and study-evaluation modules moved from `src/` to `research/` (76 modules). Run them as
  `python -m research.<module>`; `validation/` wrappers and cluster scripts were updated. The
  supported core in `src/`, `fibertypeqc/`, and `scripts/` is unchanged, and a test now prevents it
  from importing `research/`.
- `pytest` works without setting `PYTHONPATH`.
- **Output change:** image summaries now report `prop_*` and confidence intervals for the classes
  the model outputs (e.g. `iia`, `iib`, `iix`). Previously model runs reported legacy
  `type1`/`type2`/`mixed`/`unknown` columns that were always 0. Fiber calls are unchanged.
- **QC change (schema `fibertypeqc.qc.v2`):** `postrun.unknown_rate` is replaced by
  `postrun.uncertainty_rate` (low-confidence or unresolved fibers of any class) and a panel-aware
  `postrun.residual_rate` (share of the inferred-by-absence class, informational until
  `--qc-max-residual-rate` is calibrated). `--qc-max-unknown-rate` remains as an alias for
  `--qc-max-uncertainty-rate`. The reference validator now checks summary proportions and QC rates.
- Explicit typing flags that `--sensitivity`/`--mixed-strictness` override now produce a warning
  and a `preflight.typing_flags_overridden` QC check with the effective values. Effective values
  are unchanged.
- `run_pipeline --image-id` names outputs explicitly. `run_batch` uses it instead of renaming files
  after each run, which left result bundles, summaries, and reports pointing at old file names.
- `run_batch` passes the frozen model manifest (or `--model-manifest`) so the classifier digest and
  panel requirements are verified in batch runs; failure messages keep the end of the child error;
  a failed scene export is recorded as `scene_export_failed` instead of aborting the batch; duplicate
  image IDs are rejected.
- Run provenance (run-manifest schema 2) records the input image, provided-labels, and classifier
  SHA-256 digests; the Cellpose device actually used; and numpy/scipy/scikit-image/scikit-learn/
  pandas/tifffile/czifile versions. Paths in run records, summaries, QC context, and the fiber
  table's `classifier_path` are portable (no user-specific directories). The label-reuse
  fingerprint now includes the image digest, labels source, device, and Cellpose version; existing
  cached labels are recomputed once.
- **Input change:** images are read using their axis metadata. Multi-scene CZIs now stop with
  instructions to use the scene splitter; previously only the first scene was silently analyzed.
  Z/T stacks and ambiguous unlabeled axes also stop instead of taking the first plane or guessing.
- **Input change:** CZI mosaics are assembled single-threaded. Multi-threaded assembly wrote
  overlapping tiles in nondeterministic order, so repeated reads of the same file could differ in
  overlap pixels; reads are now repeatable and match the scene splitter's tile order.
- README corrections: Python 3.11 only; per-image outputs match what the pipeline writes; removed
  the nonexistent low-coverage QC flag, the `--bsize` troubleshooting advice (Cellpose requires 256),
  and the reference to an untracked training script; documented `run_batch` v0 thresholds and the
  `--sensitivity` override of explicit typing flags.

## v0.2.0

### Release Title

`FiberTypeQC v0.2.0 — panel config, diagnostics export, and baseline safeguards`

### Added

- Panel-aware channel config foundation with YAML-based channel/schema support.
- Legacy alias compatibility for `type1 -> iib` and `type2 -> iia`.
- Public CLI support for explicit marker naming in pipeline, review, and batch workflows.
- Output provenance fields such as `fiber_type_source` and `available_markers`.
- Internal support for optional extra-marker stats (`i`, `iix`) without changing default typing.
- Optional diagnostics export via `--export-diagnostics`, writing `*_feature_diagnostics.csv`.
- Feature-set comparison tooling:
  - `src/compare_feature_sets.py`
  - `validation/compare_feature_sets.py`
- `docs/modeling.md` for frozen-model feature contract, feature gaps, and experimental directions.
- Expanded frozen alpha model card.
- Baseline regression guards for the frozen alpha feature contract and legacy rule path.
- Pytest split between:
  - fast synthetic default tests: `python -m pytest -m "not integration"`
  - optional integration tests: `python -m pytest -m integration`

### Changed

- Public documentation now distinguishes:
  - stable `*_fibers.csv` biological/review output
  - optional `*_feature_diagnostics.csv` model-development/debugging output
- Batch runs preserve the frozen alpha baseline by default, while logging clearly when channel/config
  overrides move a run outside the strict baseline path.

### Not Changed

- Default classifier/model file
- Default thresholds
- Default feature contract used by the frozen alpha model
- Default channel assumptions for the alpha path
- Default erosion/preprocessing behavior
- Review merge logic

### Release Notes

`v0.2.0` is a workflow and architecture hardening release. It improves configuration, provenance,
diagnostics, documentation, and regression guardrails while keeping the stable alpha model behavior
conservative.

`FiberTypeQC remains alpha-stage research software. This release does not change the frozen default
classifier or claim final validation as a MyoSight replacement.`
