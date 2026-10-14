# FiberTypeQC Roadmap

- Updated: 2026-10-14
- Status: **frozen.** v0.3.0 is released; work after it is listed below and paused. The next step
  is an audit of the repository for a software-paper (JOSS) submission, which will replace this
  plan.
- Supersedes: [docs/history/ROADMAP_2026H2.md](docs/history/ROADMAP_2026H2.md) and earlier planning notes.

Study-specific planning (cohorts, private evidence, model-selection history) is kept outside Git.
This file contains only public-safe scope, decisions, and status.

## Goal

A reproducible, interpretable FiberTypeQC release in which another researcher can run the full chain
on a compatible image and reproduce its outputs:

```text
image -> segmentation -> panel-specific prediction -> QC -> guided review
      -> finalized analysis-ready tables -> image/mouse/cohort report
```

The release validates the workflow, not a universal classifier.

## Decisions

- **One model per panel.** Fiber-type models do not generalize across channel configurations. Each
  released model declares the panel (marker roles), ordered features, and classes it supports, and
  the pipeline refuses a mismatched panel.
- **Default model: four-class QUAD (Type I, IIa, IIb, IIx).** A TA model is deferred and will be
  rebuilt from its frozen evaluation contract when needed. The historical three-class model
  (`rebaseline_tile_v2_p75p90_iib_iia_iix`) remains reproducible but is not a release model.
- **Models trained on unpublished data are not committed.** The registry records identity and
  digest; the artifact is resolved at runtime from a private model root and verified by digest.
- **Release reviewer: the project-based Napari review** (`src/review_project_napari.py`,
  `src/review/`). The per-image reviewer is kept only as an input bridge.
- **Not a rewrite.** Separate the supported product core from research code, then close the gaps.

## Since v0.3.0 (on `main`, unreleased; frozen 2026-10-14)

Built and merged, each with tests; GUI parts checked by hand in napari:

- **Blind reference labelling** (`review --blind --reviewer NAME`): model calls, confidence, QC
  reasons, and section/mouse names withheld; labels stored per reviewer and never applied to
  results; random samples (150 per section by default), computer-placed fields, or drawn fields.
- **Reference export** (`reference`): all reviewers' labels with model output in separate columns,
  field coverage, reviewer agreement (percent, Cohen's kappa), review time.
- **Review time** recorded for every session.
- **Evaluation roles** (`evaluation_roles.csv`): pool and test mice kept apart in review, blind
  labelling, and model improvement.
- **Model improver** (`improve`): candidates (logistic, random forest, gradient boosting) from
  review decisions, evaluated on held-out mice against the current model under a declared recipe
  and promotion rule.
- **Promotion and rollback** (`promote`): explicit, archived, reversible switch to a candidate;
  review decisions carried forward.
- Reviewer fixes: Exclude button, fiber-call overlay, edge-fiber handling, reliable panels,
  shortcuts that work regardless of focus, plans that resume at the first unlabelled fiber.

Paused, not started: learning-curve runner, schema-driven classes and a "no model yet" state, a
custom-schema model generator, improve/promote buttons in the reviewer, leaving edge fibers out of
training labels.

## State at v0.3.0 (2026-10-03)

| Area | State |
|---|---|
| Ingest and segmentation | Axis-metadata loading; multi-scene CZIs split deterministically; device and digests recorded |
| Typing | Panel-specific models selected by ID or file; pinned feature extraction; fail-closed panel check |
| Per-image summary and QC | Class proportions from the model's classes; uncertainty and residual-class checks |
| Batch | Model verified before processing; outputs named by image ID; portable paths |
| Model registry | Features, classes, digests, and private-artifact entries; default model declared |
| Project review | Cohort QC, section status, guided fiber review, regions and analysis ROIs; laminin review prompts |
| Finalization | Review decisions and exclusions applied into finalized tables; predictions never modified |
| Results | Image, mouse, ROI, and cohort tables; cohort report built only from exported tables |
| Reproducibility infrastructure | Locked environment, CI, synthetic references run end to end, private-data guardrails |
| Default model validation | Software reproducibility checks pass; random hold-out validation not yet done |
| Default model distribution | Release downloads (v0.3.0), verified by SHA-256 |

## Stages

Each stage is one branch and pull request. Stages marked ⚗ change scientific outputs: each change is
explained before editing and accompanied by a before/after comparison of reference outputs.

- [x] **Stage 0 — Guardrails and plan.** Private-data checks in `scripts.check_repository`
  (unapproved model artifacts, per-fiber tables outside fixtures, private paths, optional private
  identifier denylist), pre-commit hook, this roadmap, tracked `AGENTS.md`, supported-surface map in
  `ARCHITECTURE.md`.
- [x] **Stage 1 — Carve the core** (no behavior change). Declare the supported modules; move
  research modules behind a `research/` namespace with compatibility wrappers; fix README drift.
- [x] **Stage 2 — Core correctness** ⚗. Summary classes and QC residual rate from the model's class
  list; warn when `--sensitivity` overrides explicit typing flags; batch path/digest fixes; record
  device, library versions, and input digests; explicit image scene/axis handling; extend reference
  validation to summaries and QC.
- [x] **Stage 3 — Four-class QUAD default** ⚗. Extended model manifest and registry; semantic path
  becomes the primary typing path and supports any sklearn estimator family; fail-closed panel
  compatibility; runtime model resolution with digest verification; public synthetic four-class
  reference; historical models marked retired.
- [x] **Stage 4 — Finalization** ⚗. One finalizer that applies review decisions and region/section
  exclusions, labels every value as predicted, reviewed, excluded, or unresolved, and refuses
  decisions made on different labels or tables.
- [x] **Stage 5 — Results layer.** Image, mouse, and cohort tables from finalized outputs; cohort HTML
  report including predicted-versus-finalized comparison. Every figure reproducible from exported
  CSV/JSON.
- [x] **Stage 6 — One command path and usability.** Documented entry points from batch run through
  QC, review, finalization, and report; first hands-on usability pass on a multi-mouse project on
  an HPC GUI, with its fixes merged.
- [x] **Stage 7 — Release.** Models distributed as release downloads; private reference run with
  frozen expected outputs; demo data and walkthrough; v0.3.0 tagged. Still open: a clean-checkout
  rehearsal by a non-author following the README only.

Order: `0 → 1 → 2 → {3, 4} → 5 → 6 → 7`. Reference outputs are frozen only after Stage 2.

## After this release

- **Validation:** blinded review of a random hold-out sample for the default model; until then its
  model card makes no accuracy claims.
- **Adapt-to-your-panel workflow:** create project → map channels → run a baseline → review/label a
  subset (including a blind random sample) → train a panel model with pinned features and a recorded
  evaluation → rerun → finalize/report. This is what makes the tool usable on panels without a
  published model.
- **Usability:** one project-folder convention, an init wizard for panel and sample-sheet files,
  a `status`/next-action command, a finalize-and-report button in the reviewer, panel presets;
  later a GUI channel mapper, sample-sheet editor, and napari plugin.
- **Install:** split GUI and research dependencies into optional extras.
- **Sections:** splitting separate tissue pieces imaged within one Zeiss scene.
- Optional: rebuilt TA model, OCI/Apptainer images, shared TA+QUAD model research.

## Verification for every stage

```bash
uv run ruff check .
uv run pytest -m "not integration" -q
uv run python -m scripts.run_reference
uv run python -m scripts.check_repository
```

## Completion criteria

- A non-author reproduces the declared workflow from a clean checkout without undocumented files or
  developer knowledge.
- Every released model and expected output is tied to immutable identifiers and digests.
- Users can distinguish predicted, reviewed, excluded, unresolved, and finalized values, and the
  report shows results before and after review.
- Review produces finalized outputs without altering predictions.
