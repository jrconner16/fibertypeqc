# FiberTypeQC Roadmap

- Updated: 2026-09-29
- Status: active. This is the single public execution plan.
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

## Current state (2026-09-29)

| Area | State |
|---|---|
| Ingest, segmentation, feature extraction | Works; image-axis/scene handling and device recording need hardening |
| Typing | Three-class path works; four-class semantic path is a sidecar only |
| Image summary / post-run QC | Class proportions and residual-rate QC assume legacy class names |
| Batch | Works; renamed outputs and model-digest checks need fixes |
| Model registry | Lacks ordered features, class list, training-ledger digest, code revision, private artifact URI |
| Project review | Built and tested, but its decisions do not yet feed any output |
| Finalization | Missing |
| Mouse/cohort summaries and cohort report | Missing (report is per-image) |
| Reproducibility infrastructure | Locked environment, CI, synthetic reference, result bundles, digests |

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
- [ ] **Stage 3 — Four-class QUAD default** ⚗. Extended model manifest and registry; semantic path
  becomes the primary typing path and supports any sklearn estimator family; fail-closed panel
  compatibility; runtime model resolution with digest verification; public synthetic four-class
  reference; historical models marked retired.
- [ ] **Stage 4 — Finalization** ⚗. One finalizer that applies review decisions and region/section
  exclusions, labels every value as predicted, reviewed, excluded, or unresolved, and refuses
  decisions made on different labels or tables.
- [ ] **Stage 5 — Results layer.** Image, mouse, and cohort tables from finalized outputs; cohort HTML
  report including predicted-versus-finalized comparison. Every figure reproducible from exported
  CSV/JSON.
- [ ] **Stage 6 — One command path and usability.** Documented entry points from batch run through
  QC, review, finalization, and report; usability pass on a large project.
- [ ] **Stage 7 — Rehearsal and release.** Clean-checkout rehearsal by the developer and then by a
  non-author; convert every blocker into documentation, automation, or a stated limitation; tag.

Order: `0 → 1 → 2 → {3, 4} → 5 → 6 → 7`. Reference outputs are frozen only after Stage 2.

Later and optional: rebuilt TA model, OCI/Apptainer images, shared TA+QUAD model research.

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
