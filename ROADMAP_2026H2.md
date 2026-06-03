# FiberTypeQC Roadmap (2026 H2)

Time window: July-December 2026  
Priority order: portfolio/job value first, reproducibility/citability second, optional software-note/JOSS path kept open.

## Context

- Public repo: `jrconner16/fibertypeqc`
- Current tags: `v0.1.0-alpha`, `v0.1.1-alpha`
- Current public workflow:
  1. `scripts.run_pipeline`
  2. `scripts.run_batch`
  3. `scripts.review_labels_napari`
  4. `scripts.merge_reviewed_labels`
- Frozen alpha default model: `data/models/rebaseline_tile_v2_p75p90_iib_iia_iix.joblib`
- Current channel schema:
  - membrane/border for segmentation
  - `type1` = IIb
  - `type2` = IIa
  - IIx inferred from absent IIb/IIa signal

## Target Architecture Direction

- Move from a fixed `type1/type2` typing model to a panel-aware channel schema.
- Distinguish three layers explicitly:
  - observed marker channels
  - derived biological calls
  - feature extraction / review metadata
- Treat `I`, `IIa`, `IIb`, and `IIx` as optional direct marker channels.
- Treat `membrane` as required for segmentation and `dapi` as optional for nuclear features.
- Treat inferred classes as decision outputs, not as channels.
- Record whether a final fiber call came from:
  - direct marker evidence
  - residual inference from an omitted class under an allowed panel
  - unresolved/other/uncertain logic
- Support arbitrary 4-channel panel combinations at the measurement layer, but only allow biological class claims justified by the configured panel.

## Status Snapshot (2026-06-02)

- `v0.1.1-alpha` released as pre-release with demo screenshots and documentation polish.
- Roadmap milestones created on GitHub: `v0.2`, `v0.3`, `v0.4`.
- Roadmap issue set generated and created from `docs/github_issues_2026H2.csv`.
- `v0.2` foundation work started and partially shipped:
  - panel-aware YAML config loader added
  - public CLI cleanup started for pipeline, review, and batch
  - batch runner remains frozen by default, with explicit opt-in channel overrides
  - internal quantify/QC helpers now route through `MarkerSpec` seams while preserving legacy outputs
- Next execution focus: continue `v0.2` from internal generic marker-feature scaffolding, not from public CLI/config cleanup.

## Strict Baseline Policy

### Frozen alpha baseline elements

- Default model file
- Default channel assumptions
- Class names/order
- Default preprocessing and erosion settings
- Review merge logic
- Confidence/margin/entropy definitions
- `needs_review` and QC flag logic

### Changes allowed freely

- Documentation
- Tests and synthetic fixtures
- README/demo polish
- CLI aliases/wrappers preserving behavior
- Validation scripts that analyze existing outputs
- Experimental features disabled by default

### Changes requiring explicit versioned comparison

- Default model changes
- Default threshold changes
- Feature-calculation changes for default behavior
- Channel mapping changes
- Erosion default changes
- `needs_review` logic changes
- QC flag definition changes
- Class label/order changes

---

## v0.2 - Panel-Aware Config Foundation + Workflow Hardening

### Objectives

- Keep default scientific behavior stable.
- Improve modeling transparency and usability.
- Add safer config surface without breaking legacy CLI behavior.
- Establish the panel-aware architecture without changing default scientific outputs.

### Deliverables

- `--channel-config` YAML support (legacy flags remain supported).
- Public-facing channel semantics moved away from `type1/type2` toward explicit marker names.
- Panel schema supporting:
  - required `membrane`
  - optional `dapi`
  - optional marker channels from `{I, IIa, IIb, IIx}`
  - explicit residual-inference policy
- `fiber_type_source` / equivalent metadata in outputs to distinguish direct vs inferred calls.
- `docs/modeling.md` covering features, uncertainty metrics, and limits.
- Expanded model card with intended use and failure modes.
- Feature diagnostics scripts with reproducible commands.
- Command-level smoke tests for pipeline/batch/review/merge.
- Optional experimental paths disabled by default.

### Progress So Far (2026-06-02)

- Completed:
  - `fibertypeqc/config.py` with panel-aware schema parsing
  - `docs/panel_schema.md`
  - explicit marker-channel CLI support in:
    - `scripts.run_pipeline`
    - `scripts.review_labels_napari`
    - `scripts.run_batch`
  - legacy alias compatibility:
    - `type1 -> iib`
    - `type2 -> iia`
  - batch warnings when leaving the strict frozen baseline path
  - tests for channel-config parsing and batch override behavior
  - initial internal refactor in `src/quantify_classify.py`:
    - `MarkerSpec`
    - `MarkerStats`
    - marker-aware feature/stat/QC seams that still emit legacy columns

- Not done yet:
  - `fiber_type_source` / direct-vs-inferred provenance in outputs
  - `docs/modeling.md`
  - expanded model card failure-mode writeup
  - feature diagnostics scripts
  - baseline snapshot checks for frozen legacy defaults

### Acceptance Criteria

- Legacy commands still reproduce baseline behavior.
- Config parsing has validation and clear error messages.
- Default panel remains the current lab workflow:
  - membrane + IIb + IIa
  - residual inferred `IIx`
- Alternate channel names/configs can be expressed even if full multi-panel typing logic is not yet enabled by default.
- README remains user-facing; modeling detail moved to docs.
- Lint/tests pass.

### Risks

- Config complexity may reduce usability.
- Silent drift risk while adding options.
- Naming cleanup may expose deeper coupling between current feature columns and the frozen model.

### Test/Validation Gates

- Baseline snapshot check on frozen demo image(s) under legacy path.
- Automated tests for channel-config parsing and smoke workflows.
- Explicit tests for panel validation and precedence between config and CLI overrides.

---

## v0.3 - Generic Marker Features + Panel-Aware Typing

### Objectives

- Generalize typing from a fixed two-marker implementation to a marker-aware feature engine.
- Support direct-marker and residual-inference typing modes explicitly.
- Compare candidate behavior explicitly against alpha baseline.

### Deliverables

- Generic per-marker feature extraction for any present subset of `{I, IIa, IIb, IIx}`.
- Panel-aware typing rules that distinguish:
  - direct marker calls
  - residual inferred calls
  - unresolved/other outputs when the panel does not justify a named residual class
- Support for at least one additional non-default panel beyond the current lab panel.
- Baseline-vs-candidate evaluation report.
- Feature-set comparison (for example p75/p90 vs p75/p90+coverage/SNR).
- Held-out image evaluation summary.
- Confusion/per-class metrics where manual labels exist.
- Error analysis by class/image context.
- Hard vs soft composition comparison.

### Acceptance Criteria

- Candidate artifacts are versioned and reproducible.
- Typing logic no longer assumes `type1/type2` internally for new panel-aware paths.
- Direct `IIx` stain is supported as a first-class marker when present.
- Tradeoffs are explicit (accuracy vs review burden vs uncertainty).
- Any default-behavior change is documented in changelog/release notes.

### Risks

- Limited manual labels for robust metrics.
- Overfitting to familiar cohorts.
- Generalizing across panel types may create partial-output cases that are harder to communicate.

### Test/Validation Gates

- Reproducible evaluation commands in docs.
- At least one alternate panel mode exercised end-to-end with documented limits.
- Candidate must meet predeclared metric targets or provide clear rationale.

---

## v0.4 - Residual Inference Policy + Nuclei Features + Uncertainty-Guided Review

### Objectives

- Support robust residual-class inference as an explicit panel policy rather than hardcoded `IIx`.
- Add optional DAPI-derived nuclear features where present.
- Quantify review utility of confidence/margin/entropy outputs.
- Improve review-efficiency story without overclaiming biology.

### Deliverables

- Residual-inference framework allowing any omitted class to be configured as the residual target when biologically justified by the panel.
- Explicit fallback outputs for insufficient panels (for example `untyped_other` / uncertain rather than overclaimed labels).
- Optional nuclei outputs such as mono-/multi-nucleation and central nuclei summaries when `dapi` is present.
- Review-worthy rate by uncertainty bin.
- Review-efficiency curve.
- Uncertainty heatmaps.
- Soft composition summary outputs integrated into validation docs.
- Explicit caveat language: uncertainty indicates classifier ambiguity/evidence, not proven biological hybridity.

### Acceptance Criteria

- Residual inference is panel-gated and never assumed automatically for every omitted class.
- DAPI-derived outputs are optional and absent cleanly when `dapi` is not configured.
- Review policy guidance can be written from generated outputs.
- Reproducible scripts/commands for all figures and tables.
- Limitation language is explicit and consistent.

### Risks

- Uncertainty may be misread as biological certainty.
- High review rates on noisy images may limit practical gains.
- Residual inference may be overtrusted if provenance is not surfaced clearly in outputs and UI.

### Test/Validation Gates

- Manual audit subset analyzed by uncertainty tier.
- Stability check across at least two cohorts/image types.
- Panel-specific audit showing that residual inferred calls are only enabled where justified.

---

## Weekly Next Steps (Immediate)

1. Finish the next `v0.2` internal slice:
   - allow optional internal collection of extra marker stats (`i`, `iix`) without changing default classification behavior
   - keep exported legacy columns unchanged
2. Add `fiber_type_source` / equivalent provenance fields without changing current default calls.
3. Add `docs/modeling.md` and expand the model card with current failure modes and panel limits.
4. Add baseline snapshot checks for frozen legacy defaults.
5. Keep release notes conservative and explicit about alpha limits.

## Start Here Next Time

When resuming work, start with this slice:

1. Extend `src/quantify_classify.py` so internal marker-stat collection can optionally include extra direct markers (`i`, `iix`) when present.
2. Do not change the exported default feature table or default `IIb + IIa -> inferred IIx` behavior yet.
3. After that seam is in place, add output provenance fields such as:
   - `fiber_type_source`
   - `available_markers`
   - optional residual-inference metadata

Short version: next work should continue the internal marker-aware architecture, not revisit CLI/config cleanup.

---

## GitHub Operating Model (How To Use Milestones, Labels, Issues)

Use this as the default workflow for all roadmap work.

### Milestones

- Milestone = release bucket (`v0.2`, `v0.3`, `v0.4`).
- Every issue in this roadmap should be assigned to exactly one milestone.
- A release should not be tagged until milestone acceptance criteria are reviewed.

### Labels

Use labels in two groups:

- Release-phase labels: `v0.2`, `v0.3`, `v0.4`
- Topic labels: `modeling`, `validation`, `docs`, `tests`, `cli`, `review`, `baseline-sensitive`

Recommended rule: every issue gets one release-phase label and at least one topic label.

### Issues

Each issue should contain:

1. Problem statement
2. Scope boundaries
3. Acceptance criteria
4. Risks/assumptions (if relevant)
5. Repro command(s) or test gate(s)

### Branch + PR workflow

- Create one branch per issue: `feat/<issue-id>-short-name` or `fix/<issue-id>-short-name`.
- Open a PR that links the issue using `Closes #<issue_number>`.
- Keep PR scope aligned with one issue when possible.
- Include before/after evidence for baseline-sensitive changes.

### Baseline-sensitive change protocol

For any issue labeled `baseline-sensitive`:

1. Add explicit comparison notes in PR description.
2. Include metric/output diffs against frozen alpha baseline.
3. Add release-note entry under “Behavior changes”.
4. Require a versioned artifact path (model/report) before merge.

### Weekly operating cadence

Once per week:

1. Review open issues in current milestone.
2. Move blocked issues to explicit blocked state with reason.
3. Close completed issues through linked PRs.
4. Update milestone progress notes (what shipped, what moved, why).

### Milestone close-out checklist

Before tagging a release milestone:

1. Confirm acceptance criteria met for all closed issues.
2. Confirm tests/validation gates passed.
3. Confirm docs/changelog updates are complete.
4. Confirm unresolved issues are intentionally moved to next milestone.
