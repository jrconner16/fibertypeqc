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

## Status Snapshot (2026-06-01)

- `v0.1.1-alpha` released as pre-release with demo screenshots and documentation polish.
- Roadmap milestones created on GitHub: `v0.2`, `v0.3`, `v0.4`.
- Roadmap issue set generated and created from `docs/github_issues_2026H2.csv`.
- Next execution focus: start and ship the first `v0.2` issues via issue-linked PRs.

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

## v0.2 - Model/Feature Transparency + Workflow Hardening

### Objectives

- Keep default scientific behavior stable.
- Improve modeling transparency and usability.
- Add safer config surface without breaking legacy CLI behavior.

### Deliverables

- `--channel-config` YAML support (legacy flags remain supported).
- `docs/modeling.md` covering features, uncertainty metrics, and limits.
- Expanded model card with intended use and failure modes.
- Feature diagnostics scripts with reproducible commands.
- Command-level smoke tests for pipeline/batch/review/merge.
- Optional experimental paths disabled by default.

### Acceptance Criteria

- Legacy commands still reproduce baseline behavior.
- Config parsing has validation and clear error messages.
- README remains user-facing; modeling detail moved to docs.
- Lint/tests pass.

### Risks

- Config complexity may reduce usability.
- Silent drift risk while adding options.

### Test/Validation Gates

- Baseline snapshot check on frozen demo image(s) under legacy path.
- Automated tests for channel-config parsing and smoke workflows.

---

## v0.3 - Classifier Improvement + Baseline Comparison

### Objectives

- Improve classification reliability as a first-class deliverable.
- Compare candidate behavior explicitly against alpha baseline.

### Deliverables

- Baseline-vs-candidate evaluation report.
- Feature-set comparison (for example p75/p90 vs p75/p90+coverage/SNR).
- Held-out image evaluation summary.
- Confusion/per-class metrics where manual labels exist.
- Error analysis by class/image context.
- Hard vs soft composition comparison.

### Acceptance Criteria

- Candidate artifacts are versioned and reproducible.
- Tradeoffs are explicit (accuracy vs review burden vs uncertainty).
- Any default-behavior change is documented in changelog/release notes.

### Risks

- Limited manual labels for robust metrics.
- Overfitting to familiar cohorts.

### Test/Validation Gates

- Reproducible evaluation commands in docs.
- Candidate must meet predeclared metric targets or provide clear rationale.

---

## v0.4 - Uncertainty-Guided Review Validation

### Objectives

- Quantify review utility of confidence/margin/entropy outputs.
- Improve review-efficiency story without overclaiming biology.

### Deliverables

- Review-worthy rate by uncertainty bin.
- Review-efficiency curve.
- Uncertainty heatmaps.
- Soft composition summary outputs integrated into validation docs.
- Explicit caveat language: uncertainty indicates classifier ambiguity/evidence, not proven biological hybridity.

### Acceptance Criteria

- Review policy guidance can be written from generated outputs.
- Reproducible scripts/commands for all figures and tables.
- Limitation language is explicit and consistent.

### Risks

- Uncertainty may be misread as biological certainty.
- High review rates on noisy images may limit practical gains.

### Test/Validation Gates

- Manual audit subset analyzed by uncertainty tier.
- Stability check across at least two cohorts/image types.

---

## Weekly Next Steps (Immediate)

1. Milestones created: `v0.2`, `v0.3`, `v0.4`.
2. Milestone issues created from `docs/github_issues_2026H2.csv`.
3. Start `v0.2` with `channel-config` + docs/modeling + baseline snapshot checks.
4. Keep release notes conservative and explicit about alpha limits.

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
