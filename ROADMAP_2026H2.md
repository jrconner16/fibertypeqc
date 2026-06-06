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

## Status Snapshot (2026-06-03)

- `v0.1.1-alpha` released as pre-release with demo screenshots and documentation polish.
- Roadmap milestones created on GitHub: `v0.2`, `v0.3`, `v0.4`.
- Roadmap issue set generated and created from `docs/github_issues_2026H2.csv`.
- `v0.2` foundation work started and partially shipped:
  - `docs/modeling.md` and the expanded frozen alpha model card are in place
  - panel-aware YAML config loader added
  - public CLI cleanup shipped for pipeline, review, and batch
  - batch runner remains frozen by default, with explicit opt-in channel overrides
  - internal quantify/QC helpers now route through `MarkerSpec` seams while preserving legacy outputs
  - provenance fields and optional diagnostics export landed without changing the stable fibers CSV
  - baseline snapshot tests now guard the frozen legacy feature contract and rule path
  - fast vs integration test split is in place for predictable default pytest runs
- Next execution focus: decide whether to finish a little more `v0.2` documentation polish or start explicit candidate-model evaluation work.

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

### Progress So Far (2026-06-03)

- Completed:
  - `fibertypeqc/config.py` with panel-aware schema parsing
  - `docs/panel_schema.md`
  - `docs/modeling.md`
  - expanded `data/models/model_card.md`
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
  - internal optional extra-marker stats for direct `i` / `iix`
  - output provenance fields:
    - `fiber_type_source`
    - `available_markers`
  - feature-set comparison tooling:
    - `src/compare_feature_sets.py`
    - `validation/compare_feature_sets.py`
  - optional diagnostics export:
    - `--export-diagnostics`
    - `*_feature_diagnostics.csv`
  - diagnostics export documented in user-facing docs as an advanced/model-development option
  - smoke tests for diagnostics export path
  - baseline snapshot tests for frozen defaults:
    - frozen alpha feature-contract check
    - legacy rule-path regression snapshot
  - pytest split into:
    - fast synthetic default path
    - optional `integration` path

- Not done yet:
  - choose whether the next milestone step is more `v0.2` hardening or `v0.3` candidate-model evaluation

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

### Progress So Far (2026-06-04)

- Completed:
  - `docs/baseline_comparison_protocol.md`
  - candidate feature-table assembly:
    - `src/build_candidate_feature_table.py`
    - `validation/build_candidate_feature_table.py`
  - candidate split-manifest tooling:
    - `src/build_candidate_split_manifest.py`
    - `validation/build_candidate_split_manifest.py`
  - accepted image-level `train` / `dev` / `heldout` split for the 32-image MyoSight-comparable set
  - rerun of the 32-image validation batch with diagnostics export enabled
  - assembled `outputs/validation/candidate_feature_table.csv`
  - first candidate-model training/evaluation workflow:
    - `src/train_candidate_from_feature_table.py`
    - `validation/train_candidate_from_feature_table.py`
  - first candidate comparison on `dev`:
    - `baseline_rf`
    - `baseline_gb`
    - `expanded_rf`
    - `expanded_gb`
  - result so far:
    - frozen baseline `baseline_rf` remains best on `dev`
    - expanded features and boosting did not beat the frozen baseline in first-pass evaluation
  - matched-ROI MyoSight audit tooling:
    - `src/build_matched_myosight_audit.py`
    - `validation/build_matched_myosight_audit.py`
  - combined audit-set tooling:
    - `src/build_combined_audit_set.py`
    - `validation/build_combined_audit_set.py`
  - audit sampling tooling:
    - `src/sample_combined_audit_set.py`
    - `validation/sample_combined_audit_set.py`
  - dedicated audit-review Napari helper with adjudication output:
    - `src/review_audit_napari.py`
    - `validation/review_audit_napari.py`
  - consolidated reviewed-audit benchmark tables:
    - `outputs/validation/reviewed_audit_all.csv`
    - `outputs/validation/audit_benchmark_labels.csv`
  - reviewed benchmark evaluation workflow:
    - `src/evaluate_against_audit_benchmark.py`
    - `validation/evaluate_against_audit_benchmark.py`
  - reviewed benchmark split tooling:
    - `src/split_reviewed_benchmark.py`
    - `validation/split_reviewed_benchmark.py`
  - experimental weighted-supervision candidate training workflow:
    - `src/train_weighted_candidate_from_audit.py`
    - `validation/train_weighted_candidate_from_audit.py`
  - reviewed benchmark result so far:
    - `expanded_rf` looks best on the manually reviewed benchmark
    - this differs from the earlier self-label `dev` result and confirms the need for manual supervision
  - first weighted-supervision probe result so far:
    - using `manual_train` rows with strong weight improves protected manual-holdout performance
    - current best weighted candidate is `baseline_gb`
    - promising signal, but not enough evidence yet for any default-model change
  - refreshed weighted-supervision result on the larger reviewed set:
    - `manual_train` override rows increased from the small initial probe to `216`
    - protected `manual_eval_holdout` rows increased to `221`
    - baseline-feature `baseline_gb` improves to about `0.615` accuracy / `0.616` balanced accuracy
    - accuracy trend is improving as manual review expands, which matters more now because the benchmark is driven by manual biological adjudication rather than legacy model self-labels
  - merged supervision result after adding the confirmed `true_iia_hunt` positives:
    - reviewed benchmark expanded to `1275` rows across `31` images
    - `manual_train` now includes meaningful confirmed `IIa` labels (`122`)
    - protected `manual_eval_holdout` now includes meaningful confirmed `IIa` labels (`40`)
    - weighted baseline-feature `baseline_gb` improves further to about `0.761` accuracy / `0.815` balanced accuracy
    - this is the strongest evidence so far that supervision quality, not model family search, was the main bottleneck
  - `IIa` gate analysis on the rebuilt manual benchmark:
    - `src/analyze_iia_gate.py`
    - `validation/analyze_iia_gate.py`
    - strict `IIa` evidence gating proved that the false-positive `IIa` problem is real, but the first hard gate (`q=0.10`) was too aggressive and hurt balanced accuracy
    - a softer `IIa` gate derived from confirmed `true_iia_hunt` positives (`q=0.01`) is now the leading candidate on the protected manual holdout
    - `baseline_gb_gated_iia_q0.01` reaches about `0.801` accuracy / `0.839` balanced accuracy on `manual_eval_holdout`
    - this soft gate also improves image-level `IIa` agreement against the consolidated MyoSight summary as a secondary comparison layer

- Current interpretation:
  - the modeling runway is in place
  - first-pass candidate-model changes did not improve held-out biology by themselves under legacy labels
  - manually reviewed benchmark rows are already shifting the model ranking
  - a first weighted-supervision probe already improves protected manual-holdout performance
  - the upward weighted-manual trend is encouraging, especially because it reflects manually adjudicated labels rather than chasing legacy model agreement
  - adding a curated positive `IIa` anchor set materially strengthens the weighted-manual result, which supports the idea that explicit positive-definition work is as important as negative/error cleanup
  - the `IIa` boundary now looks like a decision-threshold problem as much as a model problem; a soft evidence gate improves both the primary manual benchmark and the secondary MyoSight `IIa` composition comparison
  - the next likely bottleneck is supervision quality and class coverage, not another blind feature/model-family sweep
  - the reviewed benchmark is now materially stronger, but still should be treated as an evolving supervision asset rather than a final locked benchmark

- Not done yet:
  - operationalize the best soft `IIa` gate candidate (`q=0.01`) into a reproducible candidate pipeline path, rather than keeping it only as an analysis artifact
  - validate the soft-gated candidate on additional heldout/external image sets before considering any default-behavior change
  - grow the manual label set further where it still adds signal, especially for edge-case `IIa/iix` and difficult image-quality regimes
  - repeat weighted-supervision candidate training after each meaningful expansion of the reviewed manual set
  - test a three-tier supervision scheme explicitly:
    - baseline/self labels with low weight
    - matched MyoSight weak labels with medium weight
    - manual reviewed labels with high weight
  - note: matched MyoSight labels have **not yet** been used in training with medium weight; that remains untested

### Acceptance Criteria

- Candidate artifacts are versioned and reproducible.
- Typing logic no longer assumes `type1/type2` internally for new panel-aware paths.
- Direct `IIx` stain is supported as a first-class marker when present.
- Tradeoffs are explicit (accuracy vs review burden vs uncertainty).
- Any default-behavior change is documented in changelog/release notes.
- Reviewed audit fibers are kept separate from ordinary pipeline outputs and can be reused as a higher-trust benchmark.
- No corrected audit labels are silently written back into the stable `*_fibers.csv` outputs.
- Manual reviewed fibers are split into benchmark holdback vs optional training supervision before retraining.
- Manual supervision expansion intentionally includes underrepresented classes, especially `IIa`.

### Risks

- Limited manual labels for robust metrics.
- Overfitting to familiar cohorts.
- Generalizing across panel types may create partial-output cases that are harder to communicate.
- Candidate training may overfit to weak comparator labels if matched MyoSight labels are treated as ordinary truth.
- Small reviewed audit sets can be highly informative, but only if some portion is held back from training.
- Review-display settings can bias manual supervision if one marker channel is visually over-amplified relative to the raw evidence.
- If manual-label sampling is not stratified, the review set will keep collapsing toward already-common `IIb` / `IIx` cases and fail to improve `IIa` supervision.

### Test/Validation Gates

- Reproducible evaluation commands in docs.
- At least one alternate panel mode exercised end-to-end with documented limits.
- Candidate must meet predeclared metric targets or provide clear rationale.
- Reviewed audit labels should first be used as an evaluation benchmark before they are used as training supervision.
- If reviewed fibers are used in training, retain a separate untouched reviewed subset for evaluation.
- Any enlarged manual-label campaign should track class balance explicitly and not rely on passive/random review alone.
- Manual-label sampling should use distinct benchmark-enrichment and supervision-enrichment pools rather than one undifferentiated random sample.
- Mixed manual review sessions are acceptable, but downstream consolidation/splitting must still preserve the intended role of each sampled row (`manual_eval_candidate` vs `manual_train_candidate`) to avoid training/evaluation leakage.

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

1. Keep corrected manual labels separate from stable pipeline outputs.
2. Expand the manual label set with deliberate `IIa` enrichment rather than only collecting more `IIb`/`IIx` examples.
3. Use separate round-2 sampling pools:
   - benchmark enrichment with broader image diversity
   - supervision enrichment with denser `IIa` / ambiguity coverage
4. Treat reviewed audit fibers as a benchmark first, then as weighted supervision only after a holdback/evaluation split is defined.
5. Weighted-supervision probes are useful now, but no default-model decision should rely on them until the manual set is larger and better balanced.

## Start Here Next Time

When resuming work, start with this slice:

1. Build and review the round-2 manual-labeling sample.
2. Expand the reviewed audit set with deliberate `IIa` and `IIa/iix` enrichment before expecting meaningful supervised gains.
3. Rerun:
   - `validation.consolidate_reviewed_audit`
   - `validation.split_reviewed_benchmark`
   - while preserving `manual_round2_pool` intent downstream even if the review session itself was mixed
4. Keep the frozen alpha baseline as the explicit comparator for every candidate run.
5. Re-run the weighted-supervision probe after each meaningful manual-label expansion and watch whether the protected manual holdout continues to improve.

Short version: the remaining work is now more about supervision quality and benchmark design than about more plumbing or blind model-family search.

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
