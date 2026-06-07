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

## Candidate Promotion Protocol

Use this protocol before replacing the frozen alpha default model or changing default typing behavior.

### Current leading candidate

- Weighted manual-supervision `baseline_gb`
- Post-model soft `IIa` evidence gate at `q=0.01`
- Refreshed protected manual-holdout result after the 400-row supervision expansion: about `0.825` accuracy / `0.849` balanced accuracy
- This remains the strongest current candidate because it fixes the frozen model's main `IIa` false-positive failure mode without losing most true `IIa`.
- Candidate behavior is **not yet default-ready** because the full-cohort MyoSight comparison still has image-level leakage, no random/manual benchmark exists yet, and the remaining `IIb`/`IIx` discrepancy is unresolved.

### Minimum promotion gates

A candidate can be considered for default replacement only if all of the following are true:

- Beats frozen alpha on the protected manual benchmark, including per-class metrics for `IIa`, `IIb`, and `IIx`.
- Does not regress badly on a smaller random/manual benchmark that is not only enriched for hard cases.
- Has an apples-to-apples full-cohort comparison against frozen using one explicit validation manifest.
- Resolves the 7 section-export image provenance issue by pinning trusted source paths for both frozen and candidate.
- Ships as a versioned artifact bundle:
  - model file
  - gate/threshold config
  - feature schema
  - validation manifest
  - exact command(s)
  - git commit
  - report tables/plots
- Has model-card and changelog notes explaining behavior changes, expected gains, known failure modes, and validation caveats.

### Decision states

- `analysis-only`: useful validation result, not a runnable candidate pipeline.
- `candidate-pipeline`: runnable frozen-vs-candidate path with versioned artifacts, still experimental.
- `internal-default-candidate`: passes protected manual and full-cohort descriptive checks; ready for cleaner heldout validation.
- `default-ready`: passes promotion gates and has release notes/model card updates.

The current `baseline_gb + soft IIa gate q=0.01` should be treated as `candidate-pipeline` once the gate is operationalized in a reproducible batch path.

---

## Benchmark Taxonomy

Keep benchmark roles explicit. Do not collapse all reviewed fibers into one undifferentiated score.

### Manual challenge benchmark

- Source: curated audit rows, disagreement cases, hard `IIa/iix` and `IIb/iix` cases, review-flagged fibers.
- Purpose: model-development pressure test.
- Strength: sensitive to biologically important failure modes.
- Limitation: not prevalence-matched and likely harder than routine field performance.

### Manual random benchmark

- Source: random or prevalence-ish manual review across images, separate from the challenge set.
- Purpose: estimate everyday performance and catch regressions on common/easy cases.
- Status: not built yet.
- Priority: high before any default-model promotion.

### Positive anchor sets

- Source: targeted evidence-first sets such as `true_iia_hunt`.
- Purpose: define what clean positive evidence looks like and derive/tune class-specific gates.
- Limitation: should not be treated as prevalence or general performance benchmark.

### MyoSight descriptive cohort

- Source: 32-image MyoSight-comparable cohort.
- Purpose: compare image-level composition and biological trends.
- Strength: useful external-ish comparator for cohort-level story.
- Limitation: not primary ground truth; current candidate comparison has image-level leakage from manual supervision.

### Image-heldout validation

- Source: images with no manual-train supervision and no tuning contact.
- Purpose: cleaner generalization check before default promotion.
- Status: not fully established yet.
- Priority: high for in-house model confidence.

---

## Manifest-Driven Validation + Storage Policy

### Validation provenance rules

- Full-cohort comparisons must be manifest-driven.
- Each image row should record:
  - `image_id`
  - trusted source image path
  - input kind (`direct_czi`, `section_tiff_export`, etc.)
  - expected channel mapping
  - training/holdout role
  - MyoSight result path when applicable
- Frozen and candidate comparisons must use the same source manifest.
- Section-export images must remain pinned to the trusted normalized section-export source branch unless a better section workflow is validated.
- Never compare candidate and frozen count/typing summaries if they came from different section-image path branches.

### Output artifact policy

- Keep lightweight outputs locally by default:
  - summary CSVs
  - metric tables
  - plots
  - final `*_fibers.csv`
  - benchmark split manifests
  - model/gate configs
- Archive or externalize heavy image-derived artifacts:
  - `*_cellpose_labels.tif`
  - full diagnostic image trees
  - repeated batch rerun directories
- Prefer `--retain-mode tables` for routine validation/modeling runs and reserve `--retain-mode full` for reviewable/debug runs that need label TIFFs in Napari.
- Avoid copying whole batch directories for postprocessing experiments.
- Prefer in-memory or summary-only postprocessing when evaluating gates, thresholds, or label remaps.
- Preserve one canonical frozen batch root and one canonical candidate batch root per validation campaign; archive everything else.

### Repo cleanup priority

Treat cleanup as validation infrastructure, not cosmetic refactoring.

- High priority:
  - identify canonical candidate, frozen, benchmark, and cohort-summary artifacts
  - archive or remove redundant image/mask batch trees that are no longer active
  - add an `outputs/README.md` or equivalent artifact map explaining which outputs are canonical and which are historical
  - prevent future path/provenance confusion by moving full-cohort runs to explicit manifests
- Medium priority:
  - consolidate duplicated validation helper scripts only when it reduces actual ambiguity
  - clean up old experiment names after preserving any result tables that are still referenced by the roadmap
  - add ignore/archive policy for large generated TIFFs and diagnostics trees
- Low priority:
  - cosmetic code cleanup that does not affect reproducibility, storage, or validation decisions
  - renaming every historical artifact for tidiness alone

Cleanup should happen before another large validation rerun. The goal is to make the next result hard to misread.

### Segmentation robustness policy

- Treat segmentation and typing as separate validation layers.
- Do not interpret typing-model differences until segmentation provenance and total fiber counts are comparable.
- For section exports, define a QC-triggered fallback policy before enabling automated reruns:
  - candidate QC inputs: `n_labels`, tissue coverage, median area, type-correlation warnings, signal-warning burden
  - do not select a segmentation solely because it has more fibers
  - prefer a scored decision that penalizes oversegmentation and implausible morphology

---

## Software Note Scope

The software-note path should not wait for a perfect candidate model. The publishable software story is:

- reproducible fiber typing workflow
- panel-aware channel configuration
- explicit baseline policy
- Napari review/audit tooling
- benchmark/evaluation scripts
- transparent model card and validation caveats
- optional candidate-model case study

### Minimum publishable package

- Stable frozen-alpha workflow remains runnable and documented.
- Public docs clearly explain:
  - intended use
  - channel assumptions
  - residual `IIx` inference limits
  - review burden and uncertainty interpretation
  - segmentation caveats
- Example validation outputs can be regenerated from documented commands.
- Experimental candidate workflows are clearly labeled as such and disabled by default.
- The software note should frame the candidate model as a case study in improving biological supervision, not as a claim that the model is universally solved.

### Scope guardrails

- Do not let ongoing model research block software packaging indefinitely.
- Do not require CNNs, full multi-panel generalization, or perfect MyoSight agreement for the software-note story.
- Do require reproducible commands, clean artifact provenance, and explicit limitations.

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
  - candidate review-threshold calibration on the reviewed benchmark:
    - the earlier dramatic candidate review-burden drop was a calibration artifact, not a proved operational gain
    - inherited frozen thresholds (`0.55` confidence / `0.15` margin) flag only about `9%` of the challenge benchmark while missing most real errors
    - confidence-only review thresholding is cleaner than mixing confidence and margin, but no current candidate threshold reaches the desired low review burden without leaving too many misses
    - conclusion: candidate classification improved, but candidate `needs_review` policy is not operationally ready yet
  - next supervision-pool tooling and review campaign:
    - `src/build_next_supervision_pools.py`
    - `validation/build_next_supervision_pools.py`
    - generated and reviewed a `400`-row tranche split into:
      - `100` broad random benchmark controls
      - `100` `IIa` positives
      - `100` balanced `IIa/iix` ambiguity rows
      - `100` `IIb` positives
    - the full tranche has now been reviewed and folded back into the benchmark/supervision tables
  - refreshed reviewed benchmark after the 400-row expansion:
    - reviewed rows increase to `1702`
    - benchmark rows increase to `1674`
    - `manual_train` increases to `1274`
    - protected `manual_eval_holdout` increases to `400`
    - protected holdout now includes stronger class coverage (`IIa 66`, `IIb 170`, `IIx 159`)
  - refreshed weighted-supervision result on the larger reviewed set:
    - weighted baseline-feature `baseline_gb` improves again to about `0.805` accuracy / `0.844` balanced accuracy
    - this confirms that the supervision expansion is still translating into real protected-holdout gains
  - refreshed supervision-recipe ablation on the larger reviewed set:
    - `manual_only_high` improves to about `0.808` accuracy / `0.843` balanced accuracy
    - `manual_high_myo_medium` reaches about `0.808` / `0.845`
    - `manual_high_myo_medium_baseline_light` is now the best ungated recipe at about `0.820` / `0.855`
    - `manual_high_myo_medium_baseline_light_soft_iia_q0.01` reaches about `0.828` accuracy / `0.850` balanced accuracy
    - interpretation: manual-only is getting much stronger, but mixed supervision still wins today
  - `IIa` gate analysis on the rebuilt manual benchmark:
    - `src/analyze_iia_gate.py`
    - `validation/analyze_iia_gate.py`
    - strict `IIa` evidence gating proved that the false-positive `IIa` problem is real, but the first hard gate (`q=0.10`) was too aggressive and hurt balanced accuracy
    - a softer `IIa` gate derived from confirmed `true_iia_hunt` positives (`q=0.01`) is now the leading candidate on the protected manual holdout
    - `baseline_gb_gated_iia_q0.01` reaches about `0.801` accuracy / `0.839` balanced accuracy on `manual_eval_holdout`
    - this soft gate also improves image-level `IIa` agreement against the consolidated MyoSight summary as a secondary comparison layer
  - refreshed `IIa` gate rerun on the larger reviewed benchmark:
    - ungated weighted `baseline_gb` is now about `0.805` accuracy / `0.844` balanced accuracy
    - `baseline_gb_gated_iia_q0.01` remains the best current candidate at about `0.825` accuracy / `0.849` balanced accuracy
    - the gate drives `IIa` precision from about `0.759` to `1.000` while keeping recall about `0.939`
    - `q=0.01` still looks best; stricter gates reduce balanced accuracy
    - `IIa -> IIb` redirect variants still do not help
    - MyoSight sidecar read remains the same: better `IIa`, unchanged `IIb`, and residual excess mass still landing in `IIx`
  - full 32-image MyoSight comparison plumbing:
    - candidate batch wrapper:
      - `src/run_candidate_batch_with_iia_gate.py`
      - `validation/run_candidate_batch_with_iia_gate.py`
    - 3-way MyoSight comparison plots:
      - `src/plot_three_way_myosight_compare.py`
      - `validation/plot_three_way_myosight_compare.py`
    - 3-way biology-story plots:
      - `src/plot_three_way_biological_story.py`
      - `validation/plot_three_way_biological_story.py`
    - section-export provenance issue identified:
      - the frozen consolidated summary used `outputs/myosight_section_series_exports_normalized` for the 7 section-export images
      - the first candidate rerun used a different flat validation input branch, causing non-apples-to-apples counts and severe undersegmentation for at least `Section001_mouse354_mdx_TA`
    - trusted section-export rerun path added:
      - `src/build_section_export_input_manifest.py`
      - `validation/build_section_export_input_manifest.py`
      - `src/build_hybrid_candidate_summary.py`
      - `validation/build_hybrid_candidate_summary.py`
    - cleaned hybrid candidate summary now uses:
      - ordinary candidate rows for direct-CZI images
      - trusted section-export candidate rows for the 7 section-export images
    - after the hybrid fix, total fiber counts between frozen and candidate are effectively the same; the remaining discrepancy is typing biology, not fiber-count loss
  - full-cohort biology-story read so far:
    - candidate is much closer to MyoSight than frozen for `IIa` across genotype/age summaries
    - candidate is also closer to MyoSight for `IIb` in the summary tables
    - candidate remains high for `IIx` relative to MyoSight, suggesting residual/negative-call mass is still being absorbed by `IIx`
    - a first `IIa`-fail -> `IIb` redirect rule was tested on the protected manual holdout and did not change results; none of the failed `IIa` calls met the strong `IIb` redirect rule
    - any broader redirect/gate experiment should be treated as descriptive until disk pressure and validation provenance are cleaned up

- Current interpretation:
  - the modeling runway is in place
  - first-pass candidate-model changes did not improve held-out biology by themselves under legacy labels
  - manually reviewed benchmark rows are already shifting the model ranking
  - a first weighted-supervision probe already improves protected manual-holdout performance
  - the upward weighted-manual trend is encouraging, especially because it reflects manually adjudicated labels rather than chasing legacy model agreement
  - adding a curated positive `IIa` anchor set materially strengthens the weighted-manual result, which supports the idea that explicit positive-definition work is as important as negative/error cleanup
  - the `IIa` boundary now looks like a decision-threshold problem as much as a model problem; a soft evidence gate improves both the primary manual benchmark and the secondary MyoSight `IIa` composition comparison
  - after correcting section-export provenance, the candidate-vs-frozen count comparison is mostly a segmentation/input-artifact issue rather than a typing-model issue
  - the remaining biology discrepancy is no longer mainly `IIa`; the next class-boundary question is why the candidate tends to put more residual mass into `IIx` while MyoSight trends suggest more `IIb`
  - the earlier very low candidate review burden was a calibration artifact; the current candidate review policy should not be treated as an operational win yet
  - margin-based review thresholding is not helping much; confidence-only is the cleaner form if review triage is revisited later
  - the next likely bottleneck is still supervision quality and class coverage, not another blind feature/model-family sweep
  - the new 400-row supervision tranche materially strengthened the reviewed benchmark and the weighted candidate
  - manual-only supervision is now much more credible, but mixed supervision still gives the best protected-holdout results today
  - the refreshed `IIa` gate rerun strengthens the current best-candidate story rather than weakening it
  - the main remaining biology discrepancy is now the residual `IIb`/`IIx` split, not the old `IIa` overcall pattern
  - the reviewed benchmark is now materially stronger, but it should still be treated as an evolving supervision asset rather than a final locked benchmark

- Not done yet:
  - continue storage cleanup discipline:
    - the obvious failed/partial redirect trees have already been archived and the canonical artifact map now lives in `outputs/README.md`
    - keep archiving superseded full batch-output trees once their summary tables and manifests are preserved
  - long-term storage policy:
    - avoid copying whole batch directories for simple postprocessing tests
    - prefer summary-only analysis, manifest-driven reruns, or explicit archive-to-external-drive workflows
    - use `--retain-mode tables` for routine validation runs and keep full `*_cellpose_labels.tif` trees only for canonical reviewable runs or active segmentation debugging
  - resolve section-export provenance permanently:
    - use a manifest-driven validation run with one trusted source path per image
    - keep the 7 section-export images tied to the trusted normalized section-export branch unless/until a better section workflow is validated
    - rerun frozen and candidate from the same manifest before treating any full-cohort count/typing comparison as final
  - decide section-export segmentation robustness policy:
    - do not blindly choose the segmentation with more fibers
    - add QC-triggered fallback logic only after defining a score that accounts for `n_labels`, tissue coverage, median area, type-correlation warnings, and signal-warning burden
    - consider a section-export-specific segmentation profile if the trusted section inputs remain consistently better than the flat validation branch
  - investigate the remaining `IIb`/`IIx` discrepancy:
    - inspect whether excess candidate `IIx` is uniform across images or concentrated by age/genotype/input kind
    - compare failed/demoted `IIa` calls and high-`IIx` candidate calls against raw `IIb` evidence
    - test softer or probability-aware reassignment rules only after the descriptive cohort analysis shows a real target, not from the current strict redirect rule alone
  - candidate review-policy follow-up:
    - treat the low-burden candidate review story as resolved in the negative: the inherited frozen thresholds do not transfer
    - do not claim an operational review-burden win yet
    - revisit review triage only after more benchmark coverage or a better risk signal is available
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
- Repeated full batch reruns can exhaust local storage quickly because every run duplicates large `*_cellpose_labels.tif` masks and image-derived diagnostics.
- Section-export images can silently drift between different source-path branches; this can change segmentation counts by thousands of fibers and invalidate apparent candidate-vs-frozen count differences.
- A post-model gate that fixes `IIa` false positives can shift errors into `IIx`; the biology-story comparison must distinguish improved `IIa` precision from unresolved `IIb`/`IIx` calibration.
- Candidate review burden can look artificially excellent if frozen-model confidence thresholds are reused without recalibration.

### Test/Validation Gates

- Reproducible evaluation commands in docs.
- At least one alternate panel mode exercised end-to-end with documented limits.
- Candidate must meet predeclared metric targets or provide clear rationale.
- Reviewed audit labels should first be used as an evaluation benchmark before they are used as training supervision.
- If reviewed fibers are used in training, retain a separate untouched reviewed subset for evaluation.
- Any enlarged manual-label campaign should track class balance explicitly and not rely on passive/random review alone.
- Manual-label sampling should use distinct benchmark-enrichment and supervision-enrichment pools rather than one undifferentiated random sample.
- Mixed manual review sessions are acceptable, but downstream consolidation/splitting must still preserve the intended role of each sampled row (`manual_eval_candidate` vs `manual_train_candidate`) to avoid training/evaluation leakage.
- Full-cohort MyoSight comparison must use a single explicit image manifest for frozen and candidate runs; section-export images should not be compared across different input-path branches.
- Candidate-vs-frozen count comparison should be interpreted only after verifying that segmentation provenance and total fiber counts are comparable.
- Any `IIb`/`IIx` postprocessing refinement must be evaluated on both the protected manual benchmark and the descriptive 32-image MyoSight cohort, with the leakage caveat stated.
- Candidate review-burden claims must be validated with candidate-specific threshold calibration rather than copied frozen-model confidence cutoffs.

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

1. Keep repo/output cleanup in maintenance mode as validation infrastructure:
   - continue archiving superseded full batch-output trees
   - keep summaries/manifests/canonical batch roots local
   - use `--retain-mode tables` for routine non-review validation runs
2. Treat `myosight_candidate_baseline_gb_soft_iia_q001_hybrid_image_summary.csv` as the current cleaned descriptive cohort summary, not the earlier low-count candidate summary.
3. Treat the candidate review-threshold question as answered for now:
   - the earlier low review burden was a calibration artifact
   - do not reuse frozen `needs_review` thresholds as an operational candidate policy
4. Keep corrected manual labels separate from stable pipeline outputs.
5. Treat the refreshed supervision results as the new benchmarked baseline:
   - weighted ungated `baseline_gb` is now about `0.805` accuracy / `0.844` balanced accuracy
   - `baseline_gb + soft IIa gate q=0.01` is now about `0.825` / `0.849` and remains the current best candidate
6. Shift the next supervision push from generic `IIa` growth to the remaining boundary problem:
   - keep broad random benchmark enrichment active
   - target `IIb`/`IIx` residual-boundary supervision before inventing a new `IIx -> IIb` gate
7. Treat reviewed audit fibers as a benchmark first, then as weighted supervision only after a holdback/evaluation split is defined.
8. Mixed supervision still beats manual-only today, but rerun the manual-vs-mixed ladder after each meaningful label expansion and watch whether manual-first keeps catching up.

## Start Here Next Time

When resuming work, start with this slice:

1. Keep the current best candidate fixed as:
   - weighted manual `baseline_gb`
   - plus soft `IIa` gate `q=0.01`
2. Use manifest-driven validation paths for full-cohort comparisons, with the 7 section-export images pinned to the trusted section-export inputs.
3. Use the cleaned hybrid candidate summary and regenerated 3-way plots when discussing MyoSight trends.
4. Audit the remaining `IIb`/`IIx` discrepancy:
   - start with image/genotype/age concentration
   - then inspect whether candidate `IIx` calls have strong direct `IIb` evidence
   - build stronger `IIb` supervision before trying any `IIx -> IIb` postprocessing rule
5. Treat the candidate review-policy question as paused:
   - the low-burden story was a calibration artifact
   - revisit only if a better risk signal or broader random benchmark is available
6. Build the next manual-labeling sample only if it answers the `IIb`/`IIx` boundary question or broadens the random benchmark.
7. After each new reviewed tranche, rerun:
   - `validation.consolidate_reviewed_audit`
   - `validation.split_reviewed_benchmark`
   - weighted candidate training
   - supervision-recipe comparison
   - `validation.analyze_iia_gate`
8. Keep the frozen alpha baseline as the explicit comparator for every candidate run.
9. Watch whether manual-only or manual-heavy supervision continues to close the gap, but assume mixed supervision remains the current operational recipe until the data say otherwise.

Short version: the remaining work is now about three separate tracks: supervision quality, validation provenance/storage hygiene, and the specific `IIb`/`IIx` residual discrepancy.

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
