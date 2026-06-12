# FiberTypeQC Roadmap (2026 H2)

Time window: July-December 2026  
Priority order: portfolio/job value first, reproducibility/citability second, optional software-note/Journal of Open Source Software (JOSS) path kept open.

## Context

- Public repo: `jrconner16/fibertypeqc`
- Current published release: `v0.2.0`
- Historical tags: `v0.1.0-alpha`, `v0.1.1-alpha`
- Current public workflow:
  1. `scripts.run_pipeline`
  2. `scripts.run_batch`
  3. `scripts.review_labels_napari`
  4. `scripts.merge_reviewed_labels`
- Frozen baseline default model: `data/models/rebaseline_tile_v2_p75p90_iib_iia_iix.joblib`
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
- Treat `membrane` as required for segmentation and `dapi` (4′,6-diamidino-2-phenylindole) as optional for nuclear features.
- Treat inferred classes as decision outputs, not as channels.
- Record whether a final fiber call came from:
  - direct marker evidence
  - residual inference from an omitted class under an allowed panel
  - unresolved/other/uncertain logic
- Support arbitrary 4-channel panel combinations at the measurement layer, but only allow biological class claims justified by the configured panel.

## Status Snapshot (2026-06-12)

- `v0.2.0` is published as the current public release.
- Historical `v0.1.*-alpha` releases remain available as earlier prerelease milestones.
- Roadmap milestones created on GitHub: `v0.2`, `v0.3`, `v0.4`.
- Roadmap issue set generated and created from `docs/github_issues_2026H2.csv`.
- `v0.2.0` foundation work is complete and published:
  - `docs/modeling.md` and the expanded frozen baseline model card are in place
  - panel-aware YAML config loader added
  - public CLI cleanup shipped for pipeline, review, and batch
  - batch runner remains frozen by default, with explicit opt-in channel overrides
  - internal quantify/QC helpers now route through `MarkerSpec` seams while preserving legacy outputs
  - provenance fields and optional diagnostics export landed without changing the stable fibers CSV
  - baseline snapshot tests now guard the frozen legacy feature contract and rule path
  - fast vs integration test split is in place for predictable default pytest runs
- `v0.3` candidate-model evaluation is now the active workstream:
  - manifest-driven full-cohort validation contract exists
  - frozen batch path and candidate batch path both support input manifests
  - reviewed manual benchmark has grown to `1924` rows after the `IIx`/`IIb` boundary round
  - latest protected manual evaluation is based on `457` scored holdout rows
  - current best experimental weight setting is `manual8_myo3_base0p1`, about `0.814` accuracy / `0.847` balanced accuracy on the liberal protected manual holdout
  - the repaired cohort comparison now suggests the experimental candidate broadly matches the MyoSight biological story, with remaining mismatch concentrated in a few `IIb`/`IIx` boundary groups and hotspot images
  - review-policy calibration now suggests the operational path is not one global `needs_review` threshold but `IIa` gating, broad trust in predicted `IIb`, and ranked high-risk gated `IIx` review
- Next execution focus: keep the public release stable, clarify candidate status, and work the remaining `IIb`/`IIx` boundary gap without overclaiming review-burden wins.

## Strict Baseline Policy

### Frozen baseline elements

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

## Scientific Interpretation Framework

FiberTypeQC is evaluated against MyoSight as a historical workflow comparator and against manually reviewed subsets where available. Disagreements between methods are review targets for understanding technical, labeling, and biological sources of variation.

Evidence should be interpreted in this order:

1. Manual visual audit
2. Protected manual holdout labels
3. MyoSight / historical workflow comparison
4. Model confidence, probabilities, margins, and entropy

Current interpretation is conservative:

- The standing candidate model broadly improves alignment with the historical MyoSight workflow relative to the frozen baseline.
- Remaining disagreement is concentrated primarily in `IIb`/`IIx` boundary cases and hotspot images.
- Inferred `IIx` calls remain the highest-priority class for targeted review.
- Additional image-heldout validation and review-policy validation are required before promoting the candidate model to the public default.

FiberTypeQC should be presented as a reproducible, auditable, review-assisted workflow. It is not currently claimed to be a final replacement for MyoSight or a validated standalone biological interpretation tool.

---

## Candidate Promotion Protocol

Use this protocol before replacing the frozen baseline default model or changing default typing behavior.

### Current leading candidate

- Benchmark-leading experimental candidate after the liberal `IIb`/`IIx` relabel plus weight sweep:
  - `manual8_myo3_base0p1`
  - about `0.814` accuracy / `0.847` balanced accuracy on the protected manual holdout
  - `IIb` precision about `0.833`, recall about `0.816`
  - `IIx` precision about `0.711`, recall about `0.726`
- Close alternatives:
  - `manual8_myo4_base0p1`: about `0.811` / `0.844`, slightly more MyoSight-like on some cohort `IIb` groups but weaker protected-holdout score
  - `manual_high_myo_medium_baseline_light`: about `0.807` / `0.842` under the liberal benchmark split
  - weighted `baseline_gb + soft IIa gate q=0.01`: about `0.781` / `0.823` under the liberal benchmark split
- Status interpretation:
  - after the liberal `IIb`/`IIx` relabel and the small weight sweep, a light baseline weight with moderate matched-MyoSight weight is the best overall compromise
  - repaired cohort comparisons now indicate this candidate broadly corroborates the MyoSight biological story, especially by reducing the old `IIx` inflation and recovering more `IIb`
  - the remaining decision is no longer whether to move off the old soft-gated cohort candidate; it is whether to keep tuning around `manual8_myo3_base0p1` or treat it as the standing experimental cohort reference
  - experimental review-policy work now points toward: soft `IIa` gate, relaxed `IIb` review, and top-`N` ranked gated `IIx` review per image rather than one global uncertainty gate
  - current recommendation is to stop additional manual review here unless a promotion decision or a new hotspot-specific hypothesis requires it
- Candidate behavior is **not default-ready** because the full-cohort comparison remains descriptive, the remaining `IIb`/`IIx` discrepancy is unresolved, and the new ranked `IIx` review policy is still experimental rather than release-stable.

### Minimum promotion gates

A candidate can be considered for default replacement only if all of the following are true:

- Beats the frozen baseline on the protected manual benchmark, including per-class metrics for `IIa`, `IIb`, and `IIx`.
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

The current candidate state is still `candidate-pipeline`: useful, runnable, and improving, but not an internal default candidate.

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
- Status: started and folded into the reviewed benchmark through the broad random benchmark controls from the 400-row supervision expansion.
- Priority: keep expanding carefully, but do not describe it as a final locked prevalence benchmark yet.

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
- Status: untouched upstream images are now split into:
  - `outputs/validation/promotion_holdout_generalization_manifest.csv` for development-time generalization and robustness checks.
  - `outputs/validation/promotion_holdout_generalization_reserve_manifest.csv` for a small protected backup promotion/generalization reserve.
- Current state: the first 8-image run from the development manifest should be treated as a successful `generalization_pilot` rather than promotion-grade proof.
- Pilot readout: the untouched same-project pilot was directionally stable, showed material candidate-vs-frozen composition differences, and included one repeated-source row that is useful for robustness but not ideal for final promotion claims.
- Reserve policy: keep the protected reserve out of routine candidate iteration unless a cleaner external independently labeled holdout does not materialize.
- Role: this is a promotion/generalization benchmark contract, not a MyoSight-comparable descriptive cohort unless separate MyoSight outputs are created for those images.
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
- Keep benchmark manifests role-specific:
  - `outputs/validation/myosight_validation_canonical_manifest.csv` is the touched 32-image MyoSight descriptive cohort contract.
  - `outputs/validation/promotion_holdout_generalization_manifest.csv` is the development-time untouched upstream generalization/robustness contract.
  - `outputs/validation/promotion_holdout_generalization_reserve_manifest.csv` is the small protected untouched upstream reserve for backup promotion/generalization use.
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
- If the question is only "how does a different classifier behave on the same segmented cohort?",
  prefer reclassification from an existing manifest-aligned batch root that already contains
  `*_feature_diagnostics.csv` and `*_fibers.csv`.
- Only rerun the full image pipeline when segmentation, quantification, source-path provenance,
  or feature extraction has changed. Do not rerun Cellpose just to swap classifier weights.
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

- Stable baseline workflow remains runnable and documented.
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
- Do not require convolutional neural networks (CNNs), full multi-panel generalization, or perfect MyoSight agreement for the software-note story.
- Do require reproducible commands, clean artifact provenance, and explicit limitations.

---

## v0.2.0 - Published Release: Panel-Aware Config Foundation + Workflow Hardening

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

### Progress So Far (2026-06-12)

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
    - frozen baseline feature-contract check
    - legacy rule-path regression snapshot
  - pytest split into:
    - fast synthetic default path
    - optional `integration` path

- Not done yet:
  - no blocking `v0.2.0` work remains; active scientific work has moved to `v0.3` candidate interpretation and review-policy validation

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

## v0.3 - Candidate Interpretation, MyoSight Alignment, and Review Routing

### Objectives

- Treat `manual8_myo3_base0p1` as the standing experimental reference candidate.
- Interpret the baseline-vs-candidate-vs-MyoSight evidence already generated.
- Work the remaining `IIb`/`IIx` boundary gap without treating MyoSight as perfect per-fiber ground truth.
- Define class-aware review-routing gates, especially for risky gated-`IIx` calls.
- Keep candidate behavior experimental until promotion gates are met.

### Deliverables

- Candidate state memo linking model path, split, metrics, MyoSight comparison, and review-ranker artifacts.
- Updated validation summary reflecting current `v0.3` evidence rather than older validation utilities.
- Promotion-gate checklist for moving from `candidate-pipeline` to `internal-default-candidate`.
- Hotspot-focused `IIb`/`IIx` disagreement audit, if a promotion or specific biological hypothesis requires it.
- Review-policy validation for soft `IIa` gating, broad trust in predicted `IIb`, and ranked gated-`IIx` review.
- Explicit separation of public default behavior, standing experimental candidate behavior, and experimental review-policy prototype behavior.

### Progress So Far (2026-06-12)

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
  - matched region-of-interest (ROI) MyoSight audit tooling:
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
  - pre-`IIx`/`IIb` refreshed `IIa` gate rerun on the larger reviewed benchmark:
    - ungated weighted `baseline_gb` is now about `0.805` accuracy / `0.844` balanced accuracy
    - `baseline_gb_gated_iia_q0.01` was the best candidate at that point, about `0.825` accuracy / `0.849` balanced accuracy
    - the gate drives `IIa` precision from about `0.759` to `1.000` while keeping recall about `0.939`
    - `q=0.01` looked best in that pre-boundary benchmark; stricter gates reduced balanced accuracy
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
    - candidate is only modestly better than frozen for `IIb`, and still far below MyoSight in most age/genotype groups
    - candidate remains high for `IIx` relative to MyoSight, suggesting residual/negative-call mass is still being absorbed by `IIx`
    - a first `IIa`-fail -> `IIb` redirect rule was tested on the protected manual holdout and did not change results; none of the failed `IIa` calls met the strong `IIb` redirect rule
    - any broader redirect/gate experiment should be treated as descriptive until disk pressure and validation provenance are cleaned up
  - manifest-driven full-cohort validation contract:
    - `docs/baseline_comparison_protocol.md` now documents the full-cohort validation manifest as separate from the candidate split manifest
    - `outputs/validation/myosight_validation_canonical_manifest.csv` defines the 32-image execution contract with the 7 section-export rows pinned to trusted normalized section-export inputs
    - `src/run_batch.py` now supports `--input-manifest`, so frozen and candidate paths can both run from the same image manifest
    - exact manifest-driven rerun commands are documented in `outputs/README.md`
  - focused `IIx`/`IIb` boundary audit:
    - `src/analyze_iix_iib_boundary.py`
    - `validation/analyze_iix_iib_boundary.py`
    - initial audit found `58380` predicted `IIx` rows and `1401` high-suspicion `IIb`-boundary rows
    - broader round sampled `250` fibers across the top `10` boundary-burden images using the top `15%` within-image `IIb` evidence band
    - reviewed boundary tranche was folded back into the benchmark/supervision tables
  - refreshed benchmark after the `IIx`/`IIb` boundary round:
    - benchmark rows increased to `1924`
    - protected holdout increased to `463` rows, with `457` scored canonical eval rows
    - `manual_train` increased to `1461`
    - class counts now include stronger `IIb`/`IIx` coverage (`IIb 806`, `IIx 811`, `IIa 266` in the benchmark labels)
  - refreshed candidate results after the `IIx`/`IIb` boundary round:
    - weighted `baseline_gb`: about `0.788` accuracy / `0.837` balanced accuracy
    - `baseline_gb + soft IIa gate q=0.01`: about `0.799` / `0.834`
    - `manual_high_myo_medium_baseline_light`: about `0.799` / `0.843`, best ungated recipe before the later liberal relabel and weight sweep
    - `manual_only_high`: about `0.801` / `0.837`, close enough to keep watching as manual labels grow
    - interpretation: the soft `IIa` gate remains useful for the original `IIa` false-positive problem, but it is no longer the current balanced-accuracy winner after the boundary-label update
  - multi-method MyoSight comparison plots:
    - `src/plot_multimethod_myosight_compare.py`
    - `validation/plot_multimethod_myosight_compare.py`
    - direct method plots now compare MyoSight, frozen, and the canonical candidate by total fibers, type counts, type proportions, median CSA, signal warnings, and uncalibrated review flags
    - readout: total fibers and CSA are effectively segmentation/morphometry issues shared by frozen and candidate; the typing gap is mostly candidate `IIx` excess and persistent `IIb` undercall, while `IIa` agreement is much improved
    - candidate `needs_review` panels are descriptive only because the candidate review gate is not calibrated against frozen thresholds

- Current interpretation:
  - the pipeline and evaluation machinery are now ahead of the scientific decision
  - first-pass model-family changes did not solve the problem under legacy labels; manually reviewed supervision is what moved performance
  - the positive `IIa` anchor work solved the original frozen-model `IIa` overcall pattern well enough that `IIa` is no longer the main blocker
  - after the liberal `IIb`/`IIx` relabel and focused weight sweep, the standing experimental reference is `manual8_myo3_base0p1`
  - the older `baseline_gb + soft IIa gate q=0.01` hybrid summary is now historical context, not the active experimental reference
  - repaired full-cohort MyoSight plots now say the standing experimental candidate broadly corroborates the biological story, while still leaving some `IIb`/`IIx` hotspot mismatches
  - total fiber count differences are mostly shared by frozen and candidate, so the main candidate-vs-MyoSight discrepancy is typing biology rather than segmentation count loss
  - candidate `needs_review` should not be presented as lower operational burden because the review gate is not calibrated; signal-warning comparisons are more interpretable
  - manual-only supervision is now credible, but mixed manual/MyoSight/baseline supervision still has the best balanced-accuracy result today
  - the reviewed benchmark is much stronger than it was, but it is still an evolving supervision and evaluation asset rather than a locked final benchmark

- Not done yet:
  - continue storage cleanup discipline:
    - the obvious failed/partial redirect trees have already been archived and the canonical artifact map now lives in `outputs/README.md`
    - keep archiving superseded full batch-output trees once their summary tables and manifests are preserved
  - long-term storage policy:
    - avoid copying whole batch directories for simple postprocessing tests
    - prefer summary-only analysis, manifest-driven reruns, or explicit archive-to-external-drive workflows
    - use `--retain-mode tables` for routine validation runs and keep full `*_cellpose_labels.tif` trees only for canonical reviewable runs or active segmentation debugging
  - finish section-export provenance permanently:
    - the canonical manifest exists and both batch paths can consume manifests
    - keep the 7 section-export images tied to the trusted normalized section-export branch unless/until a better section workflow is validated
    - defer expensive frozen/candidate full-cohort reruns until a selected candidate is worth rerunning
  - decide section-export segmentation robustness policy:
    - do not blindly choose the segmentation with more fibers
    - add QC-triggered fallback logic only after defining a score that accounts for `n_labels`, tissue coverage, median area, type-correlation warnings, and signal-warning burden
    - consider a section-export-specific segmentation profile if the trusted section inputs remain consistently better than the flat validation branch
  - continue the remaining `IIb`/`IIx` discrepancy work:
    - analyze what the completed boundary review actually changed in predictions and benchmark errors
    - decide whether the next tranche should be more `IIb`/`IIx` boundary supervision or broader random/manual coverage
    - test softer or probability-aware reassignment rules only after the updated benchmark and descriptive cohort support a clear target
  - candidate review-policy follow-up:
    - treat the low-burden candidate review story as resolved in the negative: the inherited frozen thresholds do not transfer
    - do not claim an operational review-burden win yet
    - revisit review triage only after more benchmark coverage or a better risk signal is available
  - keep `manual8_myo3_base0p1` as the standing experimental reference until a clearly better tuned variant appears
  - validate the selected candidate on additional heldout/external image sets before considering any default-behavior change
  - grow the manual label set further only where it adds signal, especially `IIb`/`IIx` boundary rows and broad random/manual coverage
  - repeat weighted-supervision candidate training after each meaningful expansion of the reviewed manual set
  - keep testing the three-tier supervision scheme explicitly:
    - baseline/self labels with low weight
    - matched MyoSight weak labels with medium weight
    - manual reviewed labels with high weight
    - current best recipe uses this mixed-supervision idea through `manual8_myo3_base0p1`

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

1. Clean the new multi-method comparison outputs:
   - keep signal-warning plots
   - remove or clearly relabel `needs_review` plots as uncalibrated gate outputs
   - add one slide-ready `IIb`/`IIx` age/genotype figure if needed for interpretation
2. Treat `candidate_models/weight_sweep_liberal_iib_iix_v1_focus/image_summaries/manual8_myo3_base0p1_image_summary.csv` as the current experimental descriptive cohort summary.
3. Treat `manual8_myo3_base0p1` as the standing experimental reference candidate.
4. Treat the old global `needs_review` gate as descriptive only:
   - inherited frozen thresholds do not transfer cleanly
   - current experimental direction is soft `IIa` gating plus ranked high-risk `IIx` review, not one global uncertainty threshold
5. Defer another full-cohort rerun unless a new candidate clearly beats `manual8_myo3_base0p1` on both the protected holdout and the biology-story read.
6. Keep corrected manual labels separate from stable pipeline outputs.
7. Work the remaining biology gap:
   - candidate now fixes most of the `IIa` story
  - candidate still undercalls `IIb` and overcalls `IIx`
  - use the learned gated-`IIx` risk ranker as the current review-policy prototype, not the hand-built heuristic
  - do not schedule more routine review now; only reopen review if a promotion claim or a specific hotspot analysis needs it
8. After each future meaningful reviewed tranche, rerun:
   - `validation.consolidate_reviewed_audit`
  - `validation.split_reviewed_benchmark`
  - weighted candidate training
   - supervision-recipe comparison
   - `validation.analyze_iia_gate`

## Start Here Next Time

When resuming work, start with this slice:

1. Current public default remains the frozen baseline model. Do not change default behavior.
2. Current standing experimental reference is `manual8_myo3_base0p1`, about `0.814` / `0.847` on the liberal protected manual holdout.
3. Current descriptive cohort artifact is `candidate_models/weight_sweep_liberal_iib_iix_v1_focus/image_summaries/manual8_myo3_base0p1_image_summary.csv`.
4. Main biological read:
   - `IIa` agreement is much better than frozen
   - `IIb` remains too low
   - `IIx` remains too high
5. Review-burden read:
   - the old global candidate `needs_review` gate should not be sold as the operational review policy
   - current experimental direction is: soft `IIa` gate, trust most predicted `IIb`, rank gated `IIx` by learned correction risk, and review top `N` per image
6. Next concrete work:
   - keep `manual8_myo3_base0p1` as the experimental comparator
   - treat the learned gated-`IIx` top-`N` ranker as the current experimental review-policy prototype
   - stop additional routine review unless a promotion decision or specific hotspot hypothesis requires stronger validation
   - only return to a single global threshold if a later benchmark shows it is genuinely better
7. Keep the frozen baseline as the explicit comparator for every candidate run.

Short version: the work is now about one scientific question and two engineering hygiene questions. The scientific question is the `IIb`/`IIx` residual split. The engineering questions are stable ranked `IIx` review policy and disciplined promotion/rerun decisions.

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
2. Include metric/output diffs against the frozen baseline.
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
