---
title: 'FiberTypeQC: auditable fiber typing with quality control and guided review for skeletal muscle immunofluorescence'
tags:
  - Python
  - skeletal muscle
  - fiber typing
  - image analysis
  - microscopy
  - quality control
  - napari
authors:
  - name: James R. Conner
    orcid: 0000-0002-8886-1475
    affiliation: 1
  - name: Felipe de Souza Leite
    orcid: 0000-0003-2493-8872
    affiliation: "1, 2"
  - name: Louis M. Kunkel
    orcid: 0000-0001-6126-0242
    affiliation: "1, 3"
affiliations:
  - name: Division of Genetics and Genomics, Boston Children's Hospital, Boston, MA, United States
    index: 1
  - name: Department of Pediatrics, Harvard Medical School, Boston, MA, United States
    index: 2
  - name: The Manton Center for Orphan Disease Research, Boston Children's Hospital, Boston, MA, United States
    index: 3
date: 19 November 2026
bibliography: paper.bib
---

<!--
DRAFT STATUS: REMOVE THIS COMMENT, ALL AUTHOR NOTES, AND THE CLAIM MAP BEFORE SUBMISSION.
- Describes v0.5.1 (released 2026-11-17; Zenodo concept DOI 10.5281/zenodo.23893717). The paper
  claims released features only. Set `date` to the submission date.
- Submit after 2026-11-30: JOSS requires the repository to have been public for more than six
  months (created 2026-05-31).
- Revised 2026-11-19: positive contribution statement in State of the field; new Software design
  paragraphs on building a model and on finalized outputs; staged review rewritten; research
  impact restructured; two figures (paper/figures/make_figures.py); AI disclosure and
  acknowledgements restructured with placeholders.
- Bracketed placeholders [like this] in the text must be filled or removed by the authors.
- Leave out of the paper: eMHC builder classes (experimental, synthetic tests only); nuclei in
  results (experimental, central cut-off unchecked); any accuracy figures.
- JOSS limit: 750-1750 words, excluding front matter, comments, captions, and references.
-->

# Summary

FiberTypeQC is an open-source Python workflow for skeletal muscle immunofluorescence analysis that
combines Cellpose segmentation [@Stringer2021Cellpose; @Stringer2025Cellpose3], feature-based
fiber-type classification, morphometry, and guided quality-control review. It uses panel-specific
scikit-learn [@Pedregosa2011sklearn] models to classify fibers and a napari [@napari] interface to
inspect questionable results. Rather than replacing automated predictions with manual corrections,
FiberTypeQC preserves both and records their provenance in finalized per-fiber and cohort-level
tables and an HTML report. Researchers whose staining panel has no released model can build one
from blind reference labels and adopt it through an explicit, reversible step. Its purpose is to
make high-throughput fiber typing practical without sacrificing human oversight or traceability.
The software is archived on Zenodo [@Conner2026FiberTypeQC].

# Statement of need

Skeletal muscle fiber typing is widely used to study muscle physiology, disease, and regeneration.
Traditional approaches require substantial manual intervention or user-selected fluorescence
thresholds. These methods can produce useful results, but differences in staining intensity, image
quality, and threshold selection complicate comparisons between experimental cohorts.

FiberTypeQC originated from the need to analyze large collections of muscle sections without
manually inspecting every fiber; one internal analysis involved approximately 200,000 fibers.
Automation reduced this workload but introduced a different problem: classifiers could produce
plausible predictions on images whose staining characteristics differed from their training data.

Type IIx illustrates the difficulty. In commonly used staining panels, IIx is inferred from the
absence of other myosin heavy-chain signals. A weakly stained IIa or IIb fiber may therefore
resemble an IIx fiber, despite representing a different biological phenotype.

The practical challenge is not simply automating classification, but identifying where automation
may be unreliable, directing expert attention there, and keeping a record of what the expert
changed. FiberTypeQC is intended for muscle laboratories analyzing cohorts too large for
exhaustive manual typing, and for the computational staff who support them.

# State of the field

Muscle-specific tools including MyoSight [@Babcock2020MyoSight], Myosoft
[@EncarnacionRivera2020Myosoft], MuscleJ [@MayeufLouchart2018MuscleJ], MuscleJ2
[@Danckaert2023MuscleJ2], QuantiMus [@Kastenschmidt2019QuantiMus], and FiNuTyper
[@Lundquist2023FiNuTyper] provide automated or semi-automated segmentation, fiber typing,
morphometry, and related measurements, mostly as Fiji plugins or standalone applications.
General-purpose platforms such as QuPath [@Bankhead2017QuPath] and ilastik [@Berg2019ilastik]
support segmentation, feature extraction, interactive classification, and model training.

FiberTypeQC contributes a muscle-specific analysis system in which three properties are enforced by
the software rather than left to laboratory convention. First, quality-control checks direct
expert review to the fibers and sections most likely to need it, and reviewer decisions are stored
beside, never over, the model's predictions. Second, researchers can build or improve a classifier
for their own staining panel, evaluate it on animals held out in advance, and promote or roll it
back explicitly. Third, provenance carries from model identity and input digests to the source of
every finalized value. FiberTypeQC builds on established components (Cellpose, scikit-learn,
napari) and does not introduce new segmentation or classification algorithms.

We built a new tool rather than extending an existing one because these properties require owning
the data path from prediction through review to finalized tables across a whole project. Existing
muscle tools center on measuring images, and general-purpose platforms leave conventions for
corrections, model compatibility, and cohort aggregation to each laboratory. The cost is less
flexibility than a general-purpose platform; the benefit is safeguards that apply by default.

# Software design

FiberTypeQC's design balances automation, interpretability, adaptability, and reproducibility
(\autoref{fig:workflow}).

![FiberTypeQC workflow. Top: analysis from images to finalized tables and report. Lower left: building a model for a panel without one, from blind labels, checked on held-out test animals. Lower right: improving a model from review corrections. Both change the project's model only through explicit, reversible promotion.\label{fig:workflow}](figures/workflow.png)

**Project-level workflow.** A command-line interface manages initialization, image analysis, QC
preparation, review, finalization, and reporting for a project folder, and reports the next step.
An experimental desktop launcher runs the same steps as the same commands, so results do not
depend on the interface. This staged approach requires more workflow management than a single
processing command, but makes the analysis state and outstanding review explicit across a cohort.
The same commands run on a workstation or a cluster: batch processing has been run in Slurm
[@Yoo2003Slurm] jobs and review in Open OnDemand [@Hudak2018OnDemand] remote desktops.

**Interpretable feature-based classification.** Cellpose segments fibers from the laminin
channel, while fiber typing uses conventional supervised models (logistic regression, random
forests, gradient boosting) trained on explicit fluorescence measurements, including intensity
percentiles, high-signal coverage, and signal-to-noise summaries for each marker. This was
preferred over an end-to-end neural-network classifier because researchers can inspect the
measurements behind an ambiguous prediction, investigate staining-related failures, and train new
models from the modest number of labels a laboratory can realistically collect. The tradeoff is
dependence on feature quality and staining consistency; interpretable inputs do not guarantee
accurate predictions.

**Panel-specific models.** A model is associated with a declared staining panel, feature schema,
and extraction settings, and is identified by a SHA-256 digest; incompatible configurations are
rejected. Rather than assuming that a classifier transfers across arbitrary antibody and channel
combinations, FiberTypeQC requires an explicitly compatible model. This limits immediate
generality but reduces the risk of silently applying a model to unsuitable data. IIx identified
by exclusion is labelled as inferred.

**Quality-control triage.** Image- and fiber-level checks, including classification uncertainty,
staining characteristics, and segmentation-related features, prioritize review, and each flagged
fiber is shown with the reason it was flagged. Flags do not invalidate measurements, and unflagged
predictions are not guaranteed to be correct; the effectiveness of QC prioritization has not yet
been quantified.

**Non-destructive, staged review.** The napari window (\autoref{fig:review}a) works one stage at a
time: a verdict on each section (accept it, draw areas to leave out, or leave it out), then the
flagged fibers in priority order, with any other fiber available by clicking it, then
finalization. Section and area verdicts are stored as the same exclusion records that finalization
already applies, so the simpler window adds no second path through the data. Decisions are kept in
review state and an append-only event log, separately from predictions, and input digests prevent
them from being silently applied to changed masks or measurements. This costs more bookkeeping than
editing a results table, but preserves the automated analysis and distinguishes human
interventions from model outputs.

![Outputs on the public demonstration data. (a) The review window: fibers outlined in the color of their current call; the panel shows the stage, why the current fiber was flagged, and the decision controls. (b) The cohort report: provenance, and one animal's composition before and after an example review. The IIx shift is one reviewer's decisions on demonstration fields, not a validation.\label{fig:review}](figures/review_and_report.png)

**Building a model for a new panel.** A project can start without a model: fibers are segmented
and measured but left unclassified. Before any labels exist, some animals are set aside for
testing. A reviewer then labels a random sample of fibers blind, so labels cannot be steered by a
model's answer or by hand-picking. Candidate classifiers are compared by leave-one-animal-out
validation against a majority-class baseline, under minimum label counts and recommendation
criteria fixed in a recipe before training. The recommended candidate is evaluated on the test
animals and changes the project only when a user promotes it; promotion is recorded and can be
rolled back exactly. This adds up-front labelling compared with ad hoc training, but researchers
are no longer limited to the panels for which models are distributed. Requirements and criteria
are documented with the software.

**Controlled model improvement.** For a panel that already has a model, candidates can be
retrained from reviewed corrections, kept separate from blind reference labels, and compared with
the current model under a promotion rule declared in a versioned recipe. Promotion is again
explicit and reversible: less automatic than continuously updating a classifier, but controlled
and auditable.

**Finalized outputs.** Finalization writes new tables rather than modifying analysis outputs. Each
fiber keeps its model prediction beside its final type, morphometry (area, Feret diameters), and a
value source: `predicted`, `reviewed`, `excluded`, or `unresolved`. Summary tables report counts,
composition, and fiber size per image, animal, region of interest, and cohort, with exclusions
counted by reason. A self-contained HTML report (\autoref{fig:review}b) records the model, code
revision, and final-table digest, and compares predicted with finalized composition. The tables
are ready for downstream statistics, and every value traces to the model or to a reviewer.

**Verification and reproducibility.** Continuous integration checks supported commands,
repository boundaries, and synthetic reference analyses; a locked environment defines
dependencies. On the public demonstration images, batch outputs from a Linux cluster (NVIDIA GPU)
agreed with a macOS reference (Apple GPU) within snapshot tolerances: fiber counts differed by at
most 0.36% and fiber-type proportions by at most 0.29 percentage points per section. These checks
support computational reproducibility; they do not replace biological validation.

# Research impact statement

FiberTypeQC is used within an active skeletal muscle research laboratory, in mouse tibialis
anterior and quadriceps studies covering segmentation, classification, morphometry, QC review, and
cohort-level outputs; the two released models correspond to these staining panels. This in-house
use established research-scale operation, not classification performance.

FiberTypeQC has also been applied retrospectively to imaging data from published JAG1 muscle
research [@deSouzaLeite2025JAG1], comparing its morphometric and fiber-typing outputs with an
established threshold-based analysis. This comparison informed the emphasis on explicit quality
control and review; it is not presented as evidence of superior accuracy.

Two public demonstrations distributed with the release run end to end: four fields from images to
a reviewed cohort report, and a model-builder project with blind labels taken from no model
through building, test evaluation, promotion, and exact rollback.

An outside researcher installed FiberTypeQC and ran the demonstration workflow on the example
data; their feedback informed usability changes, such as numbering image channels from one as
microscope software does. This was a usability test, not research adoption, and the graphical
launcher has not yet been evaluated externally. Validation across laboratories and staining
conditions remains future work.

# AI usage disclosure

Generative AI tools were used during development of the software and this manuscript: ChatGPT
[model and version], OpenAI Codex [model and version], and Claude Code with Anthropic Claude Opus
models [versions]. They assisted with code implementation and refactoring, debugging, test
scaffolding, documentation, and drafting and editing of manuscript text.

James R. Conner directed the scientific objectives, software architecture, analytical workflows,
and development priorities, and made the core design decisions. All AI-assisted code,
documentation, and text were reviewed, edited, and validated by the authors, including through the
software's test suite and research analyses. The authors take full responsibility for the
accuracy, originality, and licensing of the software and manuscript.

# Acknowledgements

[Funding sources and grant numbers.] [Statement of whether the funders had any role in the design,
development, or reporting of the software.] [Thanks to the outside tester, with their consent.]
The authors declare no competing interests. [Authors: confirm.]

# References

<!--
AUTHOR NOTES (REMOVE BEFORE SUBMISSION)
- Statement of need, para 2: the 200,000-fiber figure appears here and once in Research impact;
  keep one if space is needed. It shows research-scale use only.
- State of the field: the sentence on Fiji tools being "organized around measuring individual
  images" and general platforms leaving conventions to each lab is a characterization; check it
  against each cited tool before submission.
- "train new models from the modest number of labels a laboratory can realistically collect": the
  builder demo uses 731 blind labels; the recipe minimum is 30 per class. This is a design
  rationale, not a performance claim; cut it if a reviewer reads it otherwise.
- Outside tester sentence: the user's account (2026-11-19) is that they ran the demonstration
  workflow on example data and later could not evaluate the launcher. ROADMAP C3 currently says
  more than this; fix ROADMAP separately. Confirm which changes their feedback led to (#83, #84,
  #90) and get consent before thanking them by name.
- Figure 2b: the public completed demo contains one fiber finalized as Type I (demo_mouse_2,
  demo_section_b, label 39) on a panel with no Type I stain; the report legend therefore shows
  Type I. The crop shows only demo_mouse_1 for this reason. The decision predates the current
  window, which offers only the model's classes, but finalization does not check classes against
  the panel. Consider correcting the demo and adding that check before the acceptance release.
- Version label: fibertypeqc/__init__.py reports 0.5.0 in the v0.5.1 release, so outputs record
  application_version 0.5.0. Fix in the release made at acceptance.

Claim-to-evidence map (REMOVE BEFORE SUBMISSION)
- One CLI over a project folder; status reports the next step: src/cli.py (build_parser).
- Panel mismatch refused; digest verified: fibertypeqc/model_manifest.py,
  fibertypeqc/model_resolution.py; tests/test_model_manifest.py, tests/test_model_resolution.py.
- Pinned feature extraction: tests/test_model_resolution.py
  (test_quad_manifest_pins_every_feature_setting, test_manifest_rejects_partial_feature_pins).
- Candidate families (logistic, random forest, histogram gradient boosting):
  manifests/builder/recipe.v1.yaml, manifests/improver/recipe.v1.yaml.
- QC flags with reasons: src/review/guided_review_widget.py ("Why shown", describe_reason).
- Predictions untouched; changed inputs refused; value_source labels: src/review/finalization.py;
  tests/test_review_finalization.py (test_unreviewed_fibers_keep_predictions_and_inputs_are_untouched,
  test_changed_inputs_and_changed_predictions_are_refused).
- Staged review, section and area verdicts read by finalization, click-to-check, no clicking in
  blind mode: src/review/section_stage_widget.py, src/review/guided_review_widget.py;
  tests/test_review_stages.py.
- Blind mode hides calls and names; evaluation roles: tests/test_blind_reference.py.
- Projects without a model (segment and measure, calls unclassified): tests/test_no_model_project.py.
- Builder recipe and recommendation rule (min 3 training mice; 30 labels per class from 2+ mice
  with 5+ each; median per-mouse macro-F1 >= baseline + 0.10; pooled recall >= 0.70 per class):
  manifests/builder/recipe.v1.yaml; src/model_builder.py; tests/test_model_builder.py
  (test_build_end_to_end, test_minimums, test_baseline_is_the_majority_of_each_folds_training_mice,
  test_decide_applies_both_conditions_and_breaks_ties_in_recipe_order,
  test_training_features_exclude_area_dapi_and_emhc).
- Test mice set aside automatically before labels, never trained on: tests/test_test_mice.py.
- Test evaluation, promotion, exact rollback, recorded overrides:
  tests/test_model_evaluation_promotion.py
  (test_evaluate_promote_and_rollback_restore_the_project_exactly,
  test_promotion_needs_a_current_evaluation_or_a_recorded_override,
  test_recall_warning_names_classes_below_the_bar).
- Improver promotion rule: manifests/improver/recipe.v1.yaml; src/model_improver.py;
  tests/test_model_improver.py (test_too_few_mice_never_yields_a_recommendation).
- Explicit promotion and rollback: tests/test_promote_model.py.
- Finalized outputs: final/final_fiber_table.csv (prediction, final_type, value_source);
  results/image_summary.csv, mouse_summary.csv, cohort_summary.csv (exclusions by reason),
  ROI summaries when ROIs are defined; results/cohort_report.html (model, code revision,
  final-table SHA-256, predicted vs finalized composition). docs/output_schema.md; docs/demo.md.
- Launcher runs the same commands as the CLI: src/launcher.py; tests/test_launcher.py
  (test_launcher_and_status_agree_on_the_next_step).
- HPC: docs/install.md, "HPC clusters (Slurm and Open OnDemand)". Slurm job scripts are the
  user's; FiberTypeQC does not orchestrate arrays (scripts/slurm_panel_array.sh is an example).
- Cross-platform agreement (0.36%, 0.29 pp): snapshot comparison on the demo, ROADMAP "Current
  state"; tests/test_reference_snapshot.py.
- Snapshots: tests/test_reference_snapshot.py. Core/research boundary: tests/test_core_boundary.py.
- CI and reference run: .github/workflows/ci.yml, scripts/run_reference.py.
- Model evidence: docs/model_cards/.
- Demos: release downloads fibertypeqc_demo.zip and fibertypeqc_builder_demo.zip; docs/demo.md.
  Builder demo checked from the zip: build recommends a model; rollback restores every file.
- Figure 1 is a schematic drawn by paper/figures/make_figures.py. Figure 2 combines the unedited
  screenshots docs/images/demo_review_workspace.png and docs/images/report_composition_light.png
  (cropped to the header and demo_mouse_1); demo_mouse_1 values match
  demo_project_completed/results/mouse_summary.csv (IIx 25.6% predicted, 15.8% finalized).
- Releases: v0.5.0 (Zenodo 10.5281/zenodo.24321794), v0.5.1 (2026-11-17); concept DOI
  10.5281/zenodo.23893717.
-->
