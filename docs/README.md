# Documentation

## Using FiberTypeQC

- [install.md](install.md): requirements, install, HPC notes, model downloads.
- [demo.md](demo.md): the whole workflow on four small public fields.
- [user_guide.md](user_guide.md): projects, workflow, outputs, QC, models, troubleshooting.
- [review_guide.md](review_guide.md): reviewing a project in Napari.
- [review_reference.md](review_reference.md): project QC, dashboard, and reviewer components in
  detail.
- [panel_schema.md](panel_schema.md): panel files (which channel holds which stain).
- [output_schema.md](output_schema.md): columns of every output table.
- [result_bundle_schema.md](result_bundle_schema.md): per-image result bundles and reports.
- Model cards: [quad_four_class_rf_v1](model_cards/quad_four_class_rf_v1.md),
  [ta_three_class_logistic_v1](model_cards/ta_three_class_logistic_v1.md).

## Developing FiberTypeQC

- [../ARCHITECTURE.md](../ARCHITECTURE.md): supported surface, contracts, code areas.
- [../CONTRIBUTING.md](../CONTRIBUTING.md): setup, checks, and rules for changes.
- [model_registry.md](model_registry.md): how models are registered and resolved.
- [modeling.md](modeling.md): feature contracts and candidate-model workflow.
- [baseline_comparison_protocol.md](baseline_comparison_protocol.md): comparing a candidate model
  with the frozen baseline.
- [validation_summary.md](validation_summary.md): research comparison tools.
- [baselines/ta_emhc_nuclei_v1.md](baselines/ta_emhc_nuclei_v1.md): example nuclear segmentation
  settings.

## History

[history/](history/) keeps superseded roadmaps, design plans, and the v0.1 demo manifest for
reference. They do not describe current behavior.
