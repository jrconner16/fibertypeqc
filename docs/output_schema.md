# Output Schema

FiberTypeQC writes one output folder per image.

## Main Files

- `*_cellpose_labels.tif`: integer label mask; `0` is background.
- `*_fibers.csv`: one row per segmented fiber.
- `*_feature_diagnostics.csv`: optional model-development/debugging table written only when
  `--export-diagnostics` is enabled. The same schema is built internally for semantic candidate
  inference even when the file is not retained.
- `*_model_predictions.csv`: predictions from a compatible `multiplanel_features.v1` candidate
  bundle. This sidecar never overwrites stable fiber calls.
- `*_summary.csv`: one row with image-level processing settings, counts, QC metrics, and summaries.
- `*_run.json`: versioned run provenance and stage fingerprints used for compatible fiber-label
  reuse.
- `*_result_bundle.json`: portable, versioned index of the artifacts retained for the image.
- `*_result_report.html`: self-contained results/QC summary with portable artifact links and the
  recommended review or configuration action.
- `*_preflight_qc.json`: versioned input/config/model compatibility checks written before expensive
  processing whenever an output directory can be created.
- `*_postrun_qc.json`: versioned segmentation/typing QC checks and recommended next action.
- `*_fibers_manual_review.csv`: manual review table written by the Napari review UI.
- `*_weak_labels.csv`: fibers prioritized for review.
- `batch_summary.csv`: batch-level status table from `scripts.run_batch`.

When a DAPI channel is configured, the pipeline also writes a `nuclear/` subdirectory containing:

- `*_nuclei_labels.tif`: integer nuclear segmentation labels;
- `*_nuclei.csv`: per-nucleus geometry and conservative fiber-association status;
- `*_nucleus_fiber_links.csv`: assigned nucleus-to-fiber links;
- `*_fiber_nuclei.csv`: associated and central-interior counts per fiber;
- `*_nuclear_run.json`: nuclear settings, runtime/reuse status, outputs, and terminology.

These are structural association artifacts. They do not assert that every DAPI object is a
myonucleus or make a nuclear-pathology call.

## Result Bundle

`*_result_bundle.json` uses schema version `fibertypeqc.result_bundle.v1`. It is written after
`--retain-mode` cleanup and therefore indexes only files that are present. Artifact paths are POSIX
paths relative to the per-image output directory; private or machine-specific absolute paths are
not recorded.

Top-level fields are:

- `schema_version`: `fibertypeqc.result_bundle.v1`;
- `image_id`: the normalized input-image stem;
- `retain_mode`: `full`, `tables`, or `summary`;
- `artifacts`: mapping from a stable artifact name to its descriptor.

Each descriptor contains `path`, `kind`, `media_type`, `cardinality`, `join_keys`, and `domains`.
The initial stable artifact names are:

- fiber outputs: `fiber_labels`, `fiber_table`, `feature_diagnostics`, and
  `fiber_identity_predictions`;
- run-level outputs: `image_summary`, `preflight_qc`, `postrun_qc`, `run_provenance`, and
  `html_report`;
- optional DAPI outputs: `nuclei_labels`, `nuclei_table`, `nucleus_fiber_associations`,
  `fiber_nuclei_summary`, and `nuclear_provenance`.

When an eMHC channel is configured, retained semantic diagnostics advertise the additional
`regeneration` domain. This indicates that eMHC measurements are present; it does not constitute an
automatic regeneration call. Likewise, the `nuclei` and `association` domains describe structural
outputs, not nuclear pathology.

The complete external-consumer contract—including required fields, artifact presence rules,
cardinalities, and joins—is documented in
[Result Bundle Schema v1](result_bundle_schema.md).

## HTML Results/QC Report

Every successful pipeline run writes `*_result_report.html` and indexes it as `html_report` in the
result bundle. The report uses inline CSS and requires no network connection, JavaScript, or
external assets. It summarizes safe image-level metrics, declared domains, QC checks, artifact
availability, and limited provenance; it deliberately omits the source-image path.

The highlighted action is selected from the highest-severity readable preflight or post-run QC
report. Stable action codes remain visible alongside plain-language guidance. For example, a clean
run directs the user to manual review, while warnings can direct the user to inspect segmentation,
confirm pixel scale, or verify channel mapping/crosstalk.

To regenerate a report from an existing bundle:

```bash
uv run python -m scripts.generate_result_report \
  --bundle outputs/run/image_name/image_name_result_bundle.json
```

Artifact links are relative to the report directory. Moving the complete per-image output folder
preserves those links.

## QC Artifacts

Both QC JSON files use schema version `fibertypeqc.qc.v1` and contain:

- `stage`: `preflight` or `postrun`;
- `overall_status`: `pass`, `warn`, or `fail`;
- `recommended_next_action`: a stable machine-readable action name;
- `checks`: ordered checks with stable `code`, `status`, explanatory `message`, `next_action`, and
  optional measurements;
- `context`: input, panel, model, or generated-artifact context relevant to the stage.

Stable preflight codes are:

- `preflight.arguments_valid`;
- `preflight.channel_config_valid` and `preflight.channel_config_warning`;
- `preflight.model_artifact_valid`;
- `preflight.input_readable`;
- `preflight.panel_compatible`;
- `preflight.requested_domains_supported`;
- `preflight.model_panel_compatible`;
- `preflight.pixel_size_available`.

Stable post-run codes are:

- `postrun.fiber_count`;
- `postrun.unknown_rate`;
- `postrun.median_area`;
- `postrun.marker_correlation`.

The post-run report exposes the existing CLI QC thresholds; it does not introduce new exclusion or
classification policy. A warning never silently removes fibers. Its next action directs the user to
inspect segmentation, confirm channels/pixel scale, or proceed to manual review. Preflight failures
are recorded before processing when the command supplies enough information to create the output
directory; argument-parser failures and an unwritable output directory cannot produce an artifact.

## Fiber Table Columns

Core columns:

- `label`: segmentation label ID.
- `area`: raw label area in pixels.
- `area_um2`: raw label area in square microns when image pixel-size metadata is available.
- `feret_max_px`, `feret_min_px`: maximum and minimum Feret diameters in pixel units.
- `feret_max_um`, `feret_min_um`: Feret diameters in microns when pixel-size metadata is available.
- `area_erode_<N>px`: area after inward label erosion by `N` pixels, when requested.
- `area_erode_<N>px_um2`: eroded area in square microns when pixel-size metadata is available.

The stable `*_fibers.csv` is the conservative biological/review output. Experimental modeling
features are intentionally kept out of this table by default.

Typing features:

- `type1_mean`, `type1_p75`, `type1_p90`, `type1_pctl`, `type1_coverage`: IIb channel features.
- `type2_mean`, `type2_p75`, `type2_p90`, `type2_pctl`, `type2_coverage`: IIa channel features.
- `typing_interior_area`: pixels used for typing after `typing_erode_px`.
- `typing_erode_px`: inward erosion used for typing features.
- `typing_preprocess`: typing channel preprocessing mode.

Classification columns:

- `fiber_type`: predicted class when the selected panel supports that call; diagnostics-only
  panels use `unknown`.
- `fiber_type_source`: provenance for the current call. In `v0.2` this is one of
  `direct_marker`, `hybrid_marker`, `residual_inference`, or `model_prediction`.
- `available_markers`: pipe-delimited marker channels available for that run, for
  example `iib|iia` or `iib|iia|i|iix`.
- `classification_method`: classifier/rule source.
- `prob_iib`, `prob_iia`, `prob_iix`: model class probabilities when available.
- `model_confidence`: highest model probability.
- `model_margin`: gap between the highest and second-highest probabilities.
- `needs_review`: `True` when confidence or margin thresholds flag the fiber.
- `typing_signal_qc_flags`: signal/model consistency flags separated by `|`.
- `classifier_path`: classifier file used for the run.

## Optional Diagnostics Table

When `--export-diagnostics` is enabled, FiberTypeQC also writes `*_feature_diagnostics.csv`.

This file is intended for model-development/debugging work, not for stable biological reporting.
It may include:

- stable metadata columns such as `label`, `fiber_type`, `fiber_type_source`, and `needs_review`;
- the frozen alpha baseline model features;
- experimental coverage/SNR/extra-marker features used for diagnostics.

Semantic marker columns use names such as `type_i.mean`, `type_iix.coverage_high`, and
`emhc.snr_mean`, and advertise `feature_schema_version=multiplanel_features.v1`. eMHC is a measured
marker here; no automatic positive/negative regeneration status is added to the stable fiber table.

When a compatible semantic model manifest is supplied, the model sidecar contains `label`,
`model_prediction`, `model_id`, `task`, and any available `prob_<class>` columns.

This optional file does not change classifier predictions or expand the stable `*_fibers.csv`
schema by default.

## Review Table Columns

The Napari review UI writes:

- `fiber_id`: fiber label ID.
- `corrected_type`: manual label, normalized during merge.
- `emhc_manual_label`: separate `positive`, `negative`, or `uncertain` eMHC assessment when
  reviewed; it does not replace `corrected_type`.
- `is_uncertain`: reviewer marked uncertain.
- `is_hybrid`: reviewer marked hybrid.
- `is_excluded`: reviewer excluded the fiber.
- `label_source`: `manual_gold` for manually corrected labels.

## Merged Review Columns

`scripts.merge_reviewed_labels` preserves the original fiber columns and adds:

- `fiber_id`: integer fiber ID used for merge.
- `predicted_type`: original automatic prediction.
- `predicted_internal_type`: internal prediction label.
- `predicted_biological_type`: display/biological label mapping.
- `final_type`: prediction replaced by manual correction, uncertainty, hybrid, or exclusion status.
- `emhc_manual_label`: the separately preserved manual eMHC assessment.

## Batch Summary Columns

`batch_summary.csv` includes:

- `image_name`: image stem.
- `status`: `success` or `failed`.
- `error`: failure message when applicable.
- `fiber_count`: number of segmented fibers when successful.
- `summary_path`: per-image summary CSV path.
