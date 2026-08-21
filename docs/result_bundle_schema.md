# Result Bundle Schema v1

`*_result_bundle.json` is the portable entry point for one successful FiberTypeQC image run. It
indexes retained artifacts without copying their data or changing their scientific meaning.

The schema identifier is:

```text
fibertypeqc.result_bundle.v1
```

Consumers should reject unsupported schema identifiers, resolve artifact paths relative to the
bundle file's parent directory, and ignore unknown descriptor fields for forward compatibility.
They should not search the output directory for undeclared files.

## Top-Level Contract

Every v1 bundle requires all four top-level fields:

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | Exactly `fibertypeqc.result_bundle.v1`. |
| `image_id` | non-empty string | Input stem normalized by the pipeline; spaces are replaced with underscores. |
| `retain_mode` | string | One of `full`, `tables`, or `summary`. |
| `artifacts` | object | Mapping from a stable artifact name to a descriptor. |

The pipeline writes a bundle only after a successful run and retention cleanup. A preflight failure
may therefore leave a standalone `*_preflight_qc.json` without a result bundle.

Artifact absence means **not produced or not retained**. It must not be interpreted as a negative
marker, identity, regeneration, nuclear, or pathology result.

## Artifact Descriptor Contract

Every entry in `artifacts` requires:

| Field | Type | Meaning |
|---|---|---|
| `path` | string | POSIX path relative to the bundle directory. It must not be absolute or escape that directory. |
| `kind` | string | `table`, `label_image`, `qc_report`, or `provenance`. |
| `media_type` | string | Currently `text/csv`, `image/tiff`, or `application/json`. |
| `cardinality` | string | Declared row/image/report grain. |
| `join_keys` | array of strings | Columns or label-image values used to join the artifact. Empty for run-level artifacts. |
| `domains` | array of strings | Output domains represented by the artifact. These describe content, not validated biological calls. |

All descriptor fields are required when an artifact entry is present. No artifact entry may point
outside the per-image output directory.

## Artifact Presence

| Artifact name | Presence in a successful run | Domain |
|---|---|---|
| `image_summary` | Required for every retain mode. | Summary |
| `preflight_qc` | Required for every successful bundle. | Quality control |
| `postrun_qc` | Required for every successful bundle. | Quality control |
| `run_provenance` | Required for every successful bundle. | Provenance |
| `fiber_table` | Required for `full` and `tables`; removed by `summary`. | Fiber geometry and identity |
| `fiber_labels` | Required for `full`; removed by `tables` and `summary`. | Fiber geometry |
| `feature_diagnostics` | Optional; written with `--export-diagnostics` and retained by `full` or `tables`. | Fiber identity; also regeneration measurements when eMHC is configured |
| `fiber_identity_predictions` | Optional; written only by a compatible semantic candidate model. | Fiber identity candidate predictions |
| `nuclei_labels` | Required when a configured DAPI channel activates the nuclear stage. | Nuclear segmentation |
| `nuclei_table` | Required when the nuclear stage runs. | Nuclei and association |
| `nucleus_fiber_associations` | Required when the nuclear stage runs; the CSV may contain only its header when no nucleus is assigned. | Association |
| `fiber_nuclei_summary` | Required when the nuclear stage runs. | Nuclei and association |
| `nuclear_provenance` | Required when the nuclear stage runs. | Nuclear provenance |

The v1 bundle is created by `run_pipeline`; downstream manual-review and merged-review CSVs are not
yet indexed. Tools must accept their absence rather than guessing filenames.

## Join Graph and Cardinality

All joins are scoped to the single image identified by the bundle.

| From | Key | To | Key | Relationship |
|---|---|---|---|---|
| `fiber_labels` pixel value | positive integer label | `fiber_table` | `label` | One label value to exactly one fiber row. `0` is background. |
| `fiber_table` | `label` | `feature_diagnostics` | `label` | One-to-one when diagnostics are retained. |
| `fiber_table` | `label` | `fiber_identity_predictions` | `label` | One-to-one when candidate predictions are present. |
| `nuclei_labels` pixel value | positive integer label | `nuclei_table` | `nucleus_id` | One label value to exactly one nucleus row. `0` is background. |
| `nuclei_table` | `nucleus_id` | `nucleus_fiber_associations` | `nucleus_id` | One-to-zero-or-one; only assigned nuclei receive a link row. |
| `nuclei_table` | `assigned_fiber_id` | `fiber_table` | `label` | Many-to-one only where `assignment_status=assigned`. `0` means unassigned/interstitial; ambiguous rows may carry a provisional best-overlap ID but must not be joined. |
| `nucleus_fiber_associations` | `fiber_id` | `fiber_table` | `label` | Many assigned nuclei to one fiber. |
| `fiber_nuclei_summary` | `fiber_id` | `fiber_table` | `label` | Exactly one summary row per segmented fiber. |

`label` and `fiber_id` refer to the same fiber-label namespace within one image. External tools
should preserve integer types and must not join these identifiers across bundles without also using
the bundle's `image_id`.

## Table Schemas

### Stable fiber table

`fiber_table` has one row per positive fiber label. The stable core includes:

- identity and geometry: `label`, `area`, `feret_max_px`, and `feret_min_px`;
- typing context: `available_markers`, `typing_interior_area`, `typing_erode_px`, and
  `typing_preprocess`;
- call and provenance: `fiber_type`, `fiber_type_source`, `classification_method`, and
  `classifier_path`;
- review/QC routing: `needs_review` and `typing_signal_qc_flags`.

Pixel-scale columns such as `area_um2` and `feret_max_um` are conditional on readable pixel-size
metadata. Legacy `type1_*`/`type2_*` measurements require the IIb/IIa pair. Model probability and
confidence columns may be absent or null when the active panel has no compatible classifier.

### Semantic feature diagnostics

`feature_diagnostics` has one row per fiber and requires `label` and
`feature_schema_version=multiplanel_features.v1`. Metadata columns are copied when available.
Marker feature families are conditional on observed channels and use semantic prefixes such as:

- `type_i.*`, `type_iia.*`, `type_iib.*`, and `type_iix.*`;
- `emhc.*` when eMHC is configured.

Each observed marker may expose `mean`, `p75`, `p90`, `pctl`, `coverage_high`, `snr_mean`, and
`snr_p90`; center/edge fields are conditional on spatial-feature collection. eMHC columns are
measurements, not an automatic regeneration status.

### Semantic candidate predictions

`fiber_identity_predictions` requires `label`, `model_prediction`, `model_id`, and `task`.
`prob_<class>` columns are conditional on the candidate model exposing `predict_proba`. These calls
do not overwrite `fiber_table.fiber_type`.

### Image summary

`image_summary` contains exactly one row. It records artifact paths, resolved channels, processing
settings, label/area summaries, class composition, and QC flags. Nuclear artifact paths are
conditional on the DAPI stage. Use the bundle rather than summary path fields for portable artifact
discovery; legacy summary paths may reflect how the pipeline was invoked.

### Nuclear tables

`nuclei_table` requires:

- `nucleus_id`, `area_px`, `centroid_y_px`, and `centroid_x_px`;
- `assigned_fiber_id`, `assignment_status`, and `association_category`;
- `overlap_fraction`, `distance_to_boundary_px`, and `normalized_radial_position`.

`assignment_status` is `assigned`, `ambiguous`, or `unassigned_or_interstitial`. An assigned nucleus
is categorized as `boundary_associated`, `central_interior`, or `peripheral_associated`. These are
geometric categories, not automatic myonucleus calls.

`nucleus_fiber_associations` requires `nucleus_id`, `fiber_id`, `assignment_status`,
`association_category`, and `overlap_fraction`. It contains assigned links only.

`fiber_nuclei_summary` requires `fiber_id`, `associated_nuclei_count`, `central_nuclei_count`, and
`centrally_nucleated`. The final boolean reports a central-interior geometric association; it is not
a nuclear-pathology diagnosis.

## JSON Artifacts

`preflight_qc` and `postrun_qc` use `fibertypeqc.qc.v1`; their stable fields and codes are documented
in [Output Schema](output_schema.md#qc-artifacts). `run_provenance` records the panel mapping,
software versions, source metadata, processing settings, and stage fingerprints. `nuclear_provenance`
records DAPI segmentation/association settings and explicitly uses the terminology
`fiber-associated nuclei; no automatic myonucleus call`.
