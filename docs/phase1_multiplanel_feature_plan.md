# Phase 1: Multi-Panel Feature Architecture Plan

> **Historical plan:** This document records the pre-implementation Phase 1 design. Statements
> describing DAPI, semantic features, review overlays, or caching as future work are retained for
> architectural context and are not a current capability matrix. See [Panel Schema](panel_schema.md),
> [Output Schema](output_schema.md), and [Review Workflow](../README_review_workflow.md) for current
> behavior.

## Status and scope

This document is the Phase 1 planning deliverable for the 2026 H3 roadmap. It specifies the
contracts needed before implementing multi-panel, Type I, eMHC, or DAPI features. It does not change
the frozen baseline workflow or activate a new classifier.

The stable workflow remains:

```text
run_pipeline/run_batch -> review_labels_napari -> merge_reviewed_labels
```

The first implementation slice should add restartable contracts underneath that workflow without
changing its legacy defaults or output meanings.

## Existing reusable seams

| Current location | Reuse in expansion |
|---|---|
| `fibertypeqc.config.ChannelConfig` | Fixed marker vocabulary, YAML loading, legacy aliases, and duplicate-index validation |
| `src.run_pipeline` | Stage orchestration, provenance, output lifecycle, retain modes, and CLI compatibility |
| `src.segment_cellpose.run_cellpose` | Fiber segmentation and the future second DAPI segmentation invocation |
| `src.quantify_classify.MarkerSpec` and marker-stat helpers | Semantic feature extraction for observed myosin/eMHC channels |
| `src.quantify_classify.build_feature_diagnostics_table` | Separate experimental/model feature table rather than expanding legacy fibers output by accident |
| `src.io_utils` | Image loading, pixel scale extraction, TIFF label writing, and CSV writing |
| `src.review_labels_napari` | Label-ID-preserving review persistence and layered Napari presentation |
| `src.merge_reviewed_labels` | Existing manual-label merge contract for legacy fiber identity |
| `src.run_batch` | Explicit manifest execution, portable input roots, batch status, and frozen-default behavior |

Candidate-model and audit code in `validation/` remains experimental. Phase 1 should define a
supported adaptation boundary, not promote existing experiment scripts wholesale.

## Proposed module responsibilities

Add new supported modules only when their first caller is implemented.

| Proposed module | Responsibility |
|---|---|
| `fibertypeqc/panels.py` | Fixed vocabulary, panel capability rules, output-request validation, and panel fingerprinting |
| `fibertypeqc/artifacts.py` | Versioned run manifest, artifact paths, stage fingerprints, cache compatibility, and invalidation decisions |
| `fibertypeqc/model_manifest.py` | Read/validate model sidecar manifests; compatibility checks before inference |
| `src/segment_nuclei.py` | Future DAPI Cellpose invocation only; no nuclear biology inference |
| `src/nuclei_features.py` | Future filtering, fiber association, radial-position measurement, and cautious nuclear terminology |
| `src/feature_schema.py` | Feature-schema version constants and semantic feature-column registry |
| `src/adaptation/` (later) | Supported project-adaptation workflow, separate from current validation scripts |

Keep `src/quantify_classify.py` as the legacy/frozen implementation until semantic feature extraction
has a tested adapter. Do not refactor it broadly in Phase 1.

## Fixed panel contract

### Vocabulary

Only these observed channels are supported:

```text
laminin, dapi, type_i, type_iia, type_iib, type_iix, emhc
```

`laminin` is the public semantic name for the existing `membrane` channel. Existing `i`, `iia`,
`iib`, and `iix` internal names remain supported through explicit aliases during migration.

### Validation rules

1. A panel contains at most four active observed channels.
2. Active channel indices are non-negative, unique, and must be checked against the loaded image
   channel count before a run starts.
3. Laminin is required for fiber segmentation.
4. DAPI is optional and enables only nuclear segmentation/feature requests.
5. Direct marker output is allowed only when that marker is observed and the selected model supports
   the panel.
6. Residual inference requires an explicit policy in the selected model manifest or panel profile.
7. Unsupported distinctions fail before processing; they must not be silently mapped to a residual
   biological class.

### Canonical config shape

```yaml
schema_version: 1
channels:
  laminin: 1
  dapi: 0
  type_i: 2
  type_iia: 3
  type_iib: null
  type_iix: null
  emhc: null
requested_domains:
  fiber_geometry: true
  fiber_identity: true
  regeneration: false
  nuclear_pathology: true
```

The legacy flat config and current nested config remain accepted through an explicit compatibility
adapter. Phase 2 must not require users of the frozen IIb/IIa/laminin workflow to rewrite configs.

## Feature and model contracts

### Feature schema

Introduce a named schema version, beginning with `multiplanel_features.v1`. Semantic columns use
known marker names, for example:

```text
type_i.mean, type_i.p90, type_i.coverage_high, type_i.center_mean, type_i.edge_mean
emhc.mean, emhc.p90, emhc.coverage_high, emhc.center_mean, emhc.edge_mean
```

The legacy frozen model retains its exact `type1_*`/`type2_*` feature contract. New semantic
features live in diagnostics or a new versioned table until a versioned output migration is approved.

### Model manifest sidecar

Every new model artifact requires a small JSON or YAML sidecar. Minimum fields:

```yaml
manifest_version: 1
model_id: example_type_i_starter_v1
task: fiber_identity | emhc_status | review_risk
feature_schema_version: multiplanel_features.v1
required_markers: [laminin, type_i, type_iia]
allowed_residual_inference: []
outputs: [type_i, type_iia, uncertain]
training_provenance:
  dataset_id: external_or_internal_identifier
  label_source: manually_reviewed
  split_unit: image
software_compatibility:
  minimum_fibertypeqc_version: 0.3.0.dev0
```

Inference must fail clearly if the panel, feature schema, or required marker set is incompatible.
Legacy joblib models remain usable only through a `legacy_frozen_alpha` adapter that pins their known
IIb/IIa/laminin assumptions and feature list.

## Artifact and cache contract

Use a versioned per-image directory without replacing legacy filenames in place:

```text
<output-root>/<image-id>/
  run.json
  fiber_labels.tif
  nuclei_labels.tif                 # only after DAPI stage exists
  fibers.csv
  feature_diagnostics.csv           # optional
  nuclei.csv                        # future optional domain
  nucleus_fiber_links.csv           # future optional domain
  summary.csv
```

`run.json` is the authoritative provenance record. It must include the Git commit, application
version, Python/Cellpose/PyTorch versions, device, source image identifier, image shape/channel
count, pixel scale, panel fingerprint, stage fingerprints, segmentation parameters, preprocessing,
model manifest, and output schema version.

| Changed input | Reuse fiber labels | Reuse nuclei labels | Recompute features/links |
|---|---:|---:|---:|
| Classifier, thresholds, review policy | Yes | Yes | Classification only |
| Myosin/eMHC feature recipe | Yes | Yes | Yes |
| Nucleus association policy | Yes | Yes | Links/nuclear summaries only |
| Fiber Cellpose parameters | No | Yes | Yes |
| Nuclear Cellpose parameters | Yes | No | Yes |
| Panel channel mapping | No by default | No by default | Yes |

The first Phase 2 slice may write `run.json` and a stable fiber-label artifact while retaining current
legacy output names through compatibility aliases.

## Output-domain schema

Do not add all future biology to `fiber_type`.

| Domain | Initial fields | Availability |
|---|---|---|
| Geometry | area, Feret, shape measures | Current/future stable |
| Fiber identity | call, source, confidence, uncertainty, direct/residual provenance | Legacy plus future versioned expansion |
| Regeneration | eMHC status, score, coverage, provenance | Only with eMHC model/panel |
| Nuclear pathology | associated count, central count, central-status, assignment confidence | Only after DAPI association |

Initial DAPI terminology is `fiber-associated nucleus`, `central interior nucleus`,
`boundary-associated nucleus`, and `unassigned/interstitial nucleus`. It must not claim that every
DAPI object is a myonucleus.

## CLI evolution

Keep all existing commands and defaults. Add options incrementally:

```text
run_pipeline --panel-config <yaml>
             --model-manifest <yaml>
             --reuse-artifacts auto|never|required
             --requested-domain fiber_identity|regeneration|nuclear_pathology

run_batch    --panel-config <yaml>
             --model-manifest <yaml>
             --reuse-artifacts auto|never|required
```

`--channel-config` remains an alias during migration. `review_labels_napari` later gains optional
`--nuclei-labels` and `--nucleus-fiber-links` arguments; these do not block Phase 2 foundation work.

## Backward compatibility and migration

1. No existing command requires a panel config or model manifest.
2. Existing frozen output folders remain readable by review and merge commands.
3. Existing joblib model use remains unchanged when no manifest is supplied.
4. New files and fields advertise explicit schema versions; no in-place rewrite of prior output
   folders is performed.
5. A migration utility, if needed later, creates a new artifact directory and provenance file rather
   than modifying raw images or old runs.
6. New optional domains are absent—not false/negative—when their required markers are unavailable.

## Test strategy

- Unit tests for marker vocabulary, aliases, maximum channel count, duplicate indices, image bounds,
  domain eligibility, and residual-inference rules.
- Model-manifest tests for compatible, missing-marker, wrong-schema, and legacy-adapter cases.
- Artifact tests for stage fingerprints, reuse, and invalidation matrix behavior.
- Snapshot tests proving frozen IIb/IIa/laminin behavior is unchanged.
- Synthetic semantic-marker feature tests with known masks and intensities.
- CLI tests for clear failure before expensive image processing.
- Later DAPI tests: synthetic central/peripheral/boundary/ambiguous nuclear geometry plus a reviewed
  full-image smoke test.

## Smallest useful implementation slice (Phase 2A)

Implement only:

1. `fibertypeqc.panels` with fixed vocabulary and four-channel/image-bound validation.
2. `fibertypeqc.artifacts` with `run.json`, stage fingerprints, and artifact path conventions.
3. `fibertypeqc.model_manifest` with the legacy frozen-alpha adapter.
4. Pipeline wiring that validates panel/model compatibility before Cellpose and writes `run.json`.
5. Tests proving the legacy default command produces its current behavior and outputs.

Do not add Type I classification, eMHC calls, DAPI segmentation, new biological output fields, or
automatic model adaptation in this slice. Those are subsequent, independently testable phases.

## Phase 1 exit criteria

- This architecture plan is reviewed and accepted.
- The fixed vocabulary, model-manifest, artifact, and output-domain contracts have no unresolved
  ambiguity for the Phase 2A implementer.
- Legacy compatibility and the first implementation slice are explicitly bounded.
- No scientific behavior has changed during Phase 1.
