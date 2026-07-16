# Panel Schema

FiberTypeQC is moving from a fixed `type1/type2` typing model to a panel-aware channel schema.

This document defines the config contract for `--channel-config`. It describes the intended
panel model even where the current alpha pipeline still only activates a narrower default path.

## Design Goals

- Separate observed channels from derived biological calls.
- Treat `membrane` as the required structural channel for segmentation.
- Treat `dapi` as an optional structural/nuclear channel.
- Treat `I`, `IIa`, `IIb`, and `IIx` as optional direct marker channels.
- Treat residual class inference as an explicit panel policy, not an implicit assumption.

## Top-Level Structure

```yaml
channels:
  membrane: 2
  dapi: null
  markers:
    i: null
    iia: 1
    iib: 0
    iix: null

classification:
  residual_inference:
    enabled: true
    target_class: iix
    requires_negative_markers: [iia, iib]
```

## `channels`

### Required

- `membrane`: integer channel index used for segmentation

### Optional

- `dapi`: integer channel index or `null`
- `markers`: mapping with any subset of:
  - `i`
  - `iia`
  - `iib`
  - `iix`

Each configured channel index must be unique across all configured channels.
At pipeline startup, the active panel is also checked against the image: it may use no more than
four observed channels, and every configured index must be within the image channel count.

## Run provenance and model sidecars

Each successful preprocessing start writes `<image-stem>_run.json` beside the legacy outputs. It
records the resolved semantic panel, input shape and pixel scale, Cellpose/preprocessing settings,
software versions, Git commit, and stage fingerprints. It is ignored as generated output.

`--model-manifest PATH` is an optional JSON or YAML sidecar for `--classifier-path`. A manifest
must declare its version, identifier, task, feature-schema version, required observed markers, and
outputs. The pipeline rejects a model whose required markers are absent before Cellpose runs.
It also rejects a model whose declared feature schema is not available in the running pipeline.
Until Phase 3 adds semantic features, only the frozen `legacy_type1_type2.v1` schema can run;
`multiplanel_features.v1` sidecars therefore fail safely rather than receiving mismatched features.
The no-sidecar path remains the frozen IIa/IIb/laminin compatibility adapter.

The artifact module also exposes the documented cache-invalidation decision matrix. Classifier or
threshold changes reuse fiber/nuclear labels and recompute downstream features; changing fiber
Cellpose settings invalidates fiber labels, while a future nuclear Cellpose change invalidates only
nuclear labels. The current CLI does not yet perform automatic reuse—this is the tested contract
that a later `--reuse-artifacts` mode will use.

## `classification.residual_inference`

Residual inference means a named class is assigned from the absence of configured markers under a
panel-specific policy.

Fields:

- `enabled`: `true` or `false`
- `target_class`: one of `i`, `iia`, `iib`, `iix`
- `requires_negative_markers`: list of marker names that must be absent for the residual call

Example:

- Current lab default panel:
  - direct markers: `IIa`, `IIb`
  - residual target: `IIx`
  - required negatives: `IIa`, `IIb`

Important: residual inference is panel-gated. The pipeline should not assume that every omitted
class is safe to infer automatically.

## Current v0.2 Activation Scope

The schema is broader than the currently active typing logic.

Currently active baseline path:

- `membrane` required
- `iib` required for the frozen typing model path
- `iia` required for the frozen typing model path
- residual inferred `iix`

Accepted in config but not yet fully used by the default typing logic:

- `i`
- direct `iix`
- `dapi`
- alternate residual targets

Those broader fields are being added now so the config contract can stabilize before the typing
engine is generalized.

## Legacy Flat Schema

The older flat schema is still accepted for backward compatibility:

```yaml
channels:
  type1: 0
  type2: 1
  membrane: 2
```

Legacy mapping:

- `type1` -> `iib`
- `type2` -> `iia`

The legacy flat schema remains tied to the current default two-marker panel and should be
considered transitional.
