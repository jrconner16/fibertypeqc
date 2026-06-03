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
