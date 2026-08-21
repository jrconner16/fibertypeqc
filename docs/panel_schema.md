# Panel Schema

FiberTypeQC supports a panel-aware channel schema alongside the frozen `type1/type2` baseline.

This document defines the config contract for `--panel-config` (preferred) and
`--channel-config` (compatibility alias). The frozen baseline remains the default; panel-aware
features are explicit opt-ins.

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

## Canonical semantic schema

New configs may use the public semantic channel names directly. `--panel-config` is the preferred
flag; `--channel-config` remains a compatibility alias. At most four non-null observed channels may
be configured for one run.

```yaml
channels:
  laminin: 3
  dapi: 0
  type_i: null
  type_iia: 2
  type_iib: 1
  type_iix: null
  emhc: null
```

The legacy `membrane`/`markers` format remains accepted. Configured Type I, direct IIx, and eMHC
channels are measured in the versioned semantic diagnostics table. A compatible
`multiplanel_features.v1` candidate model can consume those features and writes predictions to a
separate `*_model_predictions.csv`; it does not replace calls in the stable `*_fibers.csv`.

eMHC measurements and manual eMHC review labels are operational, but FiberTypeQC does not promote
an automatic regeneration-status policy. Consequently, `--requested-domain regeneration` remains
intentionally unsupported. When DAPI is configured, the main pipeline automatically runs nuclear
segmentation and fiber association. Those outputs are geometric/structural measurements, not
myonucleus or pathology calls, so `--requested-domain nuclear_pathology` is also intentionally
unsupported.

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
Both the frozen `legacy_type1_type2.v1` path and manifest-declared
`multiplanel_features.v1` candidate bundles can run. Candidate bundles must declare their required
observed markers and exact features. Their predictions are isolated in
`*_model_predictions.csv`; the no-sidecar path remains the frozen IIa/IIb/laminin compatibility
adapter.

The artifact module also exposes the cache-invalidation decision matrix. Classifier or threshold
changes can reuse segmentation and recompute downstream features; changing fiber Cellpose,
preprocessing, or the panel fingerprint invalidates fiber-label reuse. Nuclear segmentation writes
its own manifest and can reuse an existing same-name nuclear label TIFF when artifact reuse is
enabled; the cached shape is checked before association tables are regenerated.

`--reuse-artifacts auto|never|required` is now available in both single-image and batch commands.
It reuses only a same-output-directory `*_cellpose_labels.tif` whose prior `run.json` has an
identical fiber-segmentation fingerprint. `required` stops before Cellpose if no compatible labels
exist; `auto` recomputes safely. Cached label shape is also checked against the current image.
`--labels-path` is the explicit corrected-mask route and cannot be combined with
`--reuse-artifacts`.

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

This gate applies to automatic calls. Manual review may assign IIx when expert review establishes
that the panel's negative-marker logic supports it; review provenance distinguishes that human
decision from an automatic residual inference.

## Current Activation Scope

Stable frozen baseline path:

- `membrane` required
- `iib` required for the frozen typing model path
- `iia` required for the frozen typing model path
- residual inferred `iix`

Operational opt-in mechanics:

- Type I, direct IIx, and eMHC feature extraction in `multiplanel_features.v1` diagnostics
- manifest-gated semantic candidate inference in a separate predictions sidecar
- DAPI nuclear segmentation, nucleus-to-fiber association, and per-fiber nuclear summaries
- Type I and eMHC manual review fields plus optional nuclear-label display
- fingerprint-gated fiber-label reuse and explicit corrected-label reuse

These mechanics do not promote a candidate model, an automatic regeneration policy, or a nuclear
pathology interpretation. Alternate residual targets likewise require explicit supported policy;
they are not inferred merely because a marker is absent.

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
