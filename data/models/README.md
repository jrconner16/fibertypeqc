# Models

The v0.3.0.dev0 development workflow exposes one frozen default classifier:

`rebaseline_tile_v2_p75p90_iib_iia_iix.joblib`

This model is the frozen alpha baseline used by the documented CLI examples. Other local model
artifacts may exist during development, but they are not part of the stable public workflow.

See [model_card.md](model_card.md) for intended use, channel assumptions, and limitations.

The matching versioned sidecar is
[`rebaseline_tile_v2_p75p90_iib_iia_iix.yaml`](rebaseline_tile_v2_p75p90_iib_iia_iix.yaml). It
declares the task, required observed markers, feature schema, outputs, intended use, artifact name,
and SHA-256 digest used by the executable reference workflow.
