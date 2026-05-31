# FiberTypeQC v0.1-alpha Model Card

## Model

- File: `rebaseline_tile_v2_p75p90_iib_iia_iix.joblib`
- Task: per-fiber classification into `iib`, `iia`, or `iix`
- Framework: scikit-learn-compatible joblib classifier
- Public status: default v0.1-alpha model

## Intended Use

The model is intended for tibialis anterior immunofluorescence fiber typing where:

- one channel marks type IIb signal,
- one channel marks type IIa signal,
- one membrane/laminin channel supports fiber segmentation,
- type IIx is inferred as fibers without clear IIb or IIa signal.

The model is designed for batch pre-labeling plus human review, not fully unattended final
analysis.

## Current Channel Schema

The v0.1-alpha CLI uses legacy argument names:

- `--type1-channel`: IIb marker channel
- `--type2-channel`: IIa marker channel
- `--membrane-channel`: membrane/laminin segmentation channel

This release does not yet include a general marker-panel schema. Users with different staining
orders must explicitly pass the correct channel indices. Arbitrary fiber-type panels are out of
scope for v0.1-alpha.

## Outputs

The model writes:

- `fiber_type`: hard class label
- `prob_iib`, `prob_iia`, `prob_iix`: class probabilities when available
- `model_confidence`: highest class probability
- `model_margin`: difference between the top two class probabilities
- `needs_review`: review flag from confidence/margin thresholds
- `typing_signal_qc_flags`: signal/model consistency flags

## Limitations

- The model has been developed on a narrow lab-specific image domain.
- Channel order and staining quality matter.
- Type IIx is an inferred negative class, so weak signal and background artifacts can affect calls.
- Probability values are useful for prioritizing review, but they should not be treated as fully
  calibrated biological probabilities yet.
- Visual QC is required for v0.1-alpha outputs.

## Recommended Alpha Use

Run the batch pipeline, inspect summary outputs, review flagged fibers in Napari, and merge manual
corrections before using results for figures or biological interpretation.
