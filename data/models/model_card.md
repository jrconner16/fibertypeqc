# FiberTypeQC Frozen Alpha Model Card

## Model Identity

- File: `data/models/rebaseline_tile_v2_p75p90_iib_iia_iix.joblib`
- Public status: frozen default alpha classifier
- Framework: scikit-learn-compatible joblib model
- Task: per-fiber classification into `iib`, `iia`, or inferred `iix`
- Manifest: `data/models/rebaseline_tile_v2_p75p90_iib_iia_iix.yaml`
- SHA-256: `5a1042a31ad4d90be61a58fabf9204e52517e17b63baccf027c8d84db1d9f946`

This is the baseline model used by the documented public alpha workflow. Candidate models should be
compared against this frozen baseline before any default-model change is considered.

## Intended Use

This model is intended for review-assisted skeletal muscle fiber typing in immunofluorescence
images where:

- one channel marks direct IIb signal;
- one channel marks direct IIa signal;
- one membrane/laminin channel supports segmentation;
- type IIx is treated as the residual unstained class relative to IIb and IIa.

The intended use is:

- batch pre-labeling;
- feature extraction;
- review prioritization;
- manual correction before biological interpretation.

It is not intended to be treated as a fully unattended final-analysis system.

## Non-Use / Out of Scope

This frozen alpha model should not be assumed valid without validation for:

- arbitrary marker panels beyond the current IIb + IIa + membrane alpha path;
- protocols where the residual unstained class is not biologically interpretable as IIx;
- direct type I or direct type IIx panels as a default model path;
- strong domain shifts in stain quality, imaging conditions, tissue type, or segmentation behavior;
- claims that probability values are calibrated biological probabilities.

## Current Marker / Channel Assumptions

Current alpha typing assumptions:

- IIb direct marker channel
- IIa direct marker channel
- membrane/laminin structural channel
- IIx inferred from absent IIb/IIa signal

Current public CLI semantics prefer:

- `--iib-channel`
- `--iia-channel`
- `--membrane-channel`

Legacy aliases still exist:

- `--type1-channel` -> IIb
- `--type2-channel` -> IIa

The repo now contains a broader panel-aware config foundation, but this frozen model still assumes
the narrow IIb/IIa residual-IIx alpha workflow.

## Class Definitions

Current model output classes:

- `iib`: direct-positive IIb call
- `iia`: direct-positive IIa call
- `iix`: residual/inferred class for fibers without clear IIb or IIa evidence under the alpha panel

Important caveat:

- `iix` in this frozen model is not a direct-marker class. It is the inferred residual class used
  by the alpha workflow.

## Features Used by the Frozen Model

The frozen model expects this exact feature contract:

1. `area`
2. `type1_mean`
3. `type2_mean`
4. `type1_p75`
5. `type2_p75`
6. `type1_p90`
7. `type2_p90`
8. `type1_pctl`
9. `type2_pctl`
10. `type1_coverage`
11. `type2_coverage`
12. `type_ratio`
13. `type_diff`
14. `type_pctl_ratio`
15. `type_pctl_diff`
16. `type_p75_ratio`
17. `type_p75_diff`
18. `type_p90_ratio`
19. `type_p90_diff`
20. `type_cov_ratio`
21. `type_cov_diff`

Notes:

- `type1` is the legacy internal feature name for the IIb marker channel.
- `type2` is the legacy internal feature name for the IIa marker channel.
- These names remain because the frozen model was trained on them.

## Diagnostics / Future Features

The repo now contains internal and optional diagnostic support for richer feature analysis, but
those features are not part of the frozen alpha model contract by default.

Examples of experimental or future-facing features:

- expanded coverage summaries:
  - `type_cov_sum`
  - `type_cov_max`
  - `type_cov_min`
  - `type_cov_balance`
  - `type_cov_product`
- background-relative / SNR-style features:
  - `type1_snr_mean`
  - `type2_snr_mean`
  - `type1_snr_p90`
  - `type2_snr_p90`
  - `type_snr_ratio`
  - `type_snr_diff`
  - `type1_cov_x_snr`
  - `type2_cov_x_snr`
- optional extra-marker diagnostic features:
  - `marker_i_*`
  - `marker_iix_*`

These features may appear in optional diagnostics output such as
`*_feature_diagnostics.csv`, but they do not change the stable `*_fibers.csv` schema or the frozen
model behavior by default.

For the fuller rationale behind these feature gaps and future directions, see
[docs/modeling.md](../../docs/modeling.md).

## Outputs

The model-related outputs include:

- `fiber_type`: hard predicted class
- `fiber_type_source`: provenance for the current call
- `classification_method`: rule/model source label
- `prob_iib`, `prob_iia`, `prob_iix`: class probabilities when available
- `model_confidence`: highest class probability
- `model_margin`: gap between the highest and second-highest probabilities
- `needs_review`: review flag based on configured review policy
- `typing_signal_qc_flags`: signal/model consistency warnings
- `classifier_path`: model file used for the run

Optional diagnostics export may additionally include:

- frozen alpha baseline model features
- experimental coverage/SNR/extra-marker features

## Probability / Confidence / Margin / Entropy

Current frozen alpha outputs include:

- per-class probabilities when the underlying model exposes `predict_proba`
- `model_confidence`
- `model_margin`

Current use of these values:

- prioritize manual review
- identify low-separation or low-confidence calls
- support diagnostics

Current limitations:

- probabilities should not be treated as fully calibrated biological probabilities;
- confidence and margin quantify classifier certainty, not biological truth;
- ambiguity in these outputs does not prove hybridity.

The repo’s broader metrics/direction also includes entropy as a possible uncertainty summary, but
entropy is not part of the frozen default model contract at this stage.

## Known Failure Modes

Known risks and failure modes for this frozen alpha model include:

- narrow lab-specific development domain
- sensitivity to stain quality and background artifacts
- sensitivity to channel-order mistakes
- dependence on segmentation quality and measurement-mask definition
- residual-class fragility because `iix` is inferred rather than directly stained
- potential overcalling or undercalling when weak signal, patchy background, or edge glow is present
- probability/confidence outputs that are useful for ranking review burden but not yet formally
  calibrated

The model should therefore be treated as a review-assisted classifier, not a final truth source.

## Validation / Retraining Guidance for New Protocols

For new staining orders, new marker panels, or new acquisition protocols:

1. confirm channel mapping explicitly;
2. verify segmentation behavior visually;
3. compare feature behavior and review burden on representative images;
4. compare candidate feature/model behavior against the frozen alpha baseline;
5. only then consider retraining or promoting a candidate model.

Recommended checks before using this model on a new protocol:

- small-image audit in Napari
- per-image class distributions
- review-flag rates
- optional diagnostics export
- feature-set comparison using `uv run python -m validation.compare_feature_sets`

## Relationship to MyoSight

MyoSight is treated in this project as a historical comparator and workflow reference, not as
ground truth by default.

That means:

- image-level and group-level agreement with prior MyoSight-derived analyses can be useful;
- disagreement with MyoSight does not automatically imply FiberTypeQC is wrong;
- agreement with MyoSight does not automatically prove FiberTypeQC is right;
- ROI definitions, measurement masks, and residual-class assumptions must be considered explicitly.

## Default-Model Change Policy

This frozen alpha model remains the default until a candidate model is shown to be better under a
versioned, reproducible comparison process.

A candidate model should not become default unless it is compared against the frozen alpha baseline
for:

- feature contract differences
- held-out behavior
- review burden
- error modes
- documented tradeoffs

The baseline should remain the comparator, not a moving target, until a deliberate versioned change
is accepted.
