# Modeling Notes

This document describes the frozen alpha feature contract and the operational, opt-in semantic
feature path used to evaluate panel-aware candidate models.

## Scope

FiberTypeQC keeps the public workflow conservative:

- the stable fiber table remains the main public output;
- the default classifier remains the frozen alpha model;
- semantic features and candidate predictions stay in separate diagnostic/model sidecars unless
  they are explicitly promoted.

This document is about the per-fiber typing model and feature representation. It is not a statement
that the experimental features are ready for public use.

## Frozen Alpha Model

Default model:

- `data/models/rebaseline_tile_v2_p75p90_iib_iia_iix.joblib`

Task:

- per-fiber classification into `iib`, `iia`, or inferred `iix`

Panel assumptions:

- direct IIb marker channel
- direct IIa marker channel
- membrane/laminin channel for segmentation
- IIx inferred as the residual unstained class relative to IIb and IIa

## Frozen Alpha Feature Contract

The frozen alpha classifier expects this exact feature set:

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

- `type1` is the legacy internal name for the IIb marker channel.
- `type2` is the legacy internal name for the IIa marker channel.
- These names remain in the frozen alpha feature contract because the default model was trained on
  them.

## Current Gaps

The frozen alpha feature set is narrow by design. That keeps the baseline reproducible, but it
leaves several clear gaps for future modeling work.

### Gap 1: The frozen classifier does not consume extra markers

The frozen model uses only the legacy IIb/IIa pair. Even if an image includes:

- direct type I stain, or
- direct type IIx stain

those features are not used by the frozen alpha classifier. They are operational in the
`multiplanel_features.v1` diagnostics schema and may be consumed by a compatible manifest-declared
candidate model without changing stable fiber calls.

### Gap 2: Limited coverage representation

The frozen feature set includes:

- per-marker coverage
- coverage ratio
- coverage difference

It does not include other potentially useful coverage summaries such as:

- total marker coverage
- dominant-marker coverage
- minimum coverage across the active pair
- coverage balance
- coverage interaction/product terms

Why these may help:

- marker coverage is often more robust than mean intensity when signal is patchy;
- coverage can help distinguish diffuse weak background from real spatially coherent stain;
- interaction between coverage and intensity may help identify fibers with small but strong
  positive regions versus broad moderate signal.

### Gap 3: No background-relative or SNR-style features

The frozen model uses raw post-preprocessing intensity summaries, but it does not explicitly encode
how far a fiber sits above local tissue background.

Potentially useful additions:

- marker mean relative to tissue median
- marker p90 relative to tissue median
- robust SNR-style features using tissue MAD or similar scale estimates
- coverage multiplied by background-relative strength

Why these may help:

- absolute intensity can drift across sessions, slides, and staining quality;
- background-relative features can make a strong signal on a dim slide more comparable to a strong
  signal on a bright slide;
- robust SNR-style terms may help separate real positive fibers from noisy background or edge glow.

### Gap 4: No explicit diagnostics output by default

The stable `*_fibers.csv` is intentionally conservative. That is good for alpha stability, but it
means richer experimental feature analysis should live elsewhere until it proves useful.

## Experimental Feature Directions

The current internal refactor adds a path for experimental features without changing the stable
public schema by default.

### Coverage-oriented experimental features

Examples:

- `type_cov_sum`
- `type_cov_max`
- `type_cov_min`
- `type_cov_balance`
- `type_cov_product`

### Background-relative / SNR-oriented experimental features

Examples:

- `type1_snr_mean`
- `type2_snr_mean`
- `type1_snr_p90`
- `type2_snr_p90`
- `type_snr_ratio`
- `type_snr_diff`
- `type1_cov_x_snr`
- `type2_cov_x_snr`

### Extra-marker experimental features

When direct extra markers are present, the semantic diagnostic table includes columns such as:

- `type_i.mean`, `type_i.p90`, and `type_i.coverage_high`
- `type_iix.mean`, `type_iix.p90`, and `type_iix.coverage_high`
- `emhc.mean`, `emhc.p90`, and `emhc.coverage_high`

with corresponding SNR and optional center/edge summaries.

These are currently intended for diagnostics and later model experiments, not for stable default
classification behavior.

The same applies to optional diagnostics exports such as `*_feature_diagnostics.csv`: they are
useful for model-development and feature-comparison work, but they should not be treated as part of
the stable biological output contract unless a later version explicitly promotes them.

## Stable vs Experimental Output Policy

Current policy:

- stable `*_fibers.csv` remains conservative;
- experimental features should stay internal or diagnostic by default;
- classifier behavior should not change just because additional internal features exist.

The operational optional outputs are `*_feature_diagnostics.csv` and, for a compatible semantic
candidate bundle, `*_model_predictions.csv`. They remain outside the stable biological output
contract until the added features and a candidate model prove useful.

Current policy:

- stable `*_fibers.csv` stays focused on biological/review output;
- optional diagnostics exports can expose model-development features in a separate table;
- diagnostics exports are disabled by default.
- semantic candidate predictions never overwrite the stable `*_fibers.csv` calls.

## Comparison Workflow

The candidate-model evaluation protocol is documented in:

- `docs/baseline_comparison_protocol.md`

Use the feature comparison utility to compare:

- the frozen alpha model feature contract
- the code-level frozen baseline feature list
- the current experimental feature builder output

Command:

```bash
uv run python -m validation.compare_feature_sets
```

Optional CSV output:

```bash
uv run python -m validation.compare_feature_sets \
  --output outputs/feature_set_comparison.csv
```

## Recommended Next Modeling Step

Before changing default classifier behavior:

1. keep the frozen alpha feature contract explicit;
2. compare the experimental feature set against the frozen baseline;
3. test whether added coverage/SNR/extra-marker features improve:
   - class separation,
   - confidence quality,
   - review burden,
   - or held-out behavior;
4. only then consider promoting any feature additions into a candidate model path.
