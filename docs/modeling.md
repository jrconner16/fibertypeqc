# Modeling Notes

This document describes the current frozen alpha feature contract, the main feature gaps, and the
experimental feature directions that are under evaluation for later panel-aware releases.

## Scope

FiberTypeQC `v0.1.x` and the current `v0.2` foundation work keep the public workflow conservative:

- the stable fiber table remains the main public output;
- the default classifier remains the frozen alpha model;
- new feature ideas are treated as internal/diagnostic unless they prove useful.

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

### Gap 1: No direct support for extra markers

The frozen model uses only the legacy IIb/IIa pair. Even if an image includes:

- direct type I stain, or
- direct type IIx stain

those features are not used by the frozen alpha classifier.

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

When direct extra markers are present, future internal/diagnostic feature tables may include:

- `marker_i_*`
- `marker_iix_*`

for mean, percentiles, coverage, and background-relative summaries.

These are currently intended for diagnostics and later model experiments, not for stable default
classification behavior.

## Stable vs Experimental Output Policy

Current policy:

- stable `*_fibers.csv` remains conservative;
- experimental features should stay internal or diagnostic by default;
- classifier behavior should not change just because additional internal features exist.

Future-facing optional diagnostic outputs are reasonable, for example:

- `*_model_features.csv`
- `*_feature_diagnostics.csv`

but they should remain optional and not part of the stable public schema until the added features
prove useful.

## Comparison Workflow

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
