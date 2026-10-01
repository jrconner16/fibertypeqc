# Model card: `quad_four_class_rf_v1`

Default fiber-type model for FiberTypeQC release runs on four-marker panels.

## Summary

| Field | Value |
|---|---|
| Task | `fiber_identity` |
| Classes | Type I, IIa, IIb, IIx |
| Required panel | laminin, Type I, Type IIa, Type IIb (each mapped to a channel by the panel YAML) |
| IIx | Residual inference: a fiber negative for I, IIa, and IIb. IIx is not an observed marker. |
| Estimator | scikit-learn `RandomForestClassifier` (400 trees, `min_samples_leaf=3`, balanced class weights) |
| Features | 19 `multiplanel_features.v1` features: fiber area plus mean, p75, p90, high-signal coverage, and two SNR summaries for each of Type I, IIa, and IIb |
| Manifest | [`manifests/models/quad_four_class_rf_v1.yaml`](../../manifests/models/quad_four_class_rf_v1.yaml) |
| Artifact | Not distributed in Git. Set `FIBERTYPEQC_MODEL_ROOT` to a directory containing `quad_four_class_rf_v1.joblib`; the pipeline verifies its SHA-256 against the manifest and registry. |

## Use

```bash
export FIBERTYPEQC_MODEL_ROOT=/path/to/private/models
uv run python -m scripts.run_batch --input-dir images/ --panel-config my_quad_panel.yaml
# or, for one image:
uv run python -m scripts.run_pipeline --input image.tif --output-dir out \
  --panel-config my_quad_panel.yaml --model quad_four_class_rf_v1
```

The panel must enable residual inference for IIx (`target_class: iix`, negative markers
`[i, iia, iib]`); see [`examples/reference_four_class/panel.yaml`](../../examples/reference_four_class/panel.yaml)
for the format. Multi-scene CZIs must be split first (`run_batch --split-czi-scenes`).

## Feature extraction is pinned

Features must be computed exactly as during training. The manifest's `feature_extraction` block
pins every feature-affecting setting (`global_subtract` background, 2 px erosion, and the remaining
quantification defaults). With this model the pipeline uses those values, does not apply the
`--sensitivity` profile, and refuses typing flags that would change features.

## Training and evidence

- Trained on manually reviewed fibers from a private development cohort of four-marker sections,
  with leave-one-mouse-out development evaluation and a separately locked final test.
  Identifiers, labels, and results are private.
- This is a **locked development candidate**, not a validated general model. Intended for
  review-assisted analysis in which flagged fibers are inspected in the review workflow.

## Reproducibility verification

| Check | Result |
|---|---|
| Artifact and training manifest digests match the lock record | Pass |
| Refit from the recorded training rows and recipe | 100% class agreement with the locked artifact on all training rows; all recorded evaluation groups reproduced (counts exact, accuracy and balanced accuracy to 4 decimals). Trees are not bit-identical across CPU platforms. |
| Release pipeline features vs the historical training code, on the same regenerated sections and fixture masks | Bit-identical for all 19 features on 225,751 regions in 15 sections |
| Release-pipeline features on new segmentations of 3 regenerated sections (3 mice) vs training-feature distributions | Max Kolmogorov–Smirnov distance 0.12–0.17 across all 19 features; controls with swapped marker channels give 0.86–0.98 and a different section 0.37–0.62. Approximate input-consistency check, not a reproduction. |

The original training segmentation masks are no longer available, so per-fiber reproduction of the
training features is not possible. The checks above verify the model-from-data link and that the
release pipeline reproduces the training-time feature computation.

## Limitations

- One cohort and one imaging setup; transfer to other muscles, ages, stains, or microscopes is
  untested.
- IIx is inferred from absent signal, so weak or failed staining of any marker inflates IIx. Use the
  `postrun.residual_rate` QC check and review IIx-heavy images.
- Do not use on panels without all four required markers; the pipeline refuses them.
