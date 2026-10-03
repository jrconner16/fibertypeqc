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

- Trained on manually reviewed fibers from a private development cohort of four-marker sections.
  Identifiers, labels, and results are private.
- A group of mice was held out from training. So far the held-out mice have only been examined
  with a targeted (non-random) blinded audit. **A blinded review of a random hold-out sample has
  not been done yet**; it is the planned validation, and no accuracy figures are reported until
  then.
- This is a **locked development candidate**, not a validated general model. Intended for
  review-assisted analysis in which flagged fibers are inspected in the review workflow.

## What to expect (qualitative, pending hold-out validation)

- Type IIa calls appear robust.
- The IIb/IIx split depends on the absence of stain and is partly subjective, for the model and
  for a reviewer. Treat that split with more caution than the others.
- Too few Type I fibers have been checked to characterize those calls.

## Reproducibility verification

| Check | Result |
|---|---|
| Artifact and training manifest digests match the lock record | Pass |
| Refit from the recorded training rows and recipe reproduces the locked model's calls and its recorded development evaluation | Pass (trees are not bit-identical across CPU platforms) |
| Release pipeline features match the historical training code on the same regenerated sections and fixture masks | Pass |
| Release-pipeline features on new segmentations of regenerated sections are consistent with training-feature distributions (approximate input-consistency check) | Pass |

The original training segmentation masks are no longer available, so per-fiber reproduction of the
training features is not possible. The checks above verify the model-from-data link and that the
release pipeline reproduces the training-time feature computation. They are software checks, not
evidence of biological accuracy.

## Limitations

- One cohort and one imaging setup; transfer to other muscles, ages, stains, or microscopes is
  untested.
- IIx is inferred from absent signal, so weak or failed staining of any marker inflates IIx. Use the
  `postrun.residual_rate` QC check and review IIx-heavy images.
- Do not use on panels without all four required markers; the pipeline refuses them.
