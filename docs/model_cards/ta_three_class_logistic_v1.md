# Model card: `ta_three_class_logistic_v1`

Fiber-type model for three-marker panels (laminin, Type IIa, Type IIb).

## Summary

| Field | Value |
|---|---|
| Task | `fiber_identity` |
| Classes | IIa, IIb, IIx |
| Required panel | laminin, Type IIa, Type IIb (each mapped to a channel by the panel YAML) |
| IIx | Residual inference: a fiber negative for IIa and IIb. IIx is not an observed marker. |
| Estimator | scikit-learn pipeline: `StandardScaler` then multinomial `LogisticRegression` (`C=1.0`, balanced class weights) |
| Features | 14 `multiplanel_features.v1` features: mean, p90, high-signal coverage, two SNR summaries, and center and edge means for each of Type IIa and IIb |
| Manifest | [`manifests/models/ta_three_class_logistic_v1.yaml`](../../manifests/models/ta_three_class_logistic_v1.yaml) |
| Artifact | Not in Git; download it from the [releases page](https://github.com/jrconner16/fibertypeqc/releases). Pass the file with `--model PATH`, or set `FIBERTYPEQC_MODEL_ROOT` to a directory containing `ta_three_class_logistic_v1.joblib`; the pipeline verifies its SHA-256 against the manifest and registry. |

## Use

In a project folder, choose the `three_marker_iib_iia_laminin` panel preset (or your own panel with
the same markers) and set `model: ta_three_class_logistic_v1` in `fibertypeqc_project.yaml`. For
one image:

```bash
uv run python -m scripts.run_pipeline --input image.tif --output-dir out \
  --panel-config manifests/panels/three_marker_iib_iia_laminin.yaml \
  --model /path/to/ta_three_class_logistic_v1.joblib
```

The panel must enable residual inference for IIx (`target_class: iix`, negative markers
`[iia, iib]`).

## Feature extraction is pinned

Features must be computed exactly as during training. The manifest's `feature_extraction` block
pins every feature-affecting setting (tile background subtraction with 256 px tiles, 2 px erosion,
and the remaining quantification values). With this model the pipeline uses those values, does not
apply the `--sensitivity` profile, and refuses typing flags that would change features.

## Training and evidence

- Trained on fibers from a private development cohort of tibialis anterior sections, labelled by
  reviewed threshold-based typing. Identifiers, labels, and results are private.
- The recipe was fixed before fitting and compared with other model families by
  leave-one-mouse-out evaluation on the development mice only.
- **A blinded review of a random hold-out sample has not been done**, and no accuracy figures are
  reported.
- This is a **development candidate**, not a validated general model. Intended for
  review-assisted analysis in which flagged fibers are inspected in the review workflow.

## Limitations

- One cohort, one muscle, and one imaging setup; transfer to other muscles, ages, stains, or
  microscopes is untested.
- The IIb/IIx split depends on the absence of stain and is partly subjective, for the model and
  for a reviewer. IIx is inferred from absent signal, so weak or failed staining inflates IIx. Use
  the `postrun.residual_rate` QC check and review IIx-heavy images.
- No Type I class: do not use it where Type I fibers matter.
