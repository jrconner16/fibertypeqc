# TA 28 dpi eMHC/DAPI nuclear segmentation baseline v1

## Purpose

Provisional baseline for nuclear segmentation in the TA 28 dpi
eMHC/Laminin/DAPI cohort.

This is a workable automated starting point intended for image-level,
region-level, and object-level manual review. It is not a validated
universal nuclear segmentation model.

## Baseline settings

- Cellpose model: CPSAM
- DAPI preprocessing: tile subtraction
- Nuclear downsample factor: 1
- Nuclear diameter: 12 pixels
- Minimum nuclear mask size: 30 pixels
- Cell probability threshold: -1
- Flow threshold: 0.6

## Development images

Settings were visually reviewed on 10 sections from 2 mice:

- 351537_LL
  - 3 injected sections
  - 3 non-injected sections
- 351544_L
  - 3 injected sections
  - 1 non-injected section

## Observed performance

Approximate visual assessment:

- 3–4 sections were close to acceptable
- 2–3 sections appeared undercalled
- 2–3 sections appeared overcalled

Performance was strongly affected by image quality and DAPI exposure.

Known failure modes included:

- weak or missing DAPI regions
- saturated and noisy images
- false-positive masks in high-background regions
- missed nuclei in weak-signal images

## Parameter findings

- Flow threshold 0.4 undercalled nuclei.
- Flow threshold 0.6 gave the best observed precision/recall compromise.
- Flow threshold 0.8 produced excessive false positives.
- Automatic diameter estimation produced giant, fiber-scale masks.
- Cell probability values below -1 did not provide a convincing improvement.
- Tile normalization did not clearly improve recall and could amplify noise.

## Intended use

Use these settings as the baseline first-pass model.

Do not tune parameters independently for each image.

Images should subsequently receive:

1. image-level QC
2. optional invalid-region marking
3. object-level nuclear corrections

Reviewed masks can later be used to fine-tune a custom Cellpose nuclear model.

## Limitations

These settings were selected by visual review rather than formal object-level
precision and recall measurement. They should not be described as validated
performance estimates.
