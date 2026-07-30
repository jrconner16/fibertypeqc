# Headless Review QC (Phase 2A)

Phase 2A generates versioned technical-QC evidence and domain-specific section
selections without importing Napari or Qt. It does not implement the dashboard
and does not change prediction outputs.

The detailed formulas, denominators, missing-input behavior, and design decisions
are controlled by
[`docs/plans/qc_manual_review_system.md`](plans/qc_manual_review_system.md).

## Project artifact contract

Each image declares its expected domains and prediction artifacts:

```yaml
schema_version: review_project.v1
project_id: example
project_name: Example review project
panel_manifest: panel.yaml
model_version: model.v1
images:
  - image_id: mouse_1_section_1
    mouse_id: mouse_1
    section_id: section_1
    raw_image_path: inputs/mouse_1_section_1.czi
    prediction_directory: predictions/mouse_1_section_1
    applicable_domains:
      - fiber_segmentation
      - fiber_typing
      - nuclei
    outputs:
      fiber_labels: mouse_1_section_1_cellpose_labels.tif
      fiber_table: mouse_1_section_1_fibers.csv
      nuclei_labels: nuclear/mouse_1_section_1_nuclei_labels.tif
      nuclei_table: nuclear/mouse_1_section_1_nuclei.csv
```

Output paths are relative to `prediction_directory`. If
`applicable_domains` is omitted, fiber segmentation is applicable by default;
fiber typing is added when `fiber_table` is declared, and nuclei is added when a
nuclear label or table artifact is declared.

This distinction matters:

- an expected but missing artifact produces an explicit technical failure;
- an artifact omitted because its domain is not applicable produces
  `not_applicable`, not a failure.

## Generate QC

```bash
uv run python -m src.generate_review_qc \
  --project project.yaml \
  --selection-strategy all_passing
```

Optional arguments:

- `--rules configs/review_qc_rules.v1.yaml`
- `--output-dir path/to/qc`
- `--selection-strategy all_passing|best_passing|manual`
- `--manual-selection manual_selection.yaml`

Manual-selection YAML uses explicit manifest IDs:

```yaml
mouse_1:
  fiber_segmentation:
    - mouse_1_section_1
  fiber_typing:
    - mouse_1_section_2
  nuclei:
    - mouse_1_section_1
```

A manually selected section must be applicable and free of technical hard
failures. Empty manual selections are retained as
`manual_selection_required`.

## Outputs

The command writes:

- `qc/image_qc.csv`: one image/domain row with metrics, status, scores, explicit
  reasons, and provenance;
- `qc/fiber_qc.csv`: one row per positive fiber-mask object;
- `qc/nucleus_qc.csv`: one row per positive nucleus-mask object;
- `qc/section_selection.csv`: one mouse/domain result for the selected strategy.

`all_passing` means all applicable sections without a hard technical failure.
This intentionally includes sections with REVIEW findings. `best_passing`
prefers the highest `technical_quality_score`, then the lowest
`review_priority`, then manifest order. If no section is eligible, the result
contains no selected ID and `requires_manual_review=true`.

## Score interpretation

`technical_quality_score` is a transparent disposition:

- PASS: `1.0`
- REVIEW: `0.5`
- hard FAIL: `0.0`
- not applicable: null

It is not a calibrated probability and contains no biological outcome.

`review_priority` only orders work:

```text
100 * hard-fail reason count + 10 * review reason count
```

Informational reasons do not change either status or priority.

## Default rules

The default configuration is
[`configs/review_qc_rules.v1.yaml`](../configs/review_qc_rules.v1.yaml).

Enabled hard failures are restricted to structurally unusable output:

- required artifact missing or unreadable;
- invalid label-mask dimensionality/type/value;
- no positive fiber objects;
- empty applicable typing table;
- missing typing prediction column;
- fiber/nucleus label-shape mismatch.

Enabled REVIEW rules cover object-ID inconsistencies, invalid/duplicate IDs, and
an empty nucleus mask. Missing probability or optional association fields are
informational.

No default threshold is enabled for:

- segmented or border-touching fractions;
- fiber or nucleus area;
- confidence, margin, or entropy;
- unknown or `needs_review` fractions;
- fiber-type composition;
- nucleus density or nuclei-per-fiber;
- unassigned or ambiguous association fractions;
- central nuclei or other biological outcomes.

These values require panel/model/cohort calibration. To activate one, create a
new versioned rule file, supply an explicit threshold and rationale, set
`enabled: true`, and retain the new `rules_version` in exported QC.

## Current metric boundary

Phase 2A reads label TIFFs and object CSVs only. It does not currently compute:

- tissue-normalized coverage;
- focus or channel dropout;
- saturation;
- membrane support;
- DAPI support;
- tile/region heatmaps;
- cohort percentile or MAD rankings.

Those require stable raw-signal/tissue-mask input contracts and calibration.
They must not be approximated from biological endpoint values.
