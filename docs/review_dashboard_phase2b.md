# Cohort Dashboard (Phase 2B)

Phase 2B is the read-only cohort/mouse dashboard for precomputed Phase 2A QC.
It does not rerun prediction, recompute QC in click handlers, or edit review
state.

## Prerequisite

Generate Phase 2A outputs first:

```bash
uv run python -m src.generate_review_qc \
  --project project.yaml \
  --selection-strategy all_passing
```

Then open the dashboard:

```bash
uv run python -m src.review_project_napari \
  --project project.yaml
```

Optional arguments:

- `--qc-dir path/to/qc`
- `--selection-strategy all_passing|best_passing|manual`
- `--manual-selection manual_selection.yaml`

The authoritative frozen eMHC/DAPI baseline is
`e3_exports/ta_emhc_baseline_v1_2026-07-29`.
`phase5_vivienne_dapi` is not an accepted baseline and must not be used for
dashboard validation or QC calibration.

No user-specific SSD path is stored in source, configuration, or public
documentation.

## Dashboard contents

The dock always displays:

- project, cohort scope, current domain filter, strategy, and model version;
- mouse and section counts;
- complete, targeted-review, and no-acceptable-section mouse counts;
- review progress from an existing `review_state.json`;
- object-decision, region, and reviewed-mask burden;
- PASS / REVIEW / FAIL / not-applicable counts by domain;
- a mouse → section → domain QC tree;
- technical score, review priority, provisional selection, and reason codes;
- a details pane with rule evidence and available metrics.

The domain and status filters affect only the visible tree. They do not alter QC
or selection.

## Readiness definitions

An acceptable section is applicable and has no hard technical failure.

Mouse/domain readiness:

- `complete`: at least one PASS section and the selected section set contains no
  REVIEW section;
- `targeted_review`: selection requires review, no PASS section exists, or the
  provisional selection includes a REVIEW section;
- `no_acceptable_section`: every applicable section hard-failed;
- `not_applicable`: the domain is not expected for that mouse.

A mouse inherits `no_acceptable_section` if any applicable domain has no
acceptable section. Otherwise it inherits `targeted_review` if any applicable
domain requires it.

## Strategy switching

Changing strategy in the dashboard recomputes recommendations in memory:

- `all_passing` includes all sections without hard failures;
- `best_passing` prefers technical score, then review priority, then manifest
  order;
- `manual` displays the supplied manual selection or marks selection required.

Phase 2B does not persist strategy changes. Persisted section decisions and
opening an image for review belong to Phase 3.

## Input validation

The dashboard rejects incompatible schema versions, project/model mismatches,
unknown image IDs, duplicate or missing image/domain rows, and malformed boolean
fields. This prevents silently displaying stale or unrelated QC as current.

`image_qc.csv` is required. Object QC and stored section-selection tables are
optional because the cohort summary can be reconstructed from image QC.

## Current limitations

- No raw image or label layers are opened.
- No status, exclusion, or selection action is saved.
- No image, region, or object review controls are exposed.
- No signal-aware Laminin/DAPI/eMHC metrics are added by the dashboard.
- Multi-user review and live filesystem refresh are not supported.
