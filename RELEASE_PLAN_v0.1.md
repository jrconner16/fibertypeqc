# FiberTypeQC v0.1-alpha Release Plan

Goal: prepare a clean public alpha that demonstrates the end-to-end workflow without exposing private lab data.

## Task 1: Confirm Release Scope - Done

- Public workflow: single-image pipeline, batch runner, Napari fiber-type review, review merge.
- Validation scripts may be included, but private JAG1-specific analysis should stay separate.
- `ui_napari.py` remains experimental unless intentionally promoted.
- v0.1-alpha should include one default classifier only:
  `rebaseline_tile_v2_p75p90_iib_iia_iix.joblib`.
- Exclude raw/private data, private case-study analysis, exploratory integrated UI, and extra
  experimental models from the public alpha.

## Task 2: Fix README CLI Drift - Done

- Verify documented commands against current script arguments.
- Fix `review_labels_napari` examples to use `--image`, not `--input-image`.
- Fix `merge_reviewed_labels` examples to use `--review`, not `--reviewed`.
- Do not change source code unless required to make documented commands accurate.
- Completed in `README.md`; no source code changes required.

## Task 3: Remove Private Data From Release Surface - Done

- Exclude raw `.czi`, `.tif`, `.tiff`, `.zip`, MyoSight `Results/`, `ROISet.zip`, and generated `outputs/`.
- Exclude or sanitize notebooks with absolute paths and embedded private image output.
- Review `data/models/` and decide which model artifacts are public.
- Completed by tightening `.gitignore`, ignoring private notebooks/labels/outputs/raw data, keeping
  only the v0.1-alpha default model visible, and adding placeholder example docs without data.

## Task 4: Package Structure Cleanup - Done

- Move reusable code toward `fibertypeqc/`.
- Keep CLI wrappers in `scripts/` or expose module commands clearly.
- Move MyoSight comparison tools to `validation/`.
- Move private-data-specific biological analysis to `analysis/jag1_case_study/` or keep out of release.
- Completed as a compatibility-preserving alpha layout: `fibertypeqc/` public namespace,
  `scripts/` command wrappers, `validation/` wrappers, and `analysis/jag1_case_study/`
  placeholder. Internal implementation remains in `src/` for now to reduce release risk.

## Task 5: Model Card And Output Docs - Done

- Document the default model, training provenance at a high level, intended use, and limitations.
- Document output CSV columns and review correction schema.
- Clearly state that alpha results require visual QC.
- Completed with model docs, quickstart, output schema, review workflow, validation summary,
  release checklist, and corrected channel wording in the README.

## Task 6: Minimal Tests And Smoke Fixtures - Done

- Add tests for I/O, label masks, quantification smoke behavior, and metrics.
- Use synthetic or cleared sample data only.
- Add a command-level smoke test if feasible.
- Completed with synthetic pytest coverage for TIFF I/O, label erosion, probability metrics, and
  quantification smoke behavior. No private image data or Cellpose/Napari runtime tests included.

## Task 7: Release Metadata - Done

- Rename project/package metadata from pilot naming to FiberTypeQC.
- Add license.
- Add release checklist.
- Confirm `.gitignore` protects private data before first public commit.
- Completed with FiberTypeQC project metadata, MIT license, release checklist updates,
  ignored experimental/training scripts, and final lint/test/CLI visibility checks.
