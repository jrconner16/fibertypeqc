# Changelog

## Unreleased: v0.3.0.dev0

### Development status

- Current development version after the published v0.2.0 release.
- The frozen default classifier and review-assisted public workflow remain unchanged.
- Candidate-model evaluation, review-policy research, and cohort-specific tools remain experimental.

## v0.2.0

### Release Title

`FiberTypeQC v0.2.0 — panel config, diagnostics export, and baseline safeguards`

### Added

- Panel-aware channel config foundation with YAML-based channel/schema support.
- Legacy alias compatibility for `type1 -> iib` and `type2 -> iia`.
- Public CLI support for explicit marker naming in pipeline, review, and batch workflows.
- Output provenance fields such as `fiber_type_source` and `available_markers`.
- Internal support for optional extra-marker stats (`i`, `iix`) without changing default typing.
- Optional diagnostics export via `--export-diagnostics`, writing `*_feature_diagnostics.csv`.
- Feature-set comparison tooling:
  - `src/compare_feature_sets.py`
  - `validation/compare_feature_sets.py`
- `docs/modeling.md` for frozen-model feature contract, feature gaps, and experimental directions.
- Expanded frozen alpha model card.
- Baseline regression guards for the frozen alpha feature contract and legacy rule path.
- Pytest split between:
  - fast synthetic default tests: `python -m pytest -m "not integration"`
  - optional integration tests: `python -m pytest -m integration`

### Changed

- Public documentation now distinguishes:
  - stable `*_fibers.csv` biological/review output
  - optional `*_feature_diagnostics.csv` model-development/debugging output
- Batch runs preserve the frozen alpha baseline by default, while logging clearly when channel/config
  overrides move a run outside the strict baseline path.

### Not Changed

- Default classifier/model file
- Default thresholds
- Default feature contract used by the frozen alpha model
- Default channel assumptions for the alpha path
- Default erosion/preprocessing behavior
- Review merge logic

### Release Notes

`v0.2.0` is a workflow and architecture hardening release. It improves configuration, provenance,
diagnostics, documentation, and regression guardrails while keeping the stable alpha model behavior
conservative.

`FiberTypeQC remains alpha-stage research software. This release does not change the frozen default
classifier or claim final validation as a MyoSight replacement.`
