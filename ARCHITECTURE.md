# FiberTypeQC Architecture

## Repository identity

FiberTypeQC is currently a runnable research application, not an installable package distribution.
`pyproject.toml` deliberately uses `[tool.uv] package = false`. The `fibertypeqc/` namespace is a
small public facade over implementation in `src/`; a packaging rewrite is not part of the current
cleanup work.

The latest published release is v0.2.0. The working development version is v0.3.0.dev0.

## Stable public workflow

```text
scripts.run_pipeline -> scripts.review_labels_napari -> scripts.merge_reviewed_labels
                         \
                          scripts.run_batch (multi-image pipeline execution)
```

Public command wrappers live in `scripts/`:

- `run_pipeline.py`: single-image preprocessing, fiber segmentation, quantification, classification,
  QC, and output creation.
- `run_batch.py`: applies the frozen workflow across a directory or explicit manifest.
- `review_labels_napari.py`: records manual fiber-label corrections.
- `merge_reviewed_labels.py`: combines model outputs with manual corrections.
- `debug_fiber.py` and `backfill_feret_from_labels.py`: diagnostic/maintenance utilities, not primary
  workflow steps.

The stable biological contract is `pipeline -> review -> merge`. The frozen default model expects
IIb and IIa markers plus a membrane channel and treats IIx as residual inference. Its defaults,
thresholds, feature contract, QC flags, and merge behavior are baseline-sensitive.

## Code areas

| Area | Responsibility | Status |
|---|---|---|
| `fibertypeqc/` | Public import facade, panel/config helpers, and shared concepts | Supported facade |
| `src/run_pipeline.py`, `src/run_batch.py` | Pipeline orchestration and batch execution | Supported implementation |
| `src/preprocess_membrane.py`, `src/segment_cellpose.py` | Membrane preprocessing and Cellpose fiber segmentation | Supported implementation |
| `src/quantify_classify.py`, `src/label_masks.py`, `src/fiber_type_labels.py` | Feature extraction, typing, labels, and QC | Supported implementation; frozen path is baseline-sensitive |
| `src/review_labels_napari.py`, `src/merge_reviewed_labels.py` | Manual review and merge workflow | Supported implementation |
| `research/` | Candidate models, manual audits, MyoSight comparisons, calibration, cohort evaluation, and plots | Research tooling; not part of the release surface |
| `validation/` | Thin command wrappers over `research/` validation tools | Research tooling |

Research modules live in `research/` and may import the supported core. The core (`src/`,
`fibertypeqc/`, `scripts/`) must never import `research/`; `tests/test_core_boundary.py` enforces this.

## Supported release surface

The overhaul in [ROADMAP.md](ROADMAP.md) narrows the supported product to the modules below. Anything
else (`research/`, `validation/`, `analysis/`) is research tooling: it may change or move without
notice and is not part of the release contract.

| Layer | Supported modules |
|---|---|
| Public commands | `scripts/run_pipeline.py`, `scripts/run_batch.py`, `scripts/finalize_review_project.py`, `scripts/merge_reviewed_labels.py`, `scripts/review_labels_napari.py`, `scripts/run_reference.py`, `scripts/validate_reference_outputs.py`, `scripts/check_repository.py` |
| Pipeline core | `src/run_pipeline.py`, `src/run_batch.py`, `src/io_utils.py`, `src/preprocess_membrane.py`, `src/segment_cellpose.py`, `src/quantify_classify.py`, `src/label_masks.py`, `src/fiber_type_labels.py`, `src/nuclear_association.py`, `src/run_nuclear_stage.py`, `src/dapi_preprocess.py`, `fibertypeqc/czi_scenes.py`, `src/split_czi_scenes.py` |
| Contracts and provenance | `fibertypeqc/` (config, panels, feature schema, model manifest, model resolution, semantic model, evidence registry, QC contract, artifacts, result bundle, HTML report) |
| Review and finalization | `src/review/` (including `finalization.py`), `src/review_project_napari.py`, `src/generate_review_qc.py`, `src/finalize_review_project.py`; legacy `src/review_labels_napari.py` (decisions import into finalization) and deprecated `src/merge_reviewed_labels.py` |

Release target (see roadmap): the project-based review becomes the supported reviewer, a finalizer
produces analysis-ready tables, and typing uses one panel-specific model per channel configuration.

## Data and artifact boundaries

- Raw microscopy images, private labels, notebooks, `test_inputs/`, and local research folders are
  ignored and must not be committed.
- `outputs/` and `data/runs/` are generated run products and must not be committed.
- `manifests/` stores small, versioned input/split contracts. Tracked manifests must use
  `input_relpath`, never machine-specific absolute paths. Run them with `--input-root`.
- `data/models/` contains only model artifacts approved for public distribution and their
  documentation (enforced by `scripts.check_repository`). Models trained on private or unpublished
  data are never committed: the registry records their identity and digest, and the artifact is
  resolved from a private model root at runtime.
- Documentation belongs in `docs/`; user-facing workflow documentation belongs in the root README or
  linked docs.

## Output contracts

The stable per-image outputs include label masks, a fiber table, summary table, review template, and
optional feature diagnostics. See `docs/output_schema.md` for columns.

Feature diagnostics and candidate-model outputs are separate from the stable fibers CSV. A model or
threshold change must not silently replace existing classifications. Cached segmentation products may
be reused only when their recorded inputs and parameters are compatible.

## Planning handoff scope

Include these materials in a GPT/Codex planning handoff:

- `README.md`, `ARCHITECTURE.md`, and the current public roadmap when one is active;
- `pyproject.toml`, `data/models/model_card.md`, and relevant documents under `docs/`;
- `fibertypeqc/`, public `scripts/`, relevant `src/` modules, and their tests;
- only the small manifests relevant to the planned change.

Do not include raw images, ignored data/output folders, local paths, private labels, notebooks, or
generated validation products. Describe those assets by availability and role instead.

## Verification expectations

Before merging changes, run:

```bash
uv run python -m pytest -m "not integration" -q
uv run ruff check .
uv run python -m scripts.check_repository
```

For baseline-sensitive changes, also provide a versioned frozen-baseline comparison, document the
artifact path and configuration, and update the model/release documentation.
