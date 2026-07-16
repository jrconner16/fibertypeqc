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
| `validation/` and most validation-oriented `src/` modules | Candidate models, manual audits, MyoSight comparisons, calibration, and plots | Experimental; not public default behavior |
| `src/ui_napari.py` | Earlier prototype UI | Experimental and intentionally excluded from lint/release surface |
| `src/*jag1*`, `src/merge_batch_fiber_tables.py` | Cohort-specific Jag1 regeneration summaries and reporting | Analysis tooling; requires private outputs |

Experimental code is documented in place to avoid disruptive moves. A future `experiments/` namespace
may be introduced only through incremental moves with wrapper/import compatibility tests.

## Data and artifact boundaries

- Raw microscopy images, private labels, notebooks, `test_inputs/`, and local research folders are
  ignored and must not be committed.
- `outputs/` and `data/runs/` are generated run products and must not be committed.
- `manifests/` stores small, versioned input/split contracts. Tracked manifests must use
  `input_relpath`, never machine-specific absolute paths. Run them with `--input-root`.
- `data/models/` contains only released/frozen model artifacts and their documentation. Candidate
  models belong in ignored output/artifact locations until intentionally released with a model card.
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

- `README.md`, `ARCHITECTURE.md`, `ROADMAP_2026H2.md`, and `roadmap_2026H3.md`;
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
```

For baseline-sensitive changes, also provide a versioned frozen-baseline comparison, document the
artifact path and configuration, and update the model/release documentation.
