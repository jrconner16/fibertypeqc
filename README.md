# FiberTypeQC

**Muscle fiber segmentation, fiber typing, QC, guided review, and reporting for immunofluorescence
histology images.**

FiberTypeQC takes multi-channel muscle sections (CZI or TIFF) and produces analysis-ready fiber
tables and a cohort report:

1. **Segment** fibers from the laminin channel (Cellpose).
2. **Classify** fiber types with a model matched to your staining panel.
3. **QC** every image and fiber.
4. **Review** flagged results in Napari.
5. **Finalize** review decisions into analysis tables without altering the model's predictions.
6. **Report** image, mouse, and cohort results in a self-contained HTML report.

## Status

Development version `0.3.0.dev0`, preparing a release candidate. Read this before relying on
results:

- **Models are panel-specific.** A model only works on the staining panel it was trained on, and
  the pipeline refuses a mismatched panel. The default model, `quad_four_class_rf_v1`, is for
  four-marker panels (laminin, Type I, IIa, IIb; IIx inferred from absent signal).
- **The default model is not in this repository** because it was trained on unpublished data. You
  need the model file from the maintainers; the pipeline verifies it by its SHA-256.
- **The default model is not yet validated on a random hold-out sample.** See its
  [model card](docs/model_cards/quad_four_class_rf_v1.md). Use it for review-assisted analysis.
- Without a model for your panel you can still run segmentation, QC, and the synthetic reference.

## Requirements

- macOS or Linux; a GPU is optional but speeds up Cellpose.
- [`uv`](https://docs.astral.sh/uv/) (it installs the pinned Python 3.11 and all dependencies).
- A panel file describing which channel holds which marker (see [Inputs](#inputs-you-provide)).
- The model file for your panel.

## Install and check

Run every command from the repository root (the folder containing `pyproject.toml`).

```bash
git clone https://github.com/jrconner16/fibertypeqc.git
cd fibertypeqc
uv sync --frozen                             # first run downloads several GB (PyTorch)
uv run python -m scripts.run_reference       # runs the synthetic reference end to end
```

`run_reference` needs no private model or data. It runs the pipeline, review finalization, and
results on synthetic images and checks the outputs; it should end with `reference workflow passed`.
It tests the software, not biological accuracy. On shared or HPC filesystems see
[README_uv_setup.md](README_uv_setup.md).

## Inputs you provide

**Panel file** (`my_panel.yaml`): the channel index of each marker, counting from 0. For the
four-marker panel, with IIx inferred from the absence of Type I, IIa, and IIb:

```yaml
channels:
  laminin: 2
  dapi: null
  type_i: 0
  type_iia: 1
  type_iib: 3
  type_iix: null
  emhc: null
classification:
  residual_inference:
    enabled: true
    target_class: iix
    requires_negative_markers: [i, iia, iib]
```

Details: [docs/panel_schema.md](docs/panel_schema.md).

**Sample sheet** (`samples.csv`): one row per input image. `image_id` is the file name without its
extension. Any extra column (here `genotype`) becomes a condition for cohort summaries.

```csv
image_id,mouse_id,raw_image_path,genotype
slide_A,mouse_1,images/slide_A.czi,wt
slide_B,mouse_2,images/slide_B.czi,mdx
```

**Model**: either pass the file, `--model /path/to/quad_four_class_rf_v1.joblib`, or set
`export FIBERTYPEQC_MODEL_ROOT=/folder/containing/the/model` once and omit `--model`.

## End-to-end workflow

Put everything for one study in a **project folder** with a small config file:

```
myproject/
  fibertypeqc_project.yaml
  panel.yaml
  samples.csv
```

```yaml
# myproject/fibertypeqc_project.yaml  (paths are relative to this folder)
schema_version: fibertypeqc_project.v1
images: /path/to/images            # folder of .czi / .tif files
panel: panel.yaml
sample_sheet: samples.csv
model: /path/to/quad_four_class_rf_v1.joblib   # or a registered model ID
split_czi_scenes: true             # each Zeiss scene in a CZI becomes its own section
```

Then run the steps. Each reads the previous step's files and writes new ones; predictions are
never modified.

```bash
uv run python -m fibertypeqc status   myproject/   # where am I, and what is the next command
uv run python -m fibertypeqc run      myproject/   # 1. segment, type, and QC every image (slow)
uv run python -m fibertypeqc prepare  myproject/   # 2-3. build the review project, run project QC
uv run python -m fibertypeqc review   myproject/   # 4. review in Napari (GUI); saves automatically
uv run python -m fibertypeqc finalize myproject/   # 5-6. finalized tables, summaries, HTML report
```

Then open `myproject/results/cohort_report.html`. The folder ends up with `batch/` (masks, fiber
tables), `review/` (QC and saved decisions), `final/`, and `results/`.

Notes:

- `run` is the slow step (Cellpose); use a GPU node or a batch job for many images. It checks the
  model, panel, and inputs before processing and stops with one message if something is missing.
  Extra options are passed through, e.g. `run myproject/ --reuse-artifacts auto` to reuse masks
  from an earlier run. The other steps take seconds to minutes and can be re-run at any time.
- `finalize` works before any review; every fiber then keeps its model prediction.
- `prepare` keeps an existing review project (and its decisions) rather than rebuilding it.
- With split CZIs, tissue pieces imaged within one Zeiss scene stay together in one section.
- Keep the project folder on storage with room: masks and exported scenes are about the size of
  the raw images.

<details>
<summary>The individual commands behind each step</summary>

```bash
uv run python -m scripts.run_batch --input-dir images/ --panel-config panel.yaml \
  --split-czi-scenes --model MODEL --output-dir myproject/batch
uv run python -m scripts.make_review_project --batch-dir myproject/batch \
  --sample-sheet samples.csv --panel-config panel.yaml --project-dir myproject/review
uv run python -m scripts.generate_review_qc --project myproject/review/project.yaml
uv run python -m scripts.review_project_napari --project myproject/review/project.yaml \
  --reviewer YOUR_NAME --display-downsample 2
uv run python -m scripts.finalize_review_project --project myproject/review/project.yaml \
  --output-dir myproject/final
uv run python -m scripts.summarize_results --final-dir myproject/final \
  --project myproject/review/project.yaml --output-dir myproject/results
```

`make_review_project` never overwrites an existing project.

</details>

## Review

The review workspace has four parts: cohort QC, per-section pass/fail/exclude, guided
fiber-by-fiber review of flagged fibers (`K` keeps the model call, `1`–`4` set I/IIa/IIb/IIx), and
regions (draw an area to exclude, or name analysis ROIs). Fibers are flagged for low model
confidence or margin and for a faint or unusually thick laminin outline; "Why shown" names the
reason. Full guide: [README_review_workflow.md](README_review_workflow.md).

## What you get

| Step | Output | Notes |
|---|---|---|
| `run` | per image: `*_cellpose_labels.tif`, `*_fibers.csv`, `*_summary.csv`, `*_run.json`, QC JSON, `*_result_report.html` | `*_fibers.csv` holds the model's call, probabilities, confidence, and margin; it is never modified later |
| `prepare` | `qc/image_qc.csv`, `fiber_qc.csv`, `section_selection.csv` | technical QC and which sections are used |
| `finalize` | `<image_id>_fibers_finalized.csv`, `final_fiber_table.csv`, `finalization_manifest.json` | every model column plus `final_type` and `value_source` (`predicted`, `reviewed`, `excluded`, `unresolved`) |
| `finalize` | `image_summary.csv`, `mouse_summary.csv`, `cohort_summary.csv`, `roi_summary.csv`, `cohort_report.html` | CSVs are the source of truth; the report is built only from them |

How final values are decided:

- exclusions win: an excluded section, then a drawn exclusion region (by fiber center), then a
  fiber the reviewer excluded; otherwise the reviewer's decision; otherwise the model's call;
- fibers flagged for review but not reviewed keep the model's call and are counted in the report;
- unresolved fibers have no final type and are left out of percentages, with their count reported;
- mouse-level results pool fibers across that mouse's sections; cohort results treat each mouse as
  one unit (mean ± SD);
- finalization refuses a section whose mask or fiber table changed after review began.

Column definitions: [docs/output_schema.md](docs/output_schema.md).

## Models

| Model ID | Panel | Classes | Where the file is |
|---|---|---|---|
| `quad_four_class_rf_v1` (default) | laminin, Type I, IIa, IIb | I, IIa, IIb, IIx (inferred) | not in the repository; verified by SHA-256 |
| `synthetic_four_class_reference_v1` | laminin, Type I, IIa, IIb | I, IIa, IIb, IIx | in the repository; synthetic test fixture only |
| `rebaseline_tile_v2_p75p90_iib_iia_iix` | laminin, IIa, IIb | IIa, IIb, IIx (inferred) | in the repository; retired historical model |

`--model` takes a model ID or a path to a registered model file. The registry is
`manifests/model_registry.v1.yaml` ([docs/model_registry.md](docs/model_registry.md)). A model's
manifest can pin the feature-extraction settings it was trained with; the pipeline then uses exactly
those and refuses typing options that would change them.

## Automatic QC

Per image, written to `*_postrun_qc.json` and the summary. A warning never removes fibers.

| Check | Warns when |
|---|---|
| `postrun.fiber_count` | fewer than 300 fibers |
| `postrun.uncertainty_rate` | more than 35% of fibers are low-confidence or unresolved |
| `postrun.residual_rate` | share of the inferred class (IIx); informational until `--qc-max-residual-rate` is set |
| `postrun.median_area` | median fiber area outside 200–15,000 px² |
| `postrun.marker_correlation` | marker channels correlate above 0.92 (possible crosstalk or wrong channels) |

## Troubleshooting

- **`No module named …`, or the wrong Python version**: you are not in the repository root.
- **Not sure what to run next**: `uv run python -m fibertypeqc status myproject/`.
- **"distributed privately" / "not a registered model"**: pass `--model /path/to/model.joblib` or
  set `FIBERTYPEQC_MODEL_ROOT`; the file must be the registered model (checked by SHA-256).
- **"contains N scenes"**: set `split_czi_scenes: true` in the project config (or add
  `--split-czi-scenes` to `run_batch`).
- **"non-channel dimensions (Z=…)"**: project or split Z/T stacks into single-plane multichannel
  images first. TIFFs without channel metadata need ImageJ axes such as `CYX`.
- **Batch reports fewer sections than expected**: sections are Zeiss scenes; the batch log prints
  the scene count for each CZI.
- **Project has fewer images than the batch**: `make_review_project` prints `warning: skipped …`
  with the reason for each (failed in the batch, not in the sample sheet, or missing masks).
- **Poor segmentation**: check the laminin channel in the panel file. `run_pipeline` (single image)
  accepts `--diameter` (default 30) and `--cellpose-normalize`.
- **Out of memory in review**: use `--display-downsample 4`.

## Other entry points

- **One image**: `uv run python -m scripts.run_pipeline --input image.tif --output-dir out
  --panel-config my_panel.yaml --model MODEL`. Add `--labels-path` to use an existing mask.
- **Historical three-class model** (laminin, IIa, IIb panels): `scripts.run_batch --model
  rebaseline_tile_v2_p75p90_iib_iia_iix`; see [docs/quickstart.md](docs/quickstart.md) and its
  [model card](data/models/model_card.md).
- **Legacy per-image reviewer**: `scripts.review_labels_napari`; its corrections import into
  finalization with `--legacy-review IMAGE_ID=CSV`. `scripts.merge_reviewed_labels` is deprecated.

## Repository layout

```
scripts/      Public commands (python -m scripts.<name>)
src/          Supported pipeline, review, finalization, and results code
fibertypeqc/  Contracts: panels, model manifests and registry, QC, provenance, reports
manifests/    Model registry and model manifests
examples/     Synthetic reference images, models, and example configs
docs/         Schemas, model cards, review guides
research/     Research and study tooling; not part of the supported surface
validation/   Thin wrappers over research validation tools
tests/        Synthetic tests
```

More detail: [ARCHITECTURE.md](ARCHITECTURE.md). Plan and status: [ROADMAP.md](ROADMAP.md).
Screenshots of the earlier per-image reviewer: [docs/demo_manifest.md](docs/demo_manifest.md).

## Development

```bash
uv sync --frozen --extra dev                  # adds pytest and pre-commit
uv run ruff check .
uv run pytest -q
uv run python -m scripts.run_reference
uv run python -m scripts.check_repository     # links, tracked artifacts, private-data guardrails
uv run pre-commit install                     # once per clone
```

This repository is public. Do not commit microscopy data, real specimen identifiers, results from
unpublished data, or models trained on them; see [AGENTS.md](AGENTS.md).

## License and contact

MIT License; see [LICENSE](LICENSE). For questions or bug reports, open an issue.
