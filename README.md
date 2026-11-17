# FiberTypeQC

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23893717.svg)](https://doi.org/10.5281/zenodo.23893717)

**Muscle fiber segmentation, fiber typing, quality control, and guided human review for
immunofluorescence histology, with auditable, analysis-ready outputs.**

Fiber-type composition of skeletal muscle is usually measured by hand or with per-image threshold
tools. Manual typing is slow and varies between people; automated calls are rarely checked
systematically, and corrections are often made directly in result tables, so it is hard to tell
afterwards which values came from software and which from a person. FiberTypeQC is a Python
workflow for multi-channel muscle sections (Zeiss CZI or TIFF) that keeps every step inspectable:
it segments fibers, types them with a model matched to the staining panel, flags results that need
a human look, records the reviewer's decisions separately from the model's predictions, and builds
image-, mouse-, and cohort-level tables and a report. If no released model fits your stains, it
helps you build one from a sample of fibers you label; if a model does poorly on your images, your
review corrections can train a better one. Either way the new model is checked on mice set aside
for testing, and switching to it is your decision and can be undone.

```text
images ─► segment (Cellpose) ─► type (panel-specific model) ─► QC flags
       ─► guided review (Napari) ─► finalize (predictions untouched) ─► tables + HTML report

no model for your stains:   label fibers blind ─► build ─► check on test mice ─► apply (undoable)
model does poorly:          review corrections ─► improve ─► compare by a fixed rule ─► switch
```

![Review workspace on a demo field: each fiber outlined in the color of its current call over the raw stain channels](docs/images/demo_review_workspace.png)

## Status

Version 0.5.1. Read before relying on results:

- **Models are panel-specific.** A model works only on the staining panel it was trained on, and
  the pipeline refuses a mismatched panel. Two are released:

  | Model | Stains | Classes |
  |---|---|---|
  | `quad_four_class_rf_v1` (default) | laminin, Type I, IIa, IIb | I, IIa, IIb, IIx (inferred from absent signal) |
  | `ta_three_class_logistic_v1` | laminin, IIa, IIb | IIa, IIb, IIx (inferred) |

  For any other combination of stains, [build a model](#no-model-for-your-stains) from your own
  labels.
- **No accuracy claims yet.** The released models have not been validated on a blinded random
  hold-out sample; see their [model cards](docs/model_cards/). Use them for review-assisted
  analysis.
- **Model files are release downloads**, verified by SHA-256; they are not in the repository.
- Nuclear segmentation and extra-marker (eMHC, direct IIx) measurements are experimental.

## Install and try it

macOS or Linux, with [`uv`](https://docs.astral.sh/uv/) (it installs Python 3.11 and the locked
dependencies). A GPU is optional. Windows is untested; see
[install.md](docs/install.md#windows-untested).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # installs uv (or: brew install uv); then open a new terminal
git clone https://github.com/jrconner16/fibertypeqc.git
cd fibertypeqc
uv sync --frozen                             # first run downloads several GB (PyTorch)
uv run python -m scripts.run_reference       # synthetic end-to-end check; no data needed
```

Then follow the [demo](docs/demo.md): four small public fields from download to the HTML report.
Details, HPC notes, and model downloads: [docs/install.md](docs/install.md).

## Run your own images

```bash
uv run python -m fibertypeqc init     myproject/ --images /path/to/images   # panel, samples, model
uv run python -m fibertypeqc run      myproject/   # segment, type, and QC every image
uv run python -m fibertypeqc prepare  myproject/   # review project and project QC
uv run python -m fibertypeqc review   myproject/   # guided review in Napari
uv run python -m fibertypeqc finalize myproject/   # finalized tables and cohort report
uv run python -m fibertypeqc status   myproject/   # where am I, and what next
uv run python -m fibertypeqc app      myproject/   # the same steps in a window (experimental)
```

<img src="docs/images/launcher_steps.png" width="560" alt="The project launcher: finished steps ticked on a timeline, and the current step, Review, with its buttons">

### No model for your stains

Start the project without a model; fibers are segmented and measured but not typed. Label a sample
of them blind (the model's answer is never shown), and build one:

```bash
uv run python -m fibertypeqc init     myproject/ --images /path/to/images --model none
uv run python -m fibertypeqc run      myproject/   # segment and measure
uv run python -m fibertypeqc prepare  myproject/   # with 4+ mice, sets some aside for testing
uv run python -m fibertypeqc review   myproject/ --blind --reviewer YOUR_NAME   # label fibers
uv run python -m fibertypeqc build    myproject/   # candidates, compared mouse by mouse
uv run python -m fibertypeqc evaluate myproject/ logistic   # check the recommended one on test mice
uv run python -m fibertypeqc promote  myproject/ logistic   # apply it (promote --rollback undoes)
```

<img src="docs/images/builder_build_results.png" width="640" alt="Build results in the launcher on the builder demo: labels per class and training mouse against the minimums, then each candidate model's leave-one-mouse-out scores against a majority-class baseline, and the recommended model">

`build` recommends a model only when it beats a simple baseline by a margin fixed in advance and
recognises every class; the test mice never take part in training or choosing. The
[builder demo](docs/demo.md#build-a-model-no-model-to-start-from) runs these steps on public data
with labels included. Details: [Panels without a model](docs/user_guide.md#panels-without-a-model).

### Improve a model from your corrections

When the model does poorly on your images, the fibers you corrected in review can train a
replacement:

```bash
uv run python -m fibertypeqc improve myproject/            # candidates vs the current model
uv run python -m fibertypeqc promote myproject/ CANDIDATE  # switch (promote --rollback undoes)
```

Nothing changes until you promote. Details:
[Adapt the model from your review](docs/user_guide.md#adapt-the-model-from-your-review).

The [user guide](docs/user_guide.md) covers project setup, outputs, QC checks, models,
troubleshooting, and single-image commands.

## What it does

- **Ingest:** CZI and TIFF, multi-scene CZIs split deterministically, channel roles from a panel
  file or preset.
- **Segment:** Cellpose on the laminin channel; per-fiber area and Feret diameters.
- **Type:** panel-specific scikit-learn models with pinned feature extraction; each model declares
  its panel, features, and classes, and is identified by SHA-256.
- **QC:** per-image checks (fiber count, uncertainty, inferred-class share, fiber size, channel
  crosstalk) and per-fiber flags (low confidence or margin, faint or thick laminin outline).
- **Review:** a Napari window that works one stage at a time: a verdict per section (including
  drawing areas to leave out), then the flagged fibers one by one with the reason each is shown,
  or any fiber you click; a cohort dashboard and named analysis ROIs are under Advanced.
- **Finalize:** applies review decisions and exclusions into new tables; every final value is
  labelled `predicted`, `reviewed`, `excluded`, or `unresolved`; refuses decisions made on inputs
  that have since changed.
- **Report:** image, mouse, ROI, and cohort tables, and a self-contained HTML report comparing
  predicted and finalized results.
- **Build a first model:** for a panel with no model, measure fibers without typing them, label
  a sample blind, and build candidate models under a declared recipe (leave-one-mouse-out against
  a baseline, minimum labels per class and mouse). Check the chosen one on independent test mice
  before an explicit, reversible switch.
- **Improve a model:** candidates trained from your review corrections, compared with the current
  model under a rule fixed in advance; an explicit, reversible switch.
- **Check the checking:** blind reference labels (model output hidden) with agreement between
  reviewers, and reproducibility snapshots to compare a run with a reference.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/report_composition_dark.png">
  <img src="docs/images/report_composition_light.png" width="640" alt="The cohort report on the demo: provenance (model, code revision, SHA-256 of the final table), then each mouse's fiber-type composition as predicted by the model and as finalized after review">
</picture>

*The cohort report on the demo: each mouse's composition as the model predicted it and as
finalized after review, with the model, code revision, and table digest it came from.*

## How QC and review fit together

The model's per-fiber table is written once and never edited. Automatic QC decides which fibers
and sections a person should look at; the reviewer's decisions are saved as a separate review
state; finalization combines the two into new tables, recording where each value came from. See the
[review guide](docs/review_guide.md).

## Documentation

| For | Read |
|---|---|
| Users | [Install](docs/install.md) · [Demo](docs/demo.md) · [User guide](docs/user_guide.md) · [Review guide](docs/review_guide.md) · [Panels](docs/panel_schema.md) · [Outputs](docs/output_schema.md) · [Model cards](docs/model_cards/) |
| Developers | [Architecture](ARCHITECTURE.md) · [Contributing](CONTRIBUTING.md) · [Model registry](docs/model_registry.md) · [All docs](docs/README.md) |
| Project | [Changelog](CHANGELOG.md) · [Roadmap](ROADMAP.md) |

### A five-minute tour of the code

1. [ARCHITECTURE.md](ARCHITECTURE.md): the supported surface, data contracts, and the boundary
   between product code and research tooling.
2. [src/cli.py](src/cli.py): the project-folder workflow behind `python -m fibertypeqc`.
3. [src/review/finalization.py](src/review/finalization.py): how review decisions and exclusions
   become final values without touching predictions.
4. [fibertypeqc/model_manifest.py](fibertypeqc/model_manifest.py) and
   [fibertypeqc/model_resolution.py](fibertypeqc/model_resolution.py): panel-compatibility checks
   and digest-verified model loading.
5. [src/model_improver.py](src/model_improver.py) with
   [manifests/improver/recipe.v1.yaml](manifests/improver/recipe.v1.yaml): candidate models
   evaluated on held-out mice under a pre-declared promotion rule.
6. [.github/workflows/ci.yml](.github/workflows/ci.yml) and
   [scripts/run_reference.py](scripts/run_reference.py): what every change must pass.

## Tests

```bash
uv sync --frozen --extra dev
uv run pytest -m "not integration" -q          # fast synthetic tests
uv run python -m scripts.run_reference         # end-to-end synthetic workflow
uv run python -m scripts.check_repository      # links, tracked artifacts, private-data guardrails
```

CI runs these, plus linting and a smoke test of every public command, on each push.

## Repository layout

```
fibertypeqc/  Contracts: panels, model manifests and registry, QC, provenance, reports
src/          Pipeline, review, finalization, and results implementation
scripts/      Public commands (python -m scripts.<name>)
manifests/    Model registry, model manifests, panel presets, improver recipe
examples/     Synthetic reference images and models, example outputs
docs/         User and developer documentation; docs/history/ holds superseded plans
tests/        Synthetic tests
research/     Study and model-development tooling; not part of the supported surface
validation/   Command wrappers over research validation tools
```

## Citation, license, and contact

If you use FiberTypeQC, please cite the version you used. Every release is archived on Zenodo:
[10.5281/zenodo.23893717](https://doi.org/10.5281/zenodo.23893717) always resolves to the latest
version, and each version's Zenodo page gives its own DOI (v0.5.0:
[10.5281/zenodo.24321794](https://doi.org/10.5281/zenodo.24321794)). Cite the version DOI for a
specific analysis. GitHub's "Cite this repository" button gives the citation from
[CITATION.cff](CITATION.cff). MIT License; see
[LICENSE](LICENSE). Questions and bug reports: open an
[issue](https://github.com/jrconner16/fibertypeqc/issues). Contributions: see
[CONTRIBUTING.md](CONTRIBUTING.md).

This repository is public: do not commit microscopy data, real specimen identifiers, results from
unpublished data, or models trained on them; see [AGENTS.md](AGENTS.md).
