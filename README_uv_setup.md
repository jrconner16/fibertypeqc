# UV setup for FiberTypeQC

FiberTypeQC uses `uv` to pin Python and synchronize its application and development dependencies.

## Project files

- `.python-version` keeps Python pinned to 3.11
- `pyproject.toml` defines the uv project and optional extras
- `docs/quickstart.md` documents the current command-line workflow
- `docs/panel_schema.md` documents semantic panel configuration

## Recommended starting path

Start CPU-first. Do not begin with GPU unless you already know you need it.

## Prerequisites

- `uv` installed
- Python 3.11 available locally or installable through uv

## Create and sync the environment

From the repo root:

```bash
uv python install 3.11
uv sync
```

This installs the base stack:
- napari
- microscopy image I/O
- pandas / scikit-learn / plotting
- ROI parsing for older ImageJ outputs

## Add segmentation tools

For the pilot, install the segmentation extra and CPU torch first:

```bash
uv sync --extra segmentation --extra torch-cpu
```

If that works, you can test the basic workflow without touching GPU.

## Launch tools

Open napari:

```bash
uv run napari
```

Open JupyterLab:

```bash
uv run jupyter lab
```

## Validate the environment

Run the deterministic public reference workflow, then the fast test suite:

```bash
uv run python -m scripts.run_reference
uv run python -m pytest -m "not integration"
```

Before processing private microscopy data, inspect `uv run python -m scripts.run_pipeline --help`,
confirm channel order on one representative image, and use an explicit panel config or channel arguments. See
[docs/quickstart.md](docs/quickstart.md) for the supported pipeline/review/merge flow.

## Stop conditions

Pause and reassess if any of the following happen:

- napari will not launch cleanly
- image loading is inconsistent across files
- segmentation fails badly on even clean images
- the project starts drifting toward building a full app before the feature table exists

## GPU notes

Only try GPU after the CPU path works.

This `pyproject.toml` includes a `torch-gpu` extra placeholder, but GPU installs are system-specific and often require platform-specific torch index settings. For that reason, keep the pilot CPU-first until you know the rest of the workflow is worth it.

## A simple working definition of success

A successful pilot does all of the following on a few representative images:

- loads raw IF images reliably
- produces a usable fiber mask from an existing segmentation tool
- extracts a clean per-fiber table
- reduces manual fiber typing effort with a simple classifier and uncertainty flags
- exports a CSV the existing RMarkdown workflow can use
