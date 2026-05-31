# UV setup for the muscle IF pilot

This project is intentionally set up as a small pilot, not a polished app. The goal is to get from raw IF image to segmentation candidate, per-fiber feature table, simple fiber-type model, and clean CSV export with as little setup pain as possible.

## Project files

- `.python-version` keeps Python pinned to 3.11
- `pyproject.toml` defines the uv project and optional extras
- `roadmap.md` is the build plan and decision guide

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

## Suggested first validation steps

1. Confirm napari launches.
2. Load one representative raw image.
3. Confirm channel order is readable and sensible.
4. Try one segmentation backend on one image only.
5. Save intermediate outputs before doing anything more ambitious.

## First tasks for Codex

Give Codex very small, concrete tasks in this order.

### Task 1: repo scaffold
Create the following folders if they do not exist:

```text
src/
notebooks/
data/raw/
data/interim/
data/processed/
outputs/figures/
outputs/tables/
```

### Task 2: image loader
Write a small Python module that:
- loads OME-TIFF, TIFF, or CZI if possible
- prints image shape, dtype, and channel count
- saves a quick preview PNG per channel for one test image

### Task 3: ROI / mask reader
Write a small module that can read either:
- ImageJ ROI zip files using `roifile`, or
- label masks exported from napari

Then convert both into a common per-fiber mask representation.

### Task 4: feature extraction
Write a module that computes per-fiber:
- area
- centroid
- mean and median intensity per channel
- eroded-interior intensity per channel
- optional local background ring intensity

Save results as a tidy CSV.

### Task 5: typing baseline
Write a notebook that:
- loads the feature CSV
- trains a simple classifier
- compares logistic regression, random forest, and gradient boosting
- outputs class probabilities and an uncertainty flag

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
