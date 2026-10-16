# Installation

FiberTypeQC runs from a clone of this repository in an environment locked by `uv`. It is not
installed with `pip`.

## Requirements

- macOS or Linux. A GPU is optional; it speeds up Cellpose segmentation.
- [`uv`](https://docs.astral.sh/uv/). It installs the pinned Python (3.11) and every dependency
  from `uv.lock`.
- Several GB of disk for the environment (PyTorch, Cellpose, Napari).

## Install

```bash
git clone https://github.com/jrconner16/fibertypeqc.git
cd fibertypeqc
uv sync --frozen                             # first run downloads several GB (PyTorch)
uv run python -m scripts.run_reference       # synthetic end-to-end check
```

`run_reference` needs no model download or private data. It runs the pipeline, review
finalization, and results on synthetic images and checks the outputs; it ends with
`reference workflow passed`. It tests the software, not biological accuracy.

Run every command from the repository root (the folder containing `pyproject.toml`) so `uv run`
uses the project environment.

## Development tools

```bash
uv sync --frozen --extra dev                 # adds ruff, pytest and pre-commit
uv run pytest -m "not integration" -q
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full check list.

JupyterLab is optional: add `--extra notebook` to `uv sync` if you analyze outputs in notebooks.
Extras combine, for example `uv sync --frozen --extra dev --extra notebook`.

## Shared or HPC filesystems

If `uv sync` warns that it cannot hardlink files, its cache is on a different filesystem from the
project and every package is copied. Put the cache on the same filesystem as the environment:

```bash
export UV_CACHE_DIR=/path/on/the/same/filesystem/.uv-cache
```

The first sync on Linux downloads the CUDA build of PyTorch; later syncs reuse the cache. On a
cluster, run `fibertypeqc run` (segmentation) on a GPU node or as a batch job; the other steps run
anywhere. `scripts/slurm_panel_array.sh` is an example Slurm array script for research runs.

## Model files

Released models are not stored in the repository. Download them from the
[releases page](https://github.com/jrconner16/fibertypeqc/releases) and pass the file with
`--model` (or name it in the project config). The pipeline verifies each file by its SHA-256.

## Next steps

- [Demo](demo.md): the whole workflow on four small public fields.
- [User guide](user_guide.md): running your own images.
