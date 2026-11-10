# Installation

FiberTypeQC runs from a clone of this repository in an environment locked by `uv`. It is not
installed with `pip`.

## Requirements

- macOS or Linux. A GPU is optional; it speeds up Cellpose segmentation. Windows is untested
  ([below](#windows-untested)).
- [`uv`](https://docs.astral.sh/uv/). It installs the pinned Python (3.11) and every dependency
  from `uv.lock`. Install it with
  `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `brew install uv`), then open a new
  terminal so the `uv` command is found. Other methods are in the
  [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).
- `git`. On a Mac without it, the first `git` command offers to install Apple's Command Line
  Tools; accept, then run the command again.
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

## Windows (untested)

FiberTypeQC is developed and tested on macOS and Linux. It has no Unix-only code, so it may run
natively on Windows, but this has not been checked. Please
[report](https://github.com/jrconner16/fibertypeqc/issues) what works and what does not.

In PowerShell, install `uv`, then open a new terminal:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Install [Git for Windows](https://git-scm.com/download/win), then run the same commands as above,
each on one line (the `\` line continuations in these docs are for macOS and Linux shells). Write
paths the Windows way, in quotes if they contain spaces, for example
`--images "D:\Microscopy\My images"`.

- Keep the images and the project folder on the same drive (both on `C:` or both on `D:`); the
  review project stores image paths relative to the project.
- Close output files (for example CSVs open in Excel) before a step rewrites them; Windows does
  not allow replacing a file another program has open.
- If the native install does not work, use [WSL2](https://learn.microsoft.com/windows/wsl/install)
  with Ubuntu and follow the Linux steps unchanged. The Napari review window needs WSLg
  (Windows 11, or a recent Windows 10).

## Development tools

```bash
uv sync --frozen --extra dev                 # adds ruff, pytest and pre-commit
uv run pytest -m "not integration" -q
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full check list.

JupyterLab is optional: add `--extra notebook` to `uv sync` if you analyze outputs in notebooks.
Extras combine, for example `uv sync --frozen --extra dev --extra notebook`.

## HPC clusters (Slurm and Open OnDemand)

Install once into shared storage that both the compute nodes and your desktop sessions can see.
Keep the `uv` cache and Python installs on the same filesystem as the environment; otherwise
`uv sync` warns that it cannot hardlink and copies every package:

```bash
export WORK_ROOT=/path/to/shared/work
export UV_CACHE_DIR="$WORK_ROOT/uv-cache"
export UV_PYTHON_INSTALL_DIR="$WORK_ROOT/uv-python"
cd "$WORK_ROOT" && git clone https://github.com/jrconner16/fibertypeqc.git
cd fibertypeqc && uv sync --frozen
```

The first sync on Linux downloads the CUDA build of PyTorch. Run everything else, including
`run_reference`, on a compute node: login nodes often limit threads and fail with
`RuntimeError: can't start new thread`.

**Segmentation (`run`) on a GPU node.** A batch job such as:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=fibertypeqc-run
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:1              # some sites use --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=fibertypeqc-run-%j.log
set -euo pipefail
WORK_ROOT=/path/to/shared/work
export UV_CACHE_DIR="$WORK_ROOT/uv-cache"
export UV_PYTHON_INSTALL_DIR="$WORK_ROOT/uv-python"
cd "$WORK_ROOT/fibertypeqc"
uv run --frozen python -m fibertypeqc run /path/to/project
uv run --frozen python -m fibertypeqc status /path/to/project
```

The log should show `Cellpose device: cuda`. For each image it also prints a `timing:` line
comparing wall time with pipeline time; a large `outside pipeline` share points to slow imports
from shared storage rather than slow segmentation. `run` processes images one after another; for
large cohorts, a Slurm array that runs one image per task is faster.
`scripts/slurm_panel_array.sh` is an example array script for research runs.

**`prepare` and `finalize`** need no GPU; submit them the same way on a CPU partition.

**Review in Open OnDemand.** Start a Desktop session on a CPU partition, open a terminal inside
it, set the same `UV_*` variables, and run `fibertypeqc review` from the repository folder.
Review needs no GPU, but very large sections need more memory than a typical laptop; a session
with 8 cores and 64 GB worked in testing. In a Desktop session on a GPU partition,
`fibertypeqc app` (experimental) runs every step, segmentation included, from one window without
a batch job.

## Model files

Released models are not stored in the repository. Download them from the
[releases page](https://github.com/jrconner16/fibertypeqc/releases) and pass the file with
`--model` (or name it in the project config). The pipeline verifies each file by its SHA-256.

## Next steps

- [Demo](demo.md): the whole workflow on four small public fields.
- [User guide](user_guide.md): running your own images.
