# Contributing to FiberTypeQC

Thank you for helping. Bug reports, questions, documentation fixes, and code changes are all
welcome.

## Getting help and reporting problems

- **Questions and bug reports:** open an [issue](https://github.com/jrconner16/fibertypeqc/issues)
  using the bug or question template. Include the command you ran, the full error, your operating
  system, and the FiberTypeQC version or commit.
- **Do not attach unpublished images or data** to a public issue. If a problem only appears on
  your data, describe the image (format, size, channels, scenes) or share a small synthetic or
  cropped example you are free to publish.

## Setting up

```bash
git clone https://github.com/jrconner16/fibertypeqc.git
cd fibertypeqc
uv sync --frozen --extra dev
uv run pre-commit install      # once per clone
```

## Checks every change must pass

```bash
uv run ruff check .
uv run pytest -m "not integration" -q
uv run python -m scripts.run_reference
uv run python -m scripts.check_repository
```

CI runs the same checks on every push and pull request.

## Making a change

1. Open an issue first for anything larger than a small fix, so the approach can be agreed.
2. Work on a branch and keep the change focused on one problem.
3. Add or update tests. Tests use synthetic data only.
4. Update the documentation and add a line to `CHANGELOG.md` under "Unreleased".
5. Open a pull request describing what changed and how you checked it.

Rules that keep results trustworthy:

- **Predictions are never modified after the pipeline writes them.** Review, finalization, and
  results write new files.
- **Changes that alter scientific outputs** (segmentation, features, typing, QC thresholds,
  finalization) must say so in the pull request and include a before/after comparison on the
  synthetic references. See [docs/baseline_comparison_protocol.md](docs/baseline_comparison_protocol.md).
- **The supported code (`src/`, `fibertypeqc/`, `scripts/`) must not import `research/`**; a test
  enforces this. See [ARCHITECTURE.md](ARCHITECTURE.md).
- **This repository is public.** Do not commit microscopy images, real specimen identifiers,
  per-fiber tables from real data, results from unpublished studies, or models trained on
  unpublished data. `scripts.check_repository` and the pre-commit hook catch common mistakes.

## Code of conduct

Everyone taking part is expected to follow the [code of conduct](CODE_OF_CONDUCT.md).
