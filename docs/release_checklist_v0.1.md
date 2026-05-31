# v0.1-alpha Release Checklist

## Scope

- [ ] Public workflow documented: single image, batch, review, merge.
- [ ] `ui_napari.py` remains experimental or excluded from public workflow.
- [ ] One default model documented and included.
- [ ] Demo data decision documented.
- [ ] Confirm MIT license is acceptable for lab/institutional release.

## Privacy

- [ ] Raw `.czi`, `.tif`, `.tiff`, `.zip`, and generated outputs ignored.
- [ ] Private notebooks and lab-specific paths ignored or removed.
- [ ] `git ls-files --others --exclude-standard` shows only intended public files.
- [ ] No absolute private paths in README/docs.

## Docs

- [ ] README commands match CLI help.
- [ ] Channel assumptions documented.
- [ ] Output schema documented.
- [ ] Review workflow documented.
- [ ] Model card included.

## Checks

- [ ] `uv run ruff check fibertypeqc scripts validation src`
- [ ] `uv run python -m scripts.run_pipeline --help`
- [ ] `uv run python -m scripts.run_batch --show-v0-params`
- [ ] `uv run python -m scripts.review_labels_napari --help`
- [ ] `uv run python -m scripts.merge_reviewed_labels --help`
- [ ] `uv run python -m pytest`
