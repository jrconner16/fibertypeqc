# AGENTS.md

Rules of the road for working on FiberTypeQC.

- Keep changes scoped to the requested task.
- Do not commit raw microscopy data, MyoSight exports, private validation outputs, local paths, or generated `outputs/`.
- Do not modify source code when documentation-only fixes are enough.
- Prefer small, reviewable edits over broad refactors.
- Use existing pipeline conventions unless the release plan says otherwise.
- Before changing CLI docs, verify script arguments against the source.
- Supported workflow is:
  `run_batch` -> `make_review_project` -> `generate_review_qc` -> `review_project_napari` ->
  `finalize_review_project` -> `summarize_results` (see `README.md`). The per-image
  `review_labels_napari` / `merge_reviewed_labels` path is legacy.
- Treat `ui_napari.py` as experimental unless explicitly promoted.
- Report files changed and commands/tests run at the end of each task.
- If a task risks changing scientific behavior, stop and explain before editing.



## Public/private boundary

This repository is public.

Do not commit or push:
- unpublished biological results or evaluation metrics;
- real animal/specimen identifiers;
- real cohort assignments, splits, or private manifests;
- real manual review decisions;
- unpublished study-specific metadata;
- models trained from unpublished/private biological data unless explicitly approved;
- private filesystem paths, usernames, credentials, or infrastructure-specific identifiers.

Use synthetic, schema-only, published, or otherwise public-safe fixtures for tracked examples,
tests, manifests, and reproducibility demonstrations.

Private biological applications of FiberTypeQC belong in ignored private storage or outside Git.

Model artifacts trained on private or unpublished data are never committed. Register their identity
and digest, and resolve the artifact at runtime from a private model root.

Before every commit or push, inspect the staged diff for study-specific or private material.
If publication safety is ambiguous, stop and ask rather than assuming it is safe.

Automated checks (run by the pre-commit hook and CI):

```bash
uv run pre-commit install          # once per clone
uv run python -m scripts.check_repository
```

`check_repository` also scans against a private identifier denylist when one exists at
`~/.config/fibertypeqc/private_denylist.txt` (or `$FIBERTYPEQC_PRIVATE_DENYLIST`). The denylist is
kept outside Git and must never be committed. Public CI runs without it, so run the check locally
before pushing.

Git history was sanitized on 2026-09-28. Work only in a clone made after that date; never push from
an older clone, which would republish removed history.

## Git/GitHub discipline

Before starting a substantial work package:
- confirm current branch and upstream;
- fetch remote changes;
- inspect staged, unstaged, and untracked work;
- understand pre-existing changes before adding new work.

After each verified bounded work package:
- inspect the diff;
- stage only intended public-safe files;
- run the relevant checks/tests;
- make a descriptive commit;
- push it;
- report the commit hash and whether the branch is clean and synchronized with its upstream.

Do not allow several completed work packages to accumulate locally without a Git checkpoint.

## Documentation discipline

Use the current active roadmap as the execution source of truth.

Current roadmap:
`ROADMAP.md` (public-safe). Study-specific planning is kept outside Git.

Do not create another roadmap, handoff, or planning Markdown file when the information can reasonably
be added to the active roadmap, README, CHANGELOG, AGENTS.md, or an existing durable documentation page.

If a new planning document is genuinely needed, explain why the existing source of truth is
insufficient before creating it.
