# AGENTS.md

Rules of the road for working on FiberTypeQC.

- Keep changes scoped to the requested task.
- Do not commit raw microscopy data, MyoSight exports, private validation outputs, local paths, or generated `outputs/`.
- Do not modify source code when documentation-only fixes are enough.
- Prefer small, reviewable edits over broad refactors.
- Use existing pipeline conventions unless the release plan says otherwise.
- Before changing CLI docs, verify script arguments against the source.
- Public v0.1-alpha workflow is:
  `run_pipeline` / `run_batch` -> `review_labels_napari` -> `merge_reviewed_labels`.
- Treat `ui_napari.py` as experimental unless explicitly promoted.
- Report files changed and commands/tests run at the end of each task.
- If a task risks changing scientific behavior, stop and explain before editing.
