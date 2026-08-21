# Image-Level Review (Phase 3)

`src.review_project_napari` now opens the existing cohort dashboard alongside an
image-review dock and the selected raw-image channels. Phase 3 records only
image/domain decisions; it does not edit masks, regions, or individual fibers.

```bash
uv run python -m src.review_project_napari --project project.yaml --reviewer "name"
```

Generate and validate Phase 2A QC before launching. The project manifest paths
are validated when the launcher starts because raw images are opened in this
slice.

The image-review dock retains the current image, active domain, mode, and
position in `review/review_state.json`. Selecting Pass, Review, Fail, or Exclude
immediately saves that snapshot and appends a `set_domain_status` event to
`review/review_events.csv`. Prediction artifacts remain read-only.

The modes control the navigation set for the active domain:

- `qc_gated_automatic`: images with QC REVIEW or FAIL;
- `flagged_review`: QC REVIEW or positive review priority;
- `domain_focused`: every applicable image in the chosen domain; and
- `full_audit`: every applicable image in the chosen domain.

The difference between the last two modes is intentional at this phase: both
provide complete image navigation. Their object-level sampling policies arrive
with Phase 4. A domain marked not applicable by the project manifest cannot be
given a conflicting image-level status.
