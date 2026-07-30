# Fiber-Type Object Review (Phase 4)

The project Napari launcher opens a focused **Guided Review** dock for
fiber-type adjudication. It reads the project manifest's `fiber_table` artifacts
and never modifies them. The cohort dashboard and full image controls remain
available on demand as advanced docks.

```bash
uv run python -m src.review_project_napari --project project.yaml --reviewer "name"
```

For large sections, use display-only downsampling:

```bash
uv run python -m src.review_project_napari \
  --project project.yaml \
  --display-downsample 2
```

`1` is full resolution; `2` is the recommended first smoke-test setting, and
`4` is useful for very large sections. Raw and label layers are sampled
together, preserving fiber IDs; the original prediction artifacts are never
rewritten.

Available queue sources are model/QC flags, low confidence, high entropy, low
probability margin, near-tied probability conflicts, a deterministic random
audit, and full audit. Each decision stores the model type separately from the
reviewed type, decision status, queue source, reason, reviewer, and timestamp
in `review/review_state.json`; its audit event is appended atomically to
`review/review_events.csv`.

Guided Review visibly reports its autosave state and decision count. It resumes
the last saved standard queue and position when the project is reopened; starting
a different plan deliberately replaces that position. A new session simply shows
the flagged-fiber count without starting a queue.

The default random audit is cohort-scoped, uses seed `0`, and selects up to 25
fibers. The headless queue API also supports image- and mouse-scoped samples,
explicit seeds, and sample sizes for reproducible study protocols.

## Guided review workflow

Start by choosing **Review flagged fibers**, **Review this section**, or **View
cohort QC**. The prominent **Navigate review** bar switches to the cohort,
current section, and other domains. After selecting a plan, the dock shows one
fiber at a time with a short context line and plain-language primary action
**Keep model call (K)**. Type corrections and navigation buttons show their
hotkeys directly. Advanced queue controls and uncommon actions are grouped under
the labeled **Advanced review options** control. Saved decisions automatically
advance to the next fiber; **Undo last decision** restores the immediately
preceding value and records a corresponding audit event.

The selected fiber is a thick cyan outline only, so it does not obscure stain signal.
The visible shortcut legend is: Left/Right to navigate, `K` to keep the model
call, `1`/`2`/`3`/`4` for I/IIa/IIb/IIx, `F` to center the current fiber, and
`U` to undo the immediately prior decision.

The persistent navigator moves between cohort QC, the current section, and
domain-specific review. If a dock is closed accidentally, reopen it from the
Napari **Workspace** menu; **Restore review workspace** returns the Guided
Review dock and channel map without restarting. Raw data opens as a
panel-aware stain composite with individually toggleable, color-named layers.
choices do not modify predictions or review data.
The compact, fixed-width Channel Map identifies each role and source channel.
These display choices do not modify predictions or review data.
choices do not modify predictions or review data.

The legacy `review_labels_napari` CSV layout remains readable through the
`load_legacy_fiber_type_decisions` adapter. It is a migration input only; the
project session remains the canonical record. Region review, segmentation edits,
and masks are not part of this phase.
