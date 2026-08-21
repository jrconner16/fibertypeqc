# Nuclei Review (Phase 6.2)

Open a validated project in the shared Napari review workspace:

```bash
uv run python -m src.review_project_napari \
  --project project.yaml \
  --reviewer "name" \
  --display-downsample 1
```

Choose **Nuclei** in Guided Review, or use **Workspace → Show Nuclei Review**.
The selected image shows its configured raw-image channels (including DAPI/eMHC
when present), fiber labels, reviewed nuclei labels, and an empty **Draft new
nucleus** labels layer.

Use the dock as follows:

- Select a label in **reviewed nuclei labels**, click **Use selected nucleus**,
  then delete it or set its association.
- Paint only in **Draft new nucleus** and click **Save painted draft nucleus**.
  The draft must not overlap an existing nucleus. It receives a stable reviewed
  ID and begins with an unresolved association.
- Set an association status and, for `assigned`, a fiber ID; then click **Save
  association**.

Every action autosaves the review state, event log, reviewed nucleus mask, and
reviewed association CSV. Model artifacts are never modified. Nucleus-mask
edits mark nucleus features, associations, and fiber-level nucleus counts stale;
association-only edits mark the latter two stale. Paint additions require
`--display-downsample 1` so the draft coordinates match the saved label mask.
