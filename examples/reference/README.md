# Deterministic Reference Fixture

This directory contains a public-safe synthetic mechanics fixture for the frozen alpha workflow:

- `synthetic_reference.tif`: deterministic three-channel IIb/IIa/laminin-style TIFF;
- `synthetic_reference_labels.tif`: supplied fiber labels used to avoid Cellpose nondeterminism;
- `panel.yaml`: semantic panel and residual-IIx policy;
- `review_corrections.csv`: deterministic merge corrections;
- `reference_contract.json`: input/model digests and expected output behavior.

Run and validate the complete deterministic path from the repository root:

```bash
uv run python -m scripts.run_reference
```

The command writes to `outputs/reference/` by default. It checks the fixture, config, correction,
and model digests; runs the frozen classifier against the supplied labels; exercises review merging;
and validates schemas, label IDs, model outputs within tolerance, versioned preflight/post-run QC
codes and next actions, the portable result bundle and self-contained HTML report, and final merged
labels.

Regenerate the fixture bytes after an intentional fixture-design change with:

```bash
uv run python -m scripts.generate_reference_fixture
```

Regeneration must be followed by an explicit review of contract digests and golden expectations.

This fixture proves deterministic file, classification, QC-summary, and merge mechanics. It does
not validate Cellpose segmentation, biological accuracy, or transfer to a new acquisition domain.
