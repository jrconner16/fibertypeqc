# Fiber Typing V1 Review Workflow

This is the current minimal workflow. Cellpose masks stay fixed; manual review edits a
fiber-type correction table, not the segmentation mask.

Current biological mapping:

- `--type1-channel` / old `type1` = **IIb**
- `--type2-channel` / old `type2` = **IIa**
- blank/low marker staining = **IIx candidate**
- `uncertain` = reviewed but not trustworthy as a biological type
- `exclude` = artifact/unusable

## 1. Run first-pass pipeline

Example:

```bash
uv run python -m scripts.run_pipeline \
  --input path/to/image.czi \
  --output-dir outputs/v0_run/image_name \
  --type1-channel 0 \
  --type2-channel 1 \
  --membrane-channel 2 \
  --typing-erode-px 2
```

By default, type-channel preprocessing now uses `--typing-preprocess global_subtract`.
This subtracts a low global background quantile and avoids Gaussian high-pass background
subtraction for fiber-type calling. Do not use `--typing-preprocess gaussian_subtract`
for final calling unless you are deliberately reproducing old/noisy behavior.

If an image has a broad field/background gradient, use `--typing-preprocess tile_subtract`
and set `--typing-tile-size` to `256` or `128`. This is intended for broad illumination
correction, not for rescuing punctate Gaussian-like signal.

Outputs:

- `*_cellpose_labels.tif`
- `*_fibers.csv`
- `*_summary.csv`

## 2. Review fiber types in napari

Example:

```bash
uv run python -m scripts.review_labels_napari \
  --image path/to/image.czi \
  --labels outputs/v0_run/image_name/image_name_cellpose_labels.tif \
  --fibers outputs/v0_run/image_name/image_name_fibers.csv \
  --type1-channel 0 \
  --type2-channel 1 \
  --membrane-channel 2 \
  --typing-preprocess global_subtract \
  --typing-erode-px 2 \
  --threshold-floor 1.25 \
  --signal-scale 6
```

For unusually noisy images, use stricter display settings:

```bash
--threshold-floor 2 --signal-scale 10
```

Hotkeys:

- `b` = `IIb`
- `a` = `IIa`
- `x` = `IIx` blank
- `h` = `hybrid`
- `u` = `uncertain`
- `e` = `exclude`

Output:

- `*_fibers_manual_review.csv`

## 3. Merge manual review into fiber table

```bash
uv run python -m scripts.merge_reviewed_labels \
  --fibers outputs/v0_run/image_name/image_name_fibers.csv \
  --review outputs/v0_run/image_name/image_name_fibers_manual_review.csv
```

Output:

- `*_fibers_reviewed.csv`

Important columns:

- `predicted_internal_type`: first-pass rule/model call using older internal names
- `predicted_biological_type`: first-pass call mapped to biological labels
- `corrected_type`: manual correction, if reviewed
- `label_source`: `auto_rule`, `unreviewed`, or `manual_gold`
- `final_type`: best current label for downstream analysis
- `is_uncertain`, `is_hybrid`, `is_excluded`: manual flags

## 4. Debug one fiber

Use this when a clicked fiber looks wrong:

```bash
uv run python -m scripts.debug_fiber \
  --fibers outputs/v0_run/image_name/image_name_fibers.csv \
  --fiber-id 237
```

This prints the prediction, thresholds, signal features, coverage, and scores for that fiber.

## Current Guardrails

- Do not treat old labels or first-pass labels as gold truth.
- Do not redraw masks during fiber-type review.
- Use `uncertain` liberally for borderline signal.
- Use `exclude` for artifacts or fibers inside obvious bad regions.
- Keep channel numbers explicit in commands.
- Type-channel calls use eroded fiber interiors by default (`--typing-erode-px 2`) to reduce
  membrane/interstitial edge signal.
