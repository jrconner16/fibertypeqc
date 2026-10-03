# Review Workflow

## Project-based review (release workflow)

The release path reviews a whole project in one Napari workspace and finalizes the decisions into
analysis-ready tables. Prepare the project with steps 1–3 of the
[end-to-end workflow](README.md#end-to-end-workflow), then launch:

```bash
uv run python -m scripts.review_project_napari \
  --project review/batch1/project.yaml --reviewer YOUR_NAME --display-downsample 2
```

`--display-downsample` changes only what is drawn (use `2`, or `4` for very large sections); fiber
IDs and saved decisions always refer to full-resolution masks.

**Workspaces** (switch with the navigator bar or the **Workspace** menu; **Restore review
workspace** brings back closed docks):

1. **Cohort QC** — mice → sections → domains with PASS / REVIEW / FAIL, reasons, and the
   provisional section selection. Read-only.
2. **Image review** — mark a section Pass, Review, Fail, or Exclude per domain. Fail/Exclude for
   fiber segmentation or fiber typing removes the whole section at finalization.
3. **Guided fiber review** — start **Review flagged fibers** (or **Review this section**). One
   fiber at a time, outlined in cyan. Keys: `K` keep the model call, `1`/`2`/`3`/`4` = I/IIa/IIb/IIx,
   Left/Right to move, `F` to center, `U` to undo. Uncertain/exclude are under **Advanced review
   options**. The flagged queue also includes fibers with a faint or unusually thick laminin
   outline (from project QC); "Why shown" names the reason. These are prompts to look, not
   exclusions. Decisions save immediately and resume where you left off.
4. **Regions** — draw a polygon in the yellow *Region shapes* layer, pick a domain and action, and
   apply. *Exclude all analysis* / *ignore fiber typing* removes the fibers whose centers fall
   inside (folds, tears, bad staining); *unresolved* marks them unresolved. **Analysis ROI** with a
   name (e.g. `quad_1`) tags fibers for per-ROI results; once any ROI is drawn on a section, fibers
   outside all ROIs are excluded.

Everything is saved under `<project>/review/` (`review_state.json`, an append-only
`review_events.csv`, and `review_regions.geojson`). Predictions are never edited. When a section is
first opened, the reviewer records digests of its label mask and fiber table; if those files later
change, it warns, and finalization refuses that section rather than applying decisions to the wrong
fibers.

Then finalize and report (steps 5–6 of the end-to-end workflow). Fibers you did not review keep the
model call; flagged-but-unreviewed fibers are counted in the report.

## Legacy per-image reviewer (V1)

The per-image reviewer below still works. Its corrections CSV can be imported into finalization with
`scripts.finalize_review_project --legacy-review IMAGE_ID=CSV`; `merge_reviewed_labels` is
deprecated.

This was the original review-assisted workflow. Routine type/eMHC review edits a correction table.
The same UI also offers explicit segmentation-repair controls; repaired labels are saved to a new
TIFF and must be re-quantified before type review continues.

Current biological mapping:

- `--iib-channel` / old `type1` = **IIb**
- `--iia-channel` / old `type2` = **IIa**
- blank/low marker staining = **IIx candidate**
- `uncertain` = reviewed but not trustworthy as a biological type
- `exclude` = artifact/unusable

## 1. Run first-pass pipeline

Example:

```bash
uv run python -m scripts.run_pipeline \
  --input path/to/image.czi \
  --output-dir outputs/v0_run/image_name \
  --iib-channel 0 \
  --iia-channel 1 \
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
  --iib-channel 0 \
  --iia-channel 1 \
  --membrane-channel 2 \
  --typing-preprocess global_subtract \
  --typing-erode-px 2 \
  --threshold-floor 1.25 \
  --signal-scale 6
```

For a semantic panel, pass the same YAML used by the pipeline via `--channel-config`. Optional
review layers include `--emhc-channel`, `--i-channel`, `--dapi-channel`, and a precomputed
`--nuclei-labels` TIFF. `--display-downsample N` reduces display memory while preserving original
fiber IDs in the review CSV, and `--minimal-layers` omits nonessential image layers.

For unusually noisy images, use stricter display settings:

```bash
--threshold-floor 2 --signal-scale 10
```

Hotkeys:

- `i` = `Type I` (when a manually verified `--i-channel` is supplied)
- `b` = `IIb`
- `a` = `IIa`
- `x` = `IIx` blank
- `h` = `hybrid`
- `u` = `uncertain`
- `e` = `exclude`
- `p` = eMHC `positive`
- `n` = eMHC `negative`

The eMHC widget also supports `uncertain`. Its value is saved in `emhc_manual_label`, separately
from the fiber-type correction.

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
- `emhc_manual_label`: separate manual eMHC assessment
- `is_uncertain`, `is_hybrid`, `is_excluded`: manual flags

`i` records a manual Type I review label only; it does not enable automatic Type I classification.

## 4. Repair a segmentation when needed

Segmentation repair is available only at `--display-downsample 1`, which preserves label geometry.
Use the polygon/brush, boundary-edit, or delete controls, then choose **Save Corrected
Segmentation**. The UI writes `*_cellpose_labels_corrected.tif` without overwriting the original.
Re-run the pipeline with the corrected mask before reviewing types:

```bash
uv run python -m scripts.run_pipeline \
  --input path/to/image.czi \
  --output-dir outputs/corrected/image_name \
  --labels-path outputs/v0_run/image_name/image_name_cellpose_labels_corrected.tif \
  --iib-channel 0 \
  --iia-channel 1 \
  --membrane-channel 2
```

## 5. Debug one fiber

Use this when a clicked fiber looks wrong:

```bash
uv run python -m scripts.debug_fiber \
  --fibers outputs/v0_run/image_name/image_name_fibers.csv \
  --fiber-id 237
```

This prints the prediction, thresholds, signal features, coverage, and scores for that fiber.

## Current Guardrails

- Do not treat old labels or first-pass labels as gold truth.
- Re-quantify a saved corrected mask before assigning final fiber types.
- Use `uncertain` liberally for borderline signal.
- Use `exclude` for artifacts or fibers inside obvious bad regions.
- Keep channel numbers explicit in commands.
- Legacy aliases `--type1-channel` and `--type2-channel` are still accepted.
- Type-channel calls use eroded fiber interiors by default (`--typing-erode-px 2`) to reduce
  membrane/interstitial edge signal.
