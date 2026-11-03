# User Guide

Everything needed to run FiberTypeQC on your own images. Install first ([install.md](install.md));
try the [demo](demo.md) to see the whole workflow on small public images. Run every command from
the repository root.

## Set up a project

One command creates a project folder for a study. It asks which staining panel you used (or lets
you enter the channel for each stain), finds your images, and writes three small files:

```bash
uv run python -m fibertypeqc init myproject/ --images /path/to/images
```

```
myproject/
  fibertypeqc_project.yaml   # images folder, panel, sample sheet, model
  panel.yaml                 # which channel holds which stain
  samples.csv                # one row per image: fill in mouse_id (and any group columns)
```

Then open `samples.csv` and fill in `mouse_id` for each image. Any other column you add or fill
(for example `group` or `genotype`) becomes a condition for cohort summaries. If mouse IDs are in
your file names, `--mouse-id-pattern` fills them in, e.g. `--mouse-id-pattern '^(m\d+)_'`.

To run unattended, give everything as options:

```bash
uv run python -m fibertypeqc init myproject/ --images /path/to/images \
  --panel four_marker_i_iia_laminin_iib --model /path/to/quad_four_class_rf_v1.joblib --yes
```

**Panel presets** (`manifests/panels/`): `four_marker_i_iia_laminin_iib` (Type I, IIa, laminin,
IIb in channels 0–3; IIx inferred) and `three_marker_iib_iia_laminin` (IIb, IIa, laminin; IIx
inferred). Channels count from 0. If your channel order differs, choose "custom" in the wizard or
edit `panel.yaml`; details in [panel_schema.md](panel_schema.md).

**Model**: a registered model ID, or the path to your model file. With an ID for a model that is
not in the repository, set `export FIBERTYPEQC_MODEL_ROOT=/folder/containing/the/model`. If no
model fits your panel, use `model: none` ([Panels without a model](#panels-without-a-model)).

## End-to-end workflow

Each step reads the previous step's files and writes new ones; predictions are never modified.

```bash
uv run python -m fibertypeqc status   myproject/   # where am I, and what is the next command
uv run python -m fibertypeqc run      myproject/   # 1. segment, type, and QC every image (slow)
uv run python -m fibertypeqc prepare  myproject/   # 2-3. build the review project, run project QC
uv run python -m fibertypeqc review   myproject/   # 4. review in Napari (GUI); saves automatically;
                                                   #    "Finalize and build report" does step 5-6 there
uv run python -m fibertypeqc finalize myproject/   # 5-6. finalized tables, summaries, HTML report
```

Then open `myproject/results/cohort_report.html`. The folder ends up with `batch/` (masks, fiber
tables), `review/` (QC and saved decisions), `final/`, and `results/`.

Notes:

- `run` is the slow step (Cellpose); use a GPU node or a batch job for many images. It checks the
  model, panel, and inputs before processing and stops with one message if something is missing.
  Extra options are passed through, e.g. `run myproject/ --reuse-artifacts auto` to reuse masks
  from an earlier run. The other steps take seconds to minutes and can be re-run at any time.
- `finalize` works before any review; every fiber then keeps its model prediction.
- For cropped fields, add `exclude_image_border_fibers: true` to `fibertypeqc_project.yaml` to
  leave fibers cut off by the image edge out of the results (counted in the report). Off by
  default.
- `prepare` keeps an existing review project (and its decisions) rather than rebuilding it.
- With split CZIs, tissue pieces imaged within one Zeiss scene stay together in one section.
- Keep the project folder on storage with room: masks and exported scenes are about the size of
  the raw images.

<details>
<summary>The individual commands behind each step</summary>

```bash
uv run python -m scripts.run_batch --input-dir images/ --panel-config panel.yaml \
  --split-czi-scenes --model MODEL --output-dir myproject/batch
uv run python -m scripts.make_review_project --batch-dir myproject/batch \
  --sample-sheet samples.csv --panel-config panel.yaml --project-dir myproject/review
uv run python -m scripts.generate_review_qc --project myproject/review/project.yaml
uv run python -m scripts.review_project_napari --project myproject/review/project.yaml \
  --reviewer YOUR_NAME --display-downsample 2
uv run python -m scripts.finalize_review_project --project myproject/review/project.yaml \
  --output-dir myproject/final
uv run python -m scripts.summarize_results --final-dir myproject/final \
  --project myproject/review/project.yaml --output-dir myproject/results
```

`make_review_project` never overwrites an existing project.

</details>

## Review

The review workspace has four parts: cohort QC, per-section pass/fail/exclude, guided
fiber-by-fiber review of flagged fibers (`K` keeps the model call, `1`–`4` set I/IIa/IIb/IIx), and
regions (draw an area to exclude, or name analysis ROIs). Fibers are flagged for low model
confidence or margin and for a faint or unusually thick laminin outline; "Why shown" names the
reason. Workflow: [review_guide.md](review_guide.md); component details:
[review_reference.md](review_reference.md).

### Blind reference labels

To measure how well the model does on your images, label fibers without seeing the model:

```bash
uv run python -m fibertypeqc review myproject/ --blind --reviewer YOUR_NAME
uv run python -m fibertypeqc reference myproject/   # writes myproject/reference_export/
```

Blind mode hides model calls, confidence, QC reasons, and section and mouse names. Choose a
reproducible random sample (150 fibers per section by default), a computer-placed field, or fields
you draw. Blind labels are stored per reviewer and never change results. The export lists every
reviewer's labels beside the model's output, field coverage, reviewer agreement (percent and
Cohen's kappa), and review time.

An optional `evaluation_roles.csv` beside `fibertypeqc_project.yaml` (columns `mouse_id`, `role` =
`pool` or `test`) keeps mice apart: blind labelling offers only test mice, guided review and model
improvement only pool mice. Without the file nothing is restricted.

## Outputs

| Step | Output | Notes |
|---|---|---|
| `run` | per image: `*_cellpose_labels.tif`, `*_fibers.csv`, `*_summary.csv`, `*_run.json`, QC JSON, `*_result_report.html` | `*_fibers.csv` holds the model's call, probabilities, confidence, and margin; it is never modified later |
| `prepare` | `qc/image_qc.csv`, `fiber_qc.csv`, `section_selection.csv` | technical QC and which sections are used |
| `finalize` | `<image_id>_fibers_finalized.csv`, `final_fiber_table.csv`, `finalization_manifest.json` | every model column plus `final_type` and `value_source` (`predicted`, `reviewed`, `excluded`, `unresolved`) |
| `finalize` | `image_summary.csv`, `mouse_summary.csv`, `cohort_summary.csv`, `roi_summary.csv`, `cohort_report.html` | CSVs are the source of truth; the report is built only from them |

How final values are decided:

- exclusions win: an excluded section, then a drawn exclusion region (by fiber center), then a
  fiber the reviewer excluded; otherwise the reviewer's decision; otherwise the model's call;
- fibers flagged for review but not reviewed keep the model's call and are counted in the report;
- unresolved fibers have no final type and are left out of percentages, with their count reported;
- mouse-level results pool fibers across that mouse's sections; cohort results treat each mouse as
  one unit (mean ± SD);
- finalization refuses a section whose mask or fiber table changed after review began.

Column definitions: [output_schema.md](output_schema.md).

## Automatic QC

Per image, written to `*_postrun_qc.json` and the summary. A warning never removes fibers.

| Check | Warns when |
|---|---|
| `postrun.fiber_count` | fewer than 300 fibers |
| `postrun.uncertainty_rate` | more than 35% of fibers are low-confidence or unresolved |
| `postrun.residual_rate` | share of the inferred class (IIx); informational until `--qc-max-residual-rate` is set |
| `postrun.median_area` | median fiber area outside 200–15,000 px² |
| `postrun.marker_correlation` | marker channels correlate above 0.92 (possible crosstalk or wrong channels) |

## Models

| Model ID | Panel | Classes | Where the file is |
|---|---|---|---|
| `quad_four_class_rf_v1` (default) | laminin, Type I, IIa, IIb | I, IIa, IIb, IIx (inferred) | release download; verified by SHA-256 |
| `ta_three_class_logistic_v1` | laminin, IIa, IIb | IIa, IIb, IIx (inferred) | release download; verified by SHA-256 |
| `synthetic_four_class_reference_v1` | laminin, Type I, IIa, IIb | I, IIa, IIb, IIx | in the repository; synthetic test fixture only |
| `rebaseline_tile_v2_p75p90_iib_iia_iix` | laminin, IIa, IIb | IIa, IIb, IIx (inferred) | in the repository; retired historical model |

`--model` takes a model ID or a path to a registered model file. The registry is
`manifests/model_registry.v1.yaml` ([model_registry.md](model_registry.md)). A model's manifest can
pin the feature-extraction settings it was trained with; the pipeline then uses exactly those and
refuses typing options that would change them. Model cards:
[quad_four_class_rf_v1](model_cards/quad_four_class_rf_v1.md),
[ta_three_class_logistic_v1](model_cards/ta_three_class_logistic_v1.md).

## Panels without a model

If no model fits your staining panel, set `model: none` in `fibertypeqc_project.yaml`. `run` then
segments and measures every fiber but types none of them: every call is `unclassified`, and the
marker measurements are written to `*_feature_diagnostics.csv`. You label a sample of fibers
blind, and `build` turns those labels into candidate models.

```yaml
model: none
classes: [i, iia, negative]      # optional; this is the default for a Type I + IIa panel
```

- `classes` defaults to one class per myosin marker the panel measures, plus `negative`. Add
  `hybrid` to the list to offer it. A class for a marker the panel does not measure is refused.
- `negative` means negative for every measured myosin marker and is shown with the panel's
  markers (for example I⁻/IIa⁻). It is never counted as IIx or IIb.
- `unclassified` is not a class: it only means no model has typed the fiber yet. It is separate
  from a reviewer's `uncertain` and from the historical rule call `unknown`.

After `run` and `prepare`, label fibers with
`uv run python -m fibertypeqc review myproject/ --blind --reviewer NAME`
([Blind reference labels](#blind-reference-labels)); the buttons offer the configured classes.
`finalize` and `improve` need model calls and refuse a project until a model is promoted.

Then build candidate models from the blind labels:

```bash
uv run python -m fibertypeqc build myproject/            # --reviewer NAME to use one reviewer
```

`build` follows a recipe fixed in advance (`manifests/builder/recipe.v1.yaml`):

- It uses blind labels from training mice only. With `evaluation_roles.csv`, `test` mice are
  never read; every mouse can still be labelled blind, so test mice can be checked later.
- Labels from several reviewers are combined per fiber: agreement counts once, disagreements are
  dropped and counted, and `exclude`/`uncertain` are left out.
- It needs at least 3 training mice and, for each class, at least 30 labelled fibers from at least
  2 mice with at least 5 each.
- Models are trained on marker intensity measurements only, never fiber area or shape, and the
  manifest pins the measurement settings of the run.
- Three candidate model types are compared with a majority-class baseline by leave-one-mouse-out.
  A candidate is recommended only if its median per-mouse macro-F1 is at least 0.10 above the
  baseline and every class has out-of-fold recall of at least 0.70. Per-mouse support, recall
  (N/A when a mouse has none of a class), and macro-F1 are written to
  `models/build_<time>/selection/`.
- Every candidate is then refitted on all training mice and saved with its manifest in
  `models/build_<time>/candidates/`, with or without a recommendation.

These checks are minimum safeguards, not an accuracy estimate. `build` never switches the project
to a model; evaluating on test mice and switching are separate steps.

Choose one candidate (normally the recommended one), score it on the test mice, then switch:

```bash
uv run python -m fibertypeqc evaluate myproject/ logistic   # independent test evaluation
uv run python -m fibertypeqc promote  myproject/ logistic   # switch the project to it
uv run python -m fibertypeqc promote  myproject/ --rollback # back to no model, exactly
```

- `evaluate` scores only the candidate you name, on blind labels from test mice only, using the
  measurements already on disk; it changes nothing in the project. Results go to
  `models/evaluations/`, and every evaluation is logged in `models/evaluation_history.csv` with how
  many times the same test mice have been used. Choosing between models by their test scores makes
  the test no longer independent.
- An evaluation is tied to the exact candidate file and to the test labels, masks, and
  measurements it used. If any of these change, it is out of date.
- With test mice declared, `promote` requires a current evaluation of that candidate; `--override`
  switches anyway and is recorded in `models/independent_test_checks.csv`. Without test mice,
  `promote` warns that there was no independent test and records that. Test scores never decide
  for you: switching is your decision.
- Switching re-types every section on its existing mask and first checks that the measurements
  reproduce exactly those the model was built from. Blind labels are never changed, and
  `--rollback` restores the project as it was before the switch.

After the switch the project works like any project with a model: `review` (guided), `finalize`,
and `improve` are available, and the review buttons keep the classes and the panel's name for
`negative`. `classes` is removed from `fibertypeqc_project.yaml` because the model now sets them.
In results, `negative` keeps its panel-specific meaning: `results_manifest.json` records the name
(`class_labels`), and the cohort report shows it (for example "I⁻/IIa⁻ final") with a note that
these fibers are negative for every measured myosin marker, not IIx or IIb. The per-section run
report lists only the standard classes; the counts are in `*_fibers.csv` and the results tables.

Describe the model with the [model-card template](model_cards/built_model_template.md); each field
names the build or evaluation file it comes from. A model trained on private data, and its card,
stay in private storage.

## Adapt the model from your review

If the model does poorly on your images, your review decisions can train a replacement:

```bash
uv run python -m fibertypeqc improve myproject/
```

This compares three candidate model types with the current model on held-out mice, using only the
fibers you reviewed, and saves each candidate with a manifest in `myproject/models/`. It prints a
recommendation from a rule fixed in advance (`manifests/improver/recipe.v1.yaml`) and never
changes the model your project uses. It needs reviewed fibers from at least three mice and ten per
class. The comparison is between models on reviewed fibers, which are mostly hard cases; it is
not an accuracy estimate.

To switch the project to a candidate, and to undo that:

```bash
uv run python -m fibertypeqc promote myproject/ logistic      # add --override if not recommended
uv run python -m fibertypeqc promote myproject/ --rollback
```

Switching re-types every section on its existing mask (no segmentation), keeps your review
decisions, archives the previous outputs under `myproject/models/history/`, and records the change
in `myproject/models/model_history.csv`.

## Check a run against a reference

To confirm that a later run (new code version, another computer) reproduces an earlier one, freeze
a snapshot of a project's batch outputs and check against it later:

```bash
uv run python -m fibertypeqc snapshot myproject/ --freeze    # writes myproject/reference_snapshot.json
uv run python -m fibertypeqc snapshot myproject/ --check     # exit code 0 when every section matches
```

A section with an identical mask must have identical fiber calls. A re-segmented section (masks
can differ slightly between devices) must agree on fiber count within 2% and on every class share
within 2 percentage points; adjust with `--count-tolerance` and `--composition-tolerance`. Snapshots
contain image IDs and counts from your data, so keep snapshots of private data private.

## Troubleshooting

- **`No module named …`, or the wrong Python version**: you are not in the repository root.
- **Not sure what to run next**: `uv run python -m fibertypeqc status myproject/`.
- **"distributed privately" / "not a registered model"**: pass `--model /path/to/model.joblib` or
  set `FIBERTYPEQC_MODEL_ROOT`; the file must be the registered model (checked by SHA-256).
- **"contains N scenes"**: set `split_czi_scenes: true` in the project config (or add
  `--split-czi-scenes` to `run_batch`).
- **"non-channel dimensions (Z=…)"**: project or split Z/T stacks into single-plane multichannel
  images first. TIFFs without channel metadata need ImageJ axes such as `CYX`.
- **Batch reports fewer sections than expected**: sections are Zeiss scenes; the batch log prints
  the scene count for each CZI.
- **Project has fewer images than the batch**: `make_review_project` prints `warning: skipped …`
  with the reason for each (failed in the batch, not in the sample sheet, or missing masks).
- **Poor segmentation**: check the laminin channel in the panel file. `run_pipeline` (single image)
  accepts `--diameter` (default 30) and `--cellpose-normalize`.
- **Out of memory in review**: use `--display-downsample 4`.

## Single images, historical model, and legacy tools

### Run one image

```bash
uv run python -m scripts.run_pipeline --input image.tif --output-dir out \
  --panel-config my_panel.yaml --model MODEL
```

Add `--labels-path path/to/labels.tif` to use an existing or manually corrected mask, or
`--reuse-artifacts auto` to reuse compatible segmentation in the same output directory
(`required` makes reuse mandatory). `--export-diagnostics` writes a separate
`*_feature_diagnostics.csv` for model development; it does not change `*_fibers.csv`. Every
successful run also writes `*_result_report.html`, a self-contained page of QC checks and
artifacts.

### Optional marker and nuclear outputs (experimental)

A panel file can also map Type I, direct IIx, eMHC, and DAPI channels, for example
`manifests/i_iia_laminin_dapi_panel.yaml`:

```bash
uv run python -m scripts.run_pipeline --input image.czi --output-dir out \
  --panel-config manifests/i_iia_laminin_dapi_panel.yaml --export-diagnostics
```

Extra markers are measured in the diagnostics table. A DAPI mapping also runs nuclear segmentation
and fiber association under `nuclear/`. These outputs are not validated and do not change fiber
calls. Configuration: [panel_schema.md](panel_schema.md).

### Historical three-class model

```bash
uv run python -m scripts.run_pipeline --input image.czi --output-dir out \
  --iib-channel 0 --iia-channel 1 --membrane-channel 2 \
  --typing-preprocess tile_subtract --typing-tile-size 256 --typing-erode-px 2 \
  --classifier-path data/models/rebaseline_tile_v2_p75p90_iib_iia_iix.joblib
uv run python -m scripts.run_batch --input-dir images/ \
  --model rebaseline_tile_v2_p75p90_iib_iia_iix --output-dir out_batch
```

IIx is inferred as the class unstained for IIb and IIa. Legacy aliases `--type1-channel` and
`--type2-channel` remain accepted. Model card: [data/models/model_card.md](../data/models/model_card.md).

### Legacy per-image reviewer

```bash
uv run python -m scripts.review_labels_napari --image image.czi \
  --labels out/image_cellpose_labels.tif --fibers out/image_fibers.csv \
  --output out/image_fibers_manual_review.csv
```

Pass `--channel-config` for a panel-aware run and `--nuclei-labels` to show the nuclear overlay.
Its corrections import into finalization with `--legacy-review IMAGE_ID=CSV`.
`scripts.merge_reviewed_labels` is deprecated; use `scripts.finalize_review_project`.
