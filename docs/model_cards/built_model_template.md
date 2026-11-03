# Model card: `<model_id>`

Template for a model made with `fibertypeqc build`. Copy it next to the model (for example
`myproject/models/MODEL_CARD.md`), fill every `<...>`, and delete rows that do not apply. Each
value names the file it comes from, so the card can be checked against the run.

A card for a model trained on private or unpublished data contains study details: keep it with
the model in private storage, not in a public repository.

## Summary

| Field | Value | Source |
|---|---|---|
| Model ID | `<model_id>` | candidate `.yaml`: `model_id` |
| Artifact SHA-256 | `<sha256>` | candidate `.yaml`: `artifact_sha256` |
| Panel | `<laminin + markers, e.g. laminin, Type I, Type IIa>` | `panel.yaml` |
| Classes | `<e.g. i, iia, negative>` | candidate `.yaml`: `outputs` |
| Meaning of `negative` | `<e.g. I⁻/IIa⁻>`: negative for every myosin marker this panel measures. It is not inferred to be any other type (for example IIx or IIb). | candidate `.yaml`: `class_labels` |
| Estimator | `<logistic / random_forest / gradient_boosting>` with the recipe's fixed parameters | `build_summary.json`; `recipe.yaml` |
| Features | `<n>` marker intensity features (no fiber area or shape) | candidate `.yaml`: `features` |
| Measurement settings | pinned; promotion checked that they reproduce exactly | candidate `.yaml`: `feature_extraction` |
| Builder recipe | `builder_recipe.v1` | candidate `.yaml`: `builder_recipe` |
| Build run | `models/build_<time>/` | folder name |
| Recommended by `build` | `<yes / no>` | `build_summary.json`: `selection.decision` |

## Training labels

| Field | Value | Source |
|---|---|---|
| Training mice | `<n>` | `build_summary.json`: `labels.n_training_mice` |
| Labelled fibers per class | `<class: n, ...>` | `label_support.csv` (class by mouse) |
| Mice contributing to each class | `<class: n, ...>` | `label_support.csv` (class by mouse) |
| Reviewers | `<n; names or roles>` | `training_labels.csv`: `reviewers` |
| Reviewer conflicts dropped | `<n>` | `build_summary.json`: `labels.conflicting_fibers` |
| Left out (uncertain, excluded, changed inputs) | `<counts>` | `build_summary.json`: `labels.left_out` |

How fibers were chosen for labelling (blind sampling, sections, staining batches):
`<describe>`.

## Selection (leave-one-mouse-out on training mice)

| Candidate | Median per-mouse macro-F1 | Gain over majority baseline | Pooled recall per class |
|---|---|---|---|
| `<name>` | `<value>` | `<value>` | `<class: value, ...>` |

Source: `build_summary.json` (`selection.metrics`; the gain is in `selection.decision`) and
`selection/metrics_by_mouse.csv`. These are minimum safeguards for choosing a candidate, not an
accuracy estimate.

## Independent test

| Field | Value | Source |
|---|---|---|
| Test mice | `<n>`, or "none declared" | `evaluation.json`: `test_mice` |
| Labelled test fibers | `<n>` | `evaluation.json`: `labels.n_fibers` |
| Pooled recall per class | `<class: value, ...>` | `evaluation.json` |
| Median per-mouse macro-F1 | `<value>` | `evaluation.json` |
| Uses of these test mice | `<n>` (more than 1 weakens independence) | `models/evaluation_history.csv` |
| Status at promotion | `<current / stale / missing / no_test_mice>`; override `<yes / no>` | `models/independent_test_checks.csv` |

With no test mice, write "No independent test was done" here and do not report accuracy.

## Promotion

| Field | Value | Source |
|---|---|---|
| Promoted on | `<UTC date>` | `models/model_history.csv` |
| Replaced | `none` (first model for this panel) | `models/model_history.csv` |
| Archive for rollback | `<models/history/...>` | `models/model_history.csv` |

## Intended use and limitations

- Intended use: `<panel, muscle, species, imaging setup>`, with flagged fibers checked in review.
- Not validated for: `<other muscles, stains, microscopes, ages, disease models>`.
- Known weak points: `<classes with low recall, few labels, or few mice>`.
- `<Anything else a user should know.>`
