# FiberTypeQC Roadmap

- Updated: 2026-11-02
- Status: **active.** Goal: a clean, citable repository and a submission to the Journal of Open
  Source Software (JOSS).
- Supersedes: [docs/history/ROADMAP_v0.3.md](docs/history/ROADMAP_v0.3.md) (frozen 2026-10-14) and
  earlier plans in [docs/history/](docs/history/).

This file contains only public-safe scope, decisions, and status. Study-specific planning is kept
outside Git.

## Scope

v0.3.0 and v0.4.0 are released and immutable. Tracks A–C polish the existing software and add no
features. The model builder (Track E) is planned development for a later release; it adds a new
path for panels without a model and does not change existing behavior. Production code paths (`src/`, `fibertypeqc/`, `scripts/`, `manifests/`, `configs/`,
`examples/`) do not move, so imports, commands, model loading, and output paths stay compatible.
Method validation is a separate track (D) and does not block JOSS unless it finds a software bug.

## Current state

Done: Track A; B1 (`CITATION.cff` with ORCIDs and affiliations); B2 (Zenodo integration enabled);
C1; C2. HPC use (Slurm batch, Open OnDemand review) is documented in `docs/install.md` and was
checked on a cluster: the demo batch agreed with the macOS reference snapshot within tolerance.

v0.4.0 is released and archived on Zenodo (concept DOI 10.5281/zenodo.23893717; B3, B4 done).

In progress: C3 (outside install and demo run); C5 (paper). C3 fixes ship in a later release; the
concept DOI covers every version.

## Track A — Repository cleanup and presentation

| ID | Task | Priority | Effort | Done when |
|---|---|---|---|---|
| A1 | Remove study-specific material from tracked files | blocker | 0.5 d | repository checks pass; no study-specific names in tracked files |
| A2 | Documentation: move `README_review_workflow.md` to `docs/review_guide.md` and `README_uv_setup.md` to `docs/install.md`; consolidate `docs/quickstart.md` and the README's reference sections into `docs/user_guide.md`; move phase-era notes to `docs/history/`; add a `docs/README.md` index; rewrite or retire `docs/validation_summary.md` | important | 1.5 d | link check passes; no duplicated or contradictory pages |
| A3 | Delete the `analysis/` placeholder; replace `demo_screenshots/` with one current image in `docs/images/`; add `research/README.md` (unsupported study tooling); keep `research/` and `validation/` working | important | 0.5 d | tests and reference run pass |
| A4 | README rewrite (about 150 lines): problem, workflow figure, status and limitations, install and demo, capabilities, QC and review, docs/architecture/tests, citation and licence; plus a five-minute code walkthrough | blocker | 1.5 d | every command checked against its argparse definition |
| A5 | Dependencies: `ruff` to the dev extra; move other non-core packages to extras where safe | important | 0.5 d | fresh `uv sync --frozen` and reference run pass |

## Track B — Archive and citation

| ID | Task | Priority | Effort | Done when |
|---|---|---|---|---|
| B1 | `CITATION.cff` (authors with ORCID, version, licence, repository) | blocker | 1 h | valid CFF; GitHub shows "Cite this repository" |
| B2 | Enable the Zenodo GitHub integration (maintainer account step) | blocker | 15 min | repository switched on in Zenodo |
| B3 | Release v0.4.0 (work since v0.3.0 plus cleanup); Zenodo archives it | blocker | 0.5 d | version DOI and concept DOI resolve; metadata matches `CITATION.cff` |
| B4 | DOI badge and "How to cite" in the README (concept DOI for the software; version DOI for a specific result) | blocker | 1 h | links resolve |

Zenodo archives only releases made after the integration is enabled; v0.3.0 is left as released.

## Track C — JOSS submission

| ID | Task | Priority | Effort | Done when |
|---|---|---|---|---|
| C1 | `CONTRIBUTING.md` (setup, tests, reference run, private-data rule), `CODE_OF_CONDUCT.md`, bug and question issue templates | blocker | 0.5 d | linked from the README |
| C2 | State the demo data's source, de-identification, and licence in `docs/demo.md` | blocker | 1 h | stated |
| C3 | Install and demo run by someone other than the author, from the README only; fix what they report | important | 0.5 d | at least one issue filed and closed |
| C4 | Optional: macOS job in CI (fast tests) | optional | 2 h | green |
| C5 | `paper/paper.md` and `paper/paper.bib`: Summary, Statement of need, State of the field, Software design, Research impact statement, AI usage disclosure; 750–1750 words | blocker | 3 d | draft PDF builds; every claim maps to code or a test |
| C6 | Submit after the repository has been public for six months (created 2026-05-31) | — | 1 h | submitted |

Positioning: FiberTypeQC does not claim a new classifier. Its contribution is an integrated,
auditable workflow: project folders from raw images to a cohort report, QC-driven guided review,
corrections stored apart from predictions with every final value labelled by source, panel-specific
models checked by panel and SHA-256, blind reference labelling, a pre-declared model-promotion rule
with rollback, and reference snapshots. The paper compares it with MyoSight, MuscleJ/MuscleJ2,
Myosoft, QuantiMus, FiNuTyper, QuPath with Cellpose, and ilastik, citing each tool's own
publication, and makes no accuracy claims beyond the model cards.

## Track D — Method validation (not scheduled; does not block JOSS)

Possible studies: hold-out accuracy and calibration of the default model from blind reference
labels; QC-ranked versus random review; performance as labels accumulate; cross-cohort transfer;
segmentation and morphometry validation; nuclear analysis accuracy; comparison with other tools,
reported as disagreement rather than error. Deferred: continuous phenotype models, new network
architectures, feature and classifier ablations, new features.

## Track E — Model builder (in development; does not block the archive or JOSS submission)

Goal: a lab whose panel has no model can build one inside FiberTypeQC, with the safeguards the
model improver already has. Flow: `init` with no model -> `run` (segment and measure; calls
`unclassified`) -> `review --blind` (label a sample) -> `build` (train and evaluate) -> explicit
`promote` -> normal review and `improve`.

Decisions:

- Classes come from the project configuration. `negative` means negative for every measured
  myosin marker and has a panel-specific display label (for example I⁻/IIa⁻); it is never read as
  IIx or IIb. `unclassified` (no model call yet) and `uncertain` (a reviewer decision) stay
  separate concepts. `hybrid` is optional.
- Training features are marker intensity measurements only. Fiber area and morphology are kept
  for QC and analysis but are not used to type fibers. Existing models, including those that use
  area, are unchanged, and their model cards state any dependence on morphology.
- The builder recipe sets minimums (3 independent training mice; 30 labels per class, from at
  least 2 training mice with at least 5 each) and selects candidates by leave-one-mouse-out
  evaluation. A class absent from a held-out mouse has no recall for that mouse (N/A, not zero);
  per-class support and contributing mice are reported.
  A candidate is recommended only if its median per-mouse macro-F1 is at least 0.10 above a
  majority-class baseline and every class has recall of at least 0.70. These are configurable,
  documented safeguards, not guarantees of biological validity. Test mice never take part in
  model selection, and a recommended model is never promoted automatically.

Status: E1–E4 merged; E5 and E6 in progress; E7 optional. Tracked in issue #71.

| ID | Task | Priority | Done when |
|---|---|---|---|
| E1 | Run a project without a model: segmentation and measurements only, calls `unclassified`; QC, prepare, review, finalization, and summaries accept it | blocker | existing projects, the reference run, and the demo snapshot are unchanged |
| E2 | Classes from the project configuration, including `negative` and optional `hybrid` | blocker | blind labelling offers the configured classes |
| E3 | Builder recipe and `build` command (reusing the improver's evaluation) | blocker | candidates, manifests, and report written; recommendation follows the recipe |
| E4 | `promote` from a project without a model, with rollback to unclassified | blocker | promote and rollback are exact |
| E5 | Synthetic end-to-end test in CI (run -> labels -> build -> evaluate -> promote -> snapshot -> results -> rollback) | blocker | green in CI |
| E6 | User guide section and model-card template for built models | important | documented |
| E7 | Public demo of building a model from manual labels on published images | optional | does not delay a release or the submission |

Not in scope: importing labels from other tools, classes other than fiber types, deep-learning
models, segmentation changes.

## Critical path

```text
decisions -> A1 -> A2 -> A3 -> A4 + C1 + C2 -> A5 -> C3 -> B1 -> B2 -> B3 -> B4 -> C5 -> C6
```

Estimated remaining work: 8–12 working days, mostly documentation. Earliest submission: December
2026.

## Open decisions

- What the research-impact statement may say.
- Packaging: keep the locked run-from-repository install (current) or make the package
  installable with pip (larger change; deferred unless needed).
- Wording of the AI usage disclosure.

## Verification for every change

```bash
uv run ruff check .
uv run pytest -m "not integration" -q
uv run python -m scripts.run_reference
uv run python -m scripts.check_repository
```
