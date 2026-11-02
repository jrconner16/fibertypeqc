"""Build a first model for a panel that has none, from blind labels (no GUI imports).

Blind labels from the project's training (pool) mice are combined across reviewers, joined with
the marker measurements the no-model run wrote, and used to evaluate each declared candidate by
leave-one-mouse-out against a majority-class baseline. Each candidate is then refitted on every
eligible training mouse and written with a manifest that pins its features and measurement
settings. Test mice never enter labels, selection, or fitting, and nothing here changes the
project's model: promotion is a separate, explicit step.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut

from fibertypeqc.artifacts import file_sha256, git_commit
from fibertypeqc.feature_schema import (
    MULTIPANEL_FEATURE_SCHEMA,
    SEMANTIC_DIAGNOSTIC_SUFFIXES,
    SEMANTIC_MARKER_NAMES,
)
from src.fiber_type_labels import MYOSIN_CLASSES, normalize_review_label
from src.model_improver import build_estimator
from src.review.finalization import image_input_fingerprints
from src.review.project import Project
from src.review.roles import POOL, allowed_images
from src.review.schemas import ObjectReviewStatus
from src.review.session import ReviewSession
from src.review.storage import atomic_write_dataframe, load_session

BUILDER_SCHEMA_VERSION = "fibertypeqc.model_builder.v1"
BASELINE = "majority_baseline"
DEFAULT_RECIPE = Path(__file__).resolve().parents[1] / "manifests/builder/recipe.v1.yaml"
CAVEAT = (
    "Scores are leave-one-mouse-out on the training mice' blind labels and only compare the "
    "candidates with each other and with a majority-class baseline. They are minimum safeguards, "
    "not an accuracy estimate: check a built model on independent test mice before relying on it."
)


class BuilderError(ValueError):
    """The labels or measurements cannot support building a model."""


def load_recipe(path: Path | None = None) -> dict[str, Any]:
    recipe = yaml.safe_load((path or DEFAULT_RECIPE).read_text(encoding="utf-8"))
    for key in ("recipe_id", "minimums", "candidates", "recommendation_rule"):
        if key not in recipe:
            raise BuilderError(f"Builder recipe is missing '{key}'.")
    return recipe


def training_features(markers: tuple[str, ...] | list[str]) -> list[str]:
    """Marker-intensity features for the panel's myosin markers, in a fixed order.

    Fiber area, shape, DAPI, and eMHC are deliberately not used to type fibers.
    """
    return [
        f"{SEMANTIC_MARKER_NAMES[marker]}.{suffix}"
        for marker in MYOSIN_CLASSES
        if marker in markers
        for suffix in SEMANTIC_DIAGNOSTIC_SUFFIXES
    ]


# --- labels ---------------------------------------------------------------------------------


def blind_label_sessions(
    project: Project, reviewers: list[str] | None = None
) -> dict[str, ReviewSession]:
    """Each reviewer's saved blind labels (their final decision per fiber)."""
    sessions = {}
    for state in sorted((project.root / "reference").glob("*/review_state.json")):
        name = state.parent.name
        if reviewers and name not in reviewers:
            continue
        view = project.for_reference(name)
        sessions[name] = load_session(
            view.review_state_path, expected_project_id=project.project_id
        )
    missing = sorted(set(reviewers or ()) - set(sessions))
    if missing:
        raise BuilderError(f"No blind labels from reviewer(s): {', '.join(missing)}.")
    if not sessions:
        raise BuilderError(
            "No blind labels yet. Label fibers with: review PROJECT --blind --reviewer NAME"
        )
    return sessions


def combine_labels(
    project: Project, sessions: dict[str, ReviewSession], classes: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One row per labelled training fiber; reviewer disagreements are dropped and counted."""
    training_images = allowed_images(project, POOL)
    known = {image.image_id: image for image in project.images}
    current: dict[str, dict[str, str]] = {}
    left_out = {
        "test_mice": 0,
        "excluded": 0,
        "uncertain": 0,
        "outside_classes": 0,
        "inputs_changed_since_labelling": 0,
        "unknown_section": 0,
    }
    rows = []
    for reviewer, session in sessions.items():
        for decision in session.object_decisions:
            image_id = decision.image_id
            if image_id not in known:
                left_out["unknown_section"] += 1
                continue
            # Test mice are isolated completely: their labels are not read any further.
            if training_images is not None and image_id not in training_images:
                left_out["test_mice"] += 1
                continue
            status = decision.review_status
            label = normalize_review_label(decision.reviewed_fiber_type)
            if status is ObjectReviewStatus.EXCLUDED or label == "exclude":
                left_out["excluded"] += 1
                continue
            if status in (ObjectReviewStatus.UNCERTAIN, ObjectReviewStatus.UNRESOLVED) or (
                label == "uncertain"
            ):
                left_out["uncertain"] += 1
                continue
            if label not in classes:
                left_out["outside_classes"] += 1
                continue
            if image_id not in current:
                current[image_id] = image_input_fingerprints(known[image_id])
            if session.input_fingerprints.get(image_id) != current[image_id]:
                left_out["inputs_changed_since_labelling"] += 1
                continue
            rows.append(
                {
                    "image_id": image_id,
                    "fiber_id": int(decision.fiber_id),
                    "reviewer": reviewer,
                    "label": label,
                }
            )
    columns = ["mouse_id", "image_id", "fiber_id", "label", "n_reviewers", "reviewers"]
    if not rows:
        return pd.DataFrame(columns=columns), {
            "labels_read": 0,
            "left_out": left_out,
            "conflicting_fibers": 0,
            "agreeing_duplicates_merged": 0,
        }
    raw = pd.DataFrame(rows)
    combined, conflicts, merged = [], 0, 0
    for (image_id, fiber_id), group in raw.groupby(["image_id", "fiber_id"], sort=True):
        labels = set(group["label"])
        if len(labels) > 1:
            conflicts += 1
            continue
        merged += len(group) - 1
        combined.append(
            {
                "mouse_id": known[image_id].mouse_id,
                "image_id": image_id,
                "fiber_id": int(fiber_id),
                "label": labels.pop(),
                "n_reviewers": len(group),
                "reviewers": ";".join(sorted(group["reviewer"])),
            }
        )
    audit = {
        "labels_read": int(len(raw)) + sum(left_out.values()),
        "left_out": left_out,
        "conflicting_fibers": conflicts,
        "agreeing_duplicates_merged": merged,
    }
    return pd.DataFrame(combined, columns=columns), audit


def label_support(table: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Labelled fibers per class and training mouse."""
    counts = pd.crosstab(table["label"], table["mouse_id"]).reindex(classes, fill_value=0)
    counts.index.name = "class"
    return counts


def check_minimums(table: pd.DataFrame, classes: list[str], recipe: dict[str, Any]) -> None:
    rules = recipe["minimums"]
    problems = []
    n_mice = table["mouse_id"].nunique()
    if n_mice < int(rules["training_mice"]):
        problems.append(f"labels come from {n_mice} training mice; {rules['training_mice']} needed")
    support = label_support(table, classes) if len(table) else None
    for name in classes:
        per_mouse = support.loc[name] if support is not None else pd.Series(dtype=int)
        total = int(per_mouse.sum())
        contributing = int((per_mouse >= int(rules["labels_per_contributing_mouse"])).sum())
        if total < int(rules["labels_per_class"]):
            problems.append(f"{name}: {total} labelled fibers; {rules['labels_per_class']} needed")
        if contributing < int(rules["mice_per_class"]):
            problems.append(
                f"{name}: {contributing} mice with at least "
                f"{rules['labels_per_contributing_mouse']} labels; {rules['mice_per_class']} needed"
            )
    if problems:
        raise BuilderError("Not enough blind labels to build a model: " + "; ".join(problems) + ".")


# --- measurements ---------------------------------------------------------------------------


def _run_record(project: Project, image_id: str) -> dict[str, Any]:
    image = project.image(image_id)
    path = image.prediction_directory / f"{image_id}_run.json"
    if not path.is_file():
        raise BuilderError(f"{image_id}: run record not found ({path.name}).")
    return json.loads(path.read_text(encoding="utf-8"))


def measurement_settings(project: Project) -> tuple[dict[str, Any], dict[str, Any]]:
    """The measurement settings and panel channels every section shares, or an error."""
    settings: dict[str, Any] | None = None
    channels: dict[str, Any] | None = None
    for image in project.images:
        record = _run_record(project, image.image_id)
        if record.get("typing") != "none" or "feature_extraction" not in record:
            raise BuilderError(
                f"{image.image_id} was not measured by a run without a model (model: none); "
                "run the project again."
            )
        this_channels = record.get("panel", {}).get("channels")
        if settings is None:
            settings, channels = record["feature_extraction"], this_channels
        elif record["feature_extraction"] != settings or this_channels != channels:
            raise BuilderError(
                f"{image.image_id} was measured with different settings or channel mapping from "
                "the other sections; a model needs one set of measurements. Run every section "
                "with the same options."
            )
    return dict(settings or {}), dict(channels or {})


def join_features(
    project: Project, table: pd.DataFrame, feature_names: list[str]
) -> pd.DataFrame:
    parts = []
    for image_id, group in table.groupby("image_id", sort=True):
        image = project.image(str(image_id))
        path = image.prediction_directory / f"{image_id}_feature_diagnostics.csv"
        if not path.is_file():
            raise BuilderError(f"{image_id}: measurements not found ({path.name}).")
        diagnostics = pd.read_csv(path, low_memory=False)
        missing = [name for name in feature_names if name not in diagnostics.columns]
        if missing:
            raise BuilderError(f"{image_id}: measurements lack {', '.join(missing)}.")
        wanted = diagnostics.loc[:, ["label", *feature_names]].rename(columns={"label": "fiber_id"})
        joined = group.merge(wanted, on="fiber_id", how="left", validate="one_to_one")
        parts.append(joined)
    joined = pd.concat(parts, ignore_index=True)
    values = joined[feature_names].apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    if bad.any():
        raise BuilderError(
            f"{int(bad.sum())} labelled fibers have missing or non-finite measurements; "
            "they may no longer exist in the current masks."
        )
    joined[feature_names] = values
    return joined


# --- evaluation -----------------------------------------------------------------------------


def _majority(labels: pd.Series, classes: list[str]) -> str:
    counts = labels.value_counts()
    top = counts.max()
    return next(name for name in classes if counts.get(name, 0) == top)


def mouse_metrics(
    truth: pd.Series, predicted: pd.Series, mice: pd.Series, classes: list[str]
) -> pd.DataFrame:
    """Per held-out mouse: support and recall per class (NaN when absent) and macro-F1.

    Macro-F1 averages over the classes present in that mouse's labels.
    """
    rows = []
    for mouse, index in mice.groupby(mice).groups.items():
        y, p = truth.loc[index], predicted.loc[index]
        present = [name for name in classes if (y == name).any()]
        row: dict[str, Any] = {
            "mouse_id": mouse,
            "n_labels": int(len(y)),
            "macro_f1": float(f1_score(y, p, labels=present, average="macro", zero_division=0)),
        }
        for name in classes:
            support = int((y == name).sum())
            row[f"support_{name}"] = support
            row[f"recall_{name}"] = float(((y == name) & (p == name)).sum() / support) if (
                support
            ) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(
    table: pd.DataFrame, feature_names: list[str], classes: list[str], recipe: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Leave-one-mouse-out predictions, per-mouse metrics, and pooled scores per model."""
    mice = table["mouse_id"].astype(str)
    X, y = table[feature_names], table["label"].astype(str)
    predictions = table[["mouse_id", "image_id", "fiber_id", "label"]].copy()
    names = [BASELINE, *recipe["candidates"]]
    held_out = {name: pd.Series(index=table.index, dtype=object) for name in names}
    for train, test in LeaveOneGroupOut().split(X, y, groups=mice):
        fold_labels = y.iloc[train]
        if fold_labels.nunique() < 2:
            raise BuilderError("A training fold has only one class; label more fibers.")
        # The baseline uses only this fold's training mice.
        held_out[BASELINE].iloc[test] = _majority(fold_labels, classes)
        for name, spec in recipe["candidates"].items():
            model = build_estimator(spec).fit(X.iloc[train], fold_labels)
            held_out[name].iloc[test] = model.predict(X.iloc[test])
    parts, summary = [], {}
    for name in names:
        predictions[name] = held_out[name].astype(str)
        per_mouse = mouse_metrics(y, predictions[name], mice, classes)
        per_mouse.insert(0, "model", name)
        parts.append(per_mouse)
        pooled = {
            label: float(((y == label) & (predictions[name] == label)).sum() / (y == label).sum())
            for label in classes
        }
        summary[name] = {
            "median_mouse_macro_f1": float(per_mouse["macro_f1"].median()),
            "pooled_recall_by_class": pooled,
        }
    return predictions, pd.concat(parts, ignore_index=True), summary


def decide(
    by_mouse: pd.DataFrame, summary: dict[str, Any], recipe: dict[str, Any]
) -> dict[str, Any]:
    """Apply the recommendation rule to each candidate; never promotes anything."""
    rule = recipe["recommendation_rule"]
    wide = by_mouse.pivot(index="mouse_id", columns="model", values="macro_f1")
    checks, passing = {}, []
    for order, name in enumerate(recipe["candidates"]):
        gain = float((wide[name] - wide[BASELINE]).median())
        lowest = min(summary[name]["pooled_recall_by_class"].values())
        result = {
            "median_mouse_macro_f1_gain_over_baseline": gain,
            "lowest_pooled_class_recall": float(lowest),
        }
        result["passes"] = bool(
            gain >= float(rule["minimum_median_mouse_macro_f1_gain_over_baseline"])
            and lowest >= float(rule["minimum_pooled_class_recall"])
        )
        checks[name] = result
        if result["passes"]:
            passing.append((-summary[name]["median_mouse_macro_f1"], order, name))
    recommended = min(passing)[2] if passing else None
    return {
        "recommended": recommended,
        "reason": (
            f"{recommended} met every condition of the recommendation rule and had the highest "
            "median per-mouse macro-F1 among those that did."
            if recommended
            else "No candidate met every condition of the recommendation rule; none is recommended."
        ),
        "checks": checks,
    }


# --- build ----------------------------------------------------------------------------------


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "project"


def build(
    project: Project,
    sessions: dict[str, ReviewSession],
    *,
    classes: list[str],
    markers: tuple[str, ...] | list[str],
    output_dir: Path,
    recipe: dict[str, Any] | None = None,
    class_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Combine labels, evaluate the candidates, refit them, and write everything."""
    recipe = recipe or load_recipe()
    if project.model_version != "none":
        raise BuilderError(
            "This project already has a model; adapt it with 'improve' instead of 'build'."
        )
    feature_names = training_features(markers)
    if not feature_names:
        raise BuilderError("The panel maps no myosin marker, so there is nothing to type from.")
    settings, channels = measurement_settings(project)
    labels, audit = combine_labels(project, sessions, classes)
    check_minimums(labels, classes, recipe)
    table = join_features(project, labels, feature_names)
    support = label_support(table, classes)

    # 1. Selection: leave-one-mouse-out on the training mice only.
    predictions, by_mouse, summary = evaluate(table, feature_names, classes, recipe)
    decision = decide(by_mouse, summary, recipe)

    output_dir.mkdir(parents=True, exist_ok=False)
    selection = output_dir / "selection"
    selection.mkdir()
    atomic_write_dataframe(output_dir / "training_labels.csv", table)
    atomic_write_dataframe(output_dir / "label_support.csv", support.reset_index())
    atomic_write_dataframe(selection / "held_out_predictions.csv", predictions)
    atomic_write_dataframe(selection / "metrics_by_mouse.csv", by_mouse)
    (output_dir / "recipe.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
    training_digest = file_sha256(output_dir / "training_labels.csv")

    # 2. Final fit: every candidate refitted on all eligible training mice.
    stamp = output_dir.name
    required = ["laminin", *(SEMANTIC_MARKER_NAMES[m] for m in MYOSIN_CLASSES if m in markers)]
    candidates: dict[str, Any] = {}
    for name, spec in recipe["candidates"].items():
        folder = output_dir / "candidates" / name
        folder.mkdir(parents=True)
        model = build_estimator(spec).fit(table[feature_names], table["label"].astype(str))
        artifact = folder / f"{name}.joblib"
        joblib.dump(model, artifact)
        manifest = {
            "manifest_version": 1,
            "model_id": f"{_slug(project.project_id)}__{stamp}__{name}",
            "task": "fiber_identity",
            "feature_schema_version": MULTIPANEL_FEATURE_SCHEMA,
            "required_markers": required,
            "outputs": list(classes),
            "artifact": artifact.name,
            "artifact_sha256": file_sha256(artifact),
            "intended_use": (
                f"Built by FiberTypeQC from {len(table)} blind labels on "
                f"{table['mouse_id'].nunique()} training mice. Not validated; check it on "
                "independent test mice before use."
            ),
            "features": feature_names,
            "feature_extraction": settings,
            "training_data_sha256": training_digest,
            "builder_recipe": recipe["recipe_id"],
            "recommended": name == decision["recommended"],
        }
        if class_labels:
            manifest["class_labels"] = dict(class_labels)
        (folder / f"{name}.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        candidates[name] = {
            "artifact": str(artifact.relative_to(output_dir)),
            "manifest": str((folder / f"{name}.yaml").relative_to(output_dir)),
            "artifact_sha256": manifest["artifact_sha256"],
        }

    result = {
        "schema_version": BUILDER_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "project_id": project.project_id,
        "recipe_id": recipe["recipe_id"],
        "classes": list(classes),
        "features": feature_names,
        "feature_extraction": settings,
        "panel_channels": channels,
        "labels": {
            "reviewers": sorted(sessions),
            "n_fibers": int(len(table)),
            "n_training_mice": int(table["mouse_id"].nunique()),
            "training_mice": sorted(table["mouse_id"].astype(str).unique()),
            "by_class": {name: int(support.loc[name].sum()) for name in classes},
            **audit,
        },
        "selection": {
            "validation": "leave_one_mouse_out",
            "baseline": "majority class of each fold's training mice",
            "metrics": summary,
            "decision": decision,
        },
        "final_models": {
            "fitted_on": "all eligible training mice",
            "training_data_sha256": training_digest,
            "candidates": candidates,
        },
        "test_mice": "not used; evaluate a selected model on them in a separate step",
        "caveat": CAVEAT,
    }
    (output_dir / "build_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
