"""Turn review decisions into evaluated candidate models (no GUI imports).

Review decisions made during normal QC are joined with the fibers' model features. Each declared
candidate family is evaluated on held-out mice against the current model's own calls on the same
fibers, then fitted on all labelled fibers and written with a manifest. Nothing here changes the
active model: promotion is a separate, explicit step.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fibertypeqc.artifacts import file_sha256, git_commit
from src.fiber_type_labels import normalize_review_label
from src.review.project import Project
from src.review.schemas import ObjectReviewStatus
from src.review.session import ReviewSession
from src.review.storage import atomic_write_dataframe

IMPROVER_SCHEMA_VERSION = "fibertypeqc.model_improver.v1"
CURRENT_MODEL = "current_model"
DEFAULT_RECIPE = Path(__file__).resolve().parents[1] / "manifests/improver/recipe.v1.yaml"


class ImproverError(ValueError):
    """The labelled data cannot support building or evaluating candidates."""


def load_recipe(path: Path | None = None) -> dict[str, Any]:
    recipe = yaml.safe_load((path or DEFAULT_RECIPE).read_text(encoding="utf-8"))
    for key in ("recipe_id", "candidates", "promotion_rule", "minimum_labels_per_class"):
        if key not in recipe:
            raise ImproverError(f"Improver recipe is missing '{key}'.")
    return recipe


def build_estimator(spec: dict[str, Any]):
    family, params = spec["family"], dict(spec.get("parameters", {}))
    if family == "multinomial_logistic_regression":
        return Pipeline(
            [("scale", StandardScaler()), ("model", LogisticRegression(solver="lbfgs", **params))]
        )
    if family == "random_forest":
        return RandomForestClassifier(n_jobs=-1, **params)
    if family == "histogram_gradient_boosting":
        return HistGradientBoostingClassifier(**params)
    raise ImproverError(f"Unknown candidate family: {family}")


def review_labels(session: ReviewSession, classes: list[str]) -> tuple[pd.DataFrame, dict]:
    """Usable training labels from review decisions, and counts of what was left out."""
    rows = []
    skipped = {"excluded": 0, "uncertain_or_unresolved": 0, "outside_model_classes": 0}
    for decision in session.object_decisions:
        status = decision.review_status
        reviewed = normalize_review_label(decision.reviewed_fiber_type)
        if status is ObjectReviewStatus.EXCLUDED or reviewed == "exclude":
            skipped["excluded"] += 1
            continue
        if status in (ObjectReviewStatus.UNCERTAIN, ObjectReviewStatus.UNRESOLVED) or (
            reviewed == "uncertain"
        ):
            skipped["uncertain_or_unresolved"] += 1
            continue
        model_call = normalize_review_label(decision.model_fiber_type)
        label = model_call if status is ObjectReviewStatus.ACCEPTED else reviewed
        if label not in classes:
            skipped["outside_model_classes"] += 1
            continue
        rows.append(
            {
                "image_id": decision.image_id,
                "fiber_id": int(decision.fiber_id),
                "label": label,
                "current_model_call": model_call,
                "label_source": "kept_model_call" if label == model_call else "corrected",
                "queue_source": decision.queue_source,
                "reviewer": decision.reviewer,
            }
        )
    columns = [
        "image_id",
        "fiber_id",
        "label",
        "current_model_call",
        "label_source",
        "queue_source",
        "reviewer",
    ]
    return pd.DataFrame(rows, columns=columns), skipped


def build_training_table(
    project: Project,
    labels: pd.DataFrame,
    features_by_image: dict[str, pd.DataFrame],
    feature_names: list[str],
) -> pd.DataFrame:
    """Join review labels with model features; one row per labelled fiber."""
    parts = []
    for image_id, group in labels.groupby("image_id", sort=True):
        features = features_by_image.get(str(image_id))
        if features is None:
            raise ImproverError(f"No feature table for labelled image {image_id!r}.")
        id_column = "fiber_id" if "fiber_id" in features.columns else "label"
        missing = [name for name in feature_names if name not in features.columns]
        if missing:
            raise ImproverError(f"{image_id}: feature table lacks {', '.join(missing)}.")
        wanted = features.loc[:, [id_column, *feature_names]].rename(
            columns={id_column: "fiber_id"}
        )
        joined = group.merge(wanted, on="fiber_id", how="left", validate="one_to_one")
        joined.insert(0, "mouse_id", project.image(str(image_id)).mouse_id)
        parts.append(joined)
    if not parts:
        raise ImproverError("There are no usable review decisions to learn from.")
    table = pd.concat(parts, ignore_index=True)
    values = table[feature_names].apply(pd.to_numeric, errors="coerce")
    bad = ~np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    if bad.any():
        raise ImproverError(
            f"{int(bad.sum())} labelled fibers have missing or non-finite features; "
            "they may no longer exist in the current masks."
        )
    table[feature_names] = values
    return table


def _mouse_metrics(truth: pd.Series, predicted: pd.Series, mice: pd.Series) -> pd.DataFrame:
    rows = []
    for mouse, index in mice.groupby(mice).groups.items():
        y, p = truth.loc[index], predicted.loc[index]
        present = sorted(set(y))
        rows.append(
            {
                "mouse_id": mouse,
                "n_labels": len(y),
                "macro_f1": float(f1_score(y, p, labels=present, average="macro", zero_division=0)),
                "accuracy": float((y == p).mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate(
    table: pd.DataFrame, feature_names: list[str], classes: list[str], recipe: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Score every candidate on held-out mice beside the current model's calls."""
    mice = table["mouse_id"].astype(str)
    n_mice = mice.nunique()
    if n_mice < 2:
        raise ImproverError(
            "Labels come from one mouse, so candidates cannot be tested on a held-out mouse. "
            "Review fibers from at least two more mice."
        )
    X, y = table[feature_names], table["label"].astype(str)
    predictions = table[["mouse_id", "image_id", "fiber_id", "label"]].copy()
    predictions[CURRENT_MODEL] = table["current_model_call"].astype(str)
    for name, spec in recipe["candidates"].items():
        held_out = pd.Series(index=table.index, dtype=object)
        for train, test in LeaveOneGroupOut().split(X, y, groups=mice):
            if y.iloc[train].nunique() < 2:
                raise ImproverError(
                    "A training fold has only one class; label more fibers of other classes."
                )
            model = build_estimator(spec).fit(X.iloc[train], y.iloc[train])
            held_out.iloc[test] = model.predict(X.iloc[test])
        predictions[name] = held_out.astype(str)

    by_mouse_parts, summary = [], {}
    for name in [CURRENT_MODEL, *recipe["candidates"]]:
        per_mouse = _mouse_metrics(y, predictions[name], mice)
        per_mouse.insert(0, "model", name)
        by_mouse_parts.append(per_mouse)
        recall = recall_score(y, predictions[name], labels=classes, average=None, zero_division=0)
        summary[name] = {
            "median_mouse_macro_f1": float(per_mouse["macro_f1"].median()),
            "balanced_accuracy": float(balanced_accuracy_score(y, predictions[name])),
            "accuracy": float((y == predictions[name]).mean()),
            "recall_by_class": dict(zip(classes, map(float, recall), strict=True)),
        }
    return predictions, pd.concat(by_mouse_parts, ignore_index=True), summary


def decide(
    by_mouse: pd.DataFrame, summary: dict[str, Any], recipe: dict[str, Any]
) -> dict[str, Any]:
    """Apply the declared promotion rule; returns the recommendation and each candidate's checks."""
    rule = recipe["promotion_rule"]
    wide = by_mouse.pivot(index="mouse_id", columns="model", values="macro_f1")
    n_mice = len(wide)
    checks: dict[str, Any] = {}
    passing: list[tuple[float, str]] = []
    for name in recipe["candidates"]:
        delta = wide[name] - wide[CURRENT_MODEL]
        recall_drop = max(
            summary[CURRENT_MODEL]["recall_by_class"][c] - summary[name]["recall_by_class"][c]
            for c in summary[name]["recall_by_class"]
        )
        result = {
            "median_mouse_macro_f1_gain": float(delta.median()),
            "fraction_of_mice_not_worse": float((delta >= 0).mean()),
            "largest_class_recall_drop": float(recall_drop),
            "enough_mice": n_mice >= int(rule["minimum_mice"]),
        }
        result["passes"] = bool(
            result["enough_mice"]
            and result["median_mouse_macro_f1_gain"]
            >= float(rule["minimum_median_mouse_macro_f1_gain"])
            and result["fraction_of_mice_not_worse"]
            >= float(rule["minimum_fraction_of_mice_not_worse"])
            and result["largest_class_recall_drop"] <= float(rule["maximum_class_recall_drop"])
        )
        checks[name] = result
        if result["passes"]:
            passing.append((result["median_mouse_macro_f1_gain"], name))
    recommended = max(passing)[1] if passing else None
    if n_mice < int(rule["minimum_mice"]):
        reason = (
            f"Only {n_mice} mice have labels; the rule needs {rule['minimum_mice']} before "
            "recommending a change."
        )
    elif recommended:
        reason = f"{recommended} met every condition of the declared promotion rule."
    else:
        reason = "No candidate met every condition of the declared promotion rule."
    return {
        "recommended": recommended,
        "keep_current_model": recommended is None,
        "reason": reason,
        "n_mice": n_mice,
        "checks": checks,
    }


def improve(
    project: Project,
    session: ReviewSession,
    features_by_image: dict[str, pd.DataFrame],
    base_manifest: dict[str, Any],
    output_dir: Path,
    *,
    recipe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build, evaluate, and save candidate models; return the run summary."""
    recipe = recipe or load_recipe()
    classes = [str(value) for value in base_manifest["outputs"]]
    feature_names = [str(value) for value in base_manifest["features"]]
    labels, skipped = review_labels(session, classes)
    table = build_training_table(project, labels, features_by_image, feature_names)
    counts = table["label"].value_counts().reindex(classes, fill_value=0)
    minimum = int(recipe["minimum_labels_per_class"])
    short = [f"{name} ({int(count)})" for name, count in counts.items() if count < minimum]
    if short:
        raise ImproverError(
            f"Each class needs at least {minimum} reviewed fibers; too few for: {', '.join(short)}."
        )
    predictions, by_mouse, summary = evaluate(table, feature_names, classes, recipe)
    decision = decide(by_mouse, summary, recipe)

    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_dataframe(output_dir / "training_rows.csv", table)
    atomic_write_dataframe(output_dir / "held_out_predictions.csv", predictions)
    atomic_write_dataframe(output_dir / "metrics_by_mouse.csv", by_mouse)
    (output_dir / "recipe.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
    training_digest = file_sha256(output_dir / "training_rows.csv")
    stamp = output_dir.name
    candidates: dict[str, Any] = {}
    for name, spec in recipe["candidates"].items():
        folder = output_dir / "candidates" / name
        folder.mkdir(parents=True)
        model = build_estimator(spec).fit(table[feature_names], table["label"].astype(str))
        artifact = folder / f"{name}.joblib"
        joblib.dump(model, artifact)
        manifest = {
            **base_manifest,
            "model_id": f"{base_manifest['model_id']}__{stamp}__{name}",
            "artifact": artifact.name,
            "artifact_sha256": file_sha256(artifact),
            "training_data_sha256": training_digest,
            "parent_model_id": base_manifest["model_id"],
            "improver_recipe": recipe["recipe_id"],
            "intended_use": (
                f"Candidate adapted from {base_manifest['model_id']} using "
                f"{len(table)} review decisions. Not validated; see the improver summary."
            ),
        }
        (folder / f"{name}.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
        candidates[name] = {
            "artifact": str(artifact.relative_to(output_dir)),
            "manifest": str((folder / f"{name}.yaml").relative_to(output_dir)),
            "artifact_sha256": manifest["artifact_sha256"],
        }
    result = {
        "schema_version": IMPROVER_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "project_id": project.project_id,
        "current_model": base_manifest["model_id"],
        "recipe_id": recipe["recipe_id"],
        "n_labels": int(len(table)),
        "labels_by_class": {name: int(count) for name, count in counts.items()},
        "labels_by_source": table["label_source"].value_counts().to_dict(),
        "labels_left_out": skipped,
        "n_mice": int(table["mouse_id"].nunique()),
        "validation": "leave_one_mouse_out",
        "metrics": summary,
        "decision": decision,
        "candidates": candidates,
        "training_data_sha256": training_digest,
        "caveat": (
            "Labels come from review of the current model's calls, mostly on fibers it flagged, "
            "so they over-represent hard cases and kept calls favor the current model. Treat "
            "these numbers as a comparison between models on reviewed fibers, not as accuracy."
        ),
    }
    (output_dir / "improver_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
