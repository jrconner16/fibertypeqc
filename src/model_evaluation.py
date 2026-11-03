"""Independent test evaluation of one frozen built model (no GUI imports).

A candidate from ``build`` is chosen explicitly and scored on the blind labels of the project's
test mice, using the measurements the run without a model already wrote (nothing is re-typed and
the project's model is not changed). The candidate file, the test labels, and the test sections'
masks and measurements are fingerprinted so a later promotion can check that the evaluation still
describes the same model and inputs. Every evaluation is logged, so repeated use of the same test
mice stays visible.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from fibertypeqc.artifacts import file_sha256, git_commit
from fibertypeqc.model_manifest import load_model_manifest
from fibertypeqc.semantic_model import predict_semantic_candidate
from src.model_builder import (
    blind_label_sessions,
    combine_labels,
    measurement_settings,
    mouse_metrics,
)
from src.review.project import Project
from src.review.roles import TEST, allowed_images
from src.review.session import ReviewSession
from src.review.storage import atomic_write_dataframe

EVALUATION_SCHEMA_VERSION = "fibertypeqc.model_evaluation.v1"
HISTORY_NAME = "evaluation_history.csv"
HISTORY_COLUMNS = (
    "evaluated_at_utc",
    "model_id",
    "artifact_sha256",
    "build_run",
    "recommended_by_builder",
    "test_mice",
    "n_test_labels",
    "test_set_uses",
    "inputs_sha256",
    "evaluation",
)
CAVEAT = (
    "One independent check on held-out test mice for a model chosen without them. It is not a "
    "guarantee of accuracy on other data, and each further evaluation on the same test mice "
    "weakens their independence."
)


class EvaluationError(ValueError):
    """The candidate or the test labels cannot support an independent evaluation."""


def find_build_candidate(root: Path, candidate: str) -> Path:
    """A built candidate given as its model file, or by name from the latest build run."""
    path = Path(candidate).expanduser()
    if path.is_file():
        return path.resolve()
    for run in sorted((root / "models").glob("build_*"), reverse=True):
        found = run / "candidates" / candidate / f"{candidate}.joblib"
        if found.is_file():
            return found.resolve()
    raise EvaluationError(
        f"No built candidate named {candidate!r} in {root / 'models'}. Run 'build' first, or "
        "pass the candidate's .joblib file."
    )


def build_recommended(artifact: Path) -> bool | None:
    """Whether the build run that made this candidate recommended it (None if unknown)."""
    summary = artifact.parents[2] / "build_summary.json"
    if not summary.is_file():
        return None
    decision = json.loads(summary.read_text())["selection"]["decision"]
    return decision.get("recommended") == artifact.stem


def test_images(project: Project) -> list[str]:
    images = allowed_images(project, TEST)
    if not images:
        raise EvaluationError(
            "This project declares no test mice (evaluation_roles.csv with role 'test'), so "
            "there is nothing independent to evaluate on."
        )
    return sorted(images)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def evaluation_inputs(
    project: Project, sessions: dict[str, ReviewSession], artifact: Path
) -> dict[str, Any]:
    """Fingerprints of everything an evaluation depends on.

    The candidate file and manifest, every reviewer's saved decisions on test mice, and each test
    section's mask and measurements. Any change makes an earlier evaluation stale.
    """
    images = test_images(project)
    decisions = sorted(
        (
            reviewer,
            decision.image_id,
            int(decision.fiber_id),
            str(decision.reviewed_fiber_type),
            decision.review_status.value,
        )
        for reviewer, session in sessions.items()
        for decision in session.object_decisions
        if decision.image_id in images
    )
    sections = {}
    for image_id in images:
        image = project.image(image_id)
        sections[image_id] = {
            "fiber_labels": file_sha256(image.outputs["fiber_labels"]),
            "measurements": file_sha256(
                image.prediction_directory / f"{image_id}_feature_diagnostics.csv"
            ),
        }
    return {
        "artifact_sha256": file_sha256(artifact),
        "manifest_sha256": file_sha256(artifact.with_suffix(".yaml")),
        "test_labels_sha256": _digest(decisions),
        "test_sections": sections,
    }


def _history(root: Path) -> pd.DataFrame:
    path = root / "models" / HISTORY_NAME
    if not path.is_file():
        return pd.DataFrame(columns=list(HISTORY_COLUMNS))
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def evaluation_status(root: Path, project: Project, artifact: Path) -> dict[str, Any]:
    """Whether the latest evaluation of this exact candidate still matches its inputs.

    Returns ``status`` ``current``, ``stale``, ``missing``, or ``no_test_mice``.
    """
    if not allowed_images(project, TEST):
        return {"status": "no_test_mice", "evaluation": ""}
    sha = file_sha256(artifact)
    rows = _history(root)
    rows = rows[rows["artifact_sha256"] == sha]
    if rows.empty:
        return {"status": "missing", "evaluation": ""}
    last = rows.iloc[-1]
    current = _digest(evaluation_inputs(project, blind_label_sessions(project), artifact))
    status = "current" if current == last["inputs_sha256"] else "stale"
    return {"status": status, "evaluation": last["evaluation"]}


def evaluate(root: Path, project: Project, candidate: str) -> dict[str, Any]:
    """Score one explicitly chosen candidate on the test mice and log the evaluation."""
    if project.model_version != "none":
        raise EvaluationError(
            "Independent evaluation is for built models in a project with 'model: none', "
            "before a model is promoted."
        )
    artifact = find_build_candidate(root, candidate)
    manifest_path = artifact.with_suffix(".yaml")
    if not manifest_path.is_file():
        raise EvaluationError(f"{artifact.name} has no manifest beside it ({manifest_path.name}).")
    manifest = load_model_manifest(manifest_path)
    if file_sha256(artifact) != yaml.safe_load(manifest_path.read_text())["artifact_sha256"]:
        raise EvaluationError(f"{artifact.name} does not match its manifest; it has changed.")
    classes = [str(name) for name in manifest.outputs]
    images = test_images(project)
    settings, _ = measurement_settings(project)
    if settings != manifest.feature_extraction:
        raise EvaluationError(
            "The test sections were measured with different settings from the ones this model "
            "was trained on."
        )
    sessions = blind_label_sessions(project)
    labels, audit = combine_labels(project, sessions, classes, role=TEST)
    if labels.empty:
        raise EvaluationError(
            "No usable blind labels on test mice. Label test mice with review --blind first."
        )

    parts = []
    for image_id, group in labels.groupby("image_id", sort=True):
        image = project.image(str(image_id))
        diagnostics = pd.read_csv(
            image.prediction_directory / f"{image_id}_feature_diagnostics.csv", low_memory=False
        )
        predicted = predict_semantic_candidate(diagnostics, artifact, manifest)
        calls = predicted[["label", "model_prediction"]].rename(
            columns={"label": "fiber_id", "model_prediction": "predicted"}
        )
        parts.append(group.merge(calls, on="fiber_id", how="left", validate="one_to_one"))
    table = pd.concat(parts, ignore_index=True)
    if table["predicted"].isna().any():
        raise EvaluationError("Some labelled test fibers are missing from the measurements.")
    truth, predicted = table["label"].astype(str), table["predicted"].astype(str)
    by_mouse = mouse_metrics(truth, predicted, table["mouse_id"].astype(str), classes)
    pooled = {
        name: (
            float(((truth == name) & (predicted == name)).sum() / (truth == name).sum())
            if (truth == name).any()
            else None
        )
        for name in classes
    }
    confusion = pd.crosstab(truth, predicted).reindex(index=classes, columns=classes, fill_value=0)
    confusion.index.name, confusion.columns.name = "labelled", "predicted"

    inputs = evaluation_inputs(project, sessions, artifact)
    history = _history(root)
    test_mice = sorted({project.image(image_id).mouse_id for image_id in images})
    earlier = history[history["test_mice"] == ";".join(test_mice)]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    folder = root / "models" / "evaluations" / f"{stamp}__{manifest.model_id}"
    for repeat in range(2, 100):  # two evaluations within the same second
        if not folder.exists():
            break
        folder = folder.with_name(f"{stamp}_{repeat}__{manifest.model_id}")
    folder.mkdir(parents=True, exist_ok=False)
    atomic_write_dataframe(folder / "test_predictions.csv", table)
    atomic_write_dataframe(folder / "metrics_by_mouse.csv", by_mouse)
    atomic_write_dataframe(folder / "confusion_matrix.csv", confusion.reset_index())
    result = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "model_id": manifest.model_id,
        "artifact": str(artifact),
        "build_run": artifact.parents[2].name,
        "recommended_by_builder": build_recommended(artifact),
        "classes": classes,
        "test_mice": test_mice,
        "labels": {"n_fibers": int(len(table)), **audit},
        "pooled_recall_by_class": pooled,
        "median_mouse_macro_f1": float(by_mouse["macro_f1"].median()),
        "test_set_uses": int(len(earlier)) + 1,
        "inputs": inputs,
        "caveat": CAVEAT,
    }
    (folder / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    row = {
        "evaluated_at_utc": result["evaluated_at_utc"],
        "model_id": manifest.model_id,
        "artifact_sha256": inputs["artifact_sha256"],
        "build_run": result["build_run"],
        "recommended_by_builder": result["recommended_by_builder"],
        "test_mice": ";".join(test_mice),
        "n_test_labels": len(table),
        "test_set_uses": result["test_set_uses"],
        "inputs_sha256": _digest(inputs),
        "evaluation": str(folder.relative_to(root)),
    }
    history = pd.concat([history, pd.DataFrame([{k: str(v) for k, v in row.items()}])])
    history.to_csv(root / "models" / HISTORY_NAME, index=False, columns=list(HISTORY_COLUMNS))
    result["folder"] = str(folder)
    return result
