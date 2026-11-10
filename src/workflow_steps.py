"""The project's workflow as a list of steps, for the launcher (no GUI imports).

Built from ``project_status`` and ``next_step`` in ``src.cli``, so the launcher shows exactly the
step ``status`` would print. Only the steps on this project's path are listed: a project with a
model goes run -> prepare -> review -> finalize; a project with ``model: none`` goes run ->
prepare -> label -> build -> evaluate -> promote, then continues as a project with a model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fibertypeqc.config import load_channel_config
from fibertypeqc.model_resolution import default_model_id, resolve_model_argument
from src.cli import (
    CONFIG_NAME,
    ProjectError,
    ProjectFolder,
    check_sample_sheet,
    next_step,
    project_status,
)

DONE, CURRENT, UPCOMING, ATTENTION, SKIPPED = "done", "current", "upcoming", "attention", "skipped"

MODEL_PATH = ("setup", "samples", "run", "prepare", "review", "finalize")
BUILDER_PATH = ("setup", "samples", "run", "prepare", "label", "build", "evaluate", "promote")

TITLES = {
    "setup": "Project set up",
    "samples": "Samples",
    "run": "Segment and type",
    "prepare": "Prepare review",
    "review": "Review",
    "finalize": "Finalize and report",
    "label": "Label fibers (blind)",
    "build": "Build a model",
    "evaluate": "Check on test mice",
    "promote": "Switch to the new model",
}
COMMANDS = {
    "run": "run PROJECT",
    "prepare": "prepare PROJECT",
    "review": "review PROJECT",
    "finalize": "finalize PROJECT",
    "label": "review PROJECT --blind --reviewer NAME",
    "build": "build PROJECT",
}


@dataclass(frozen=True)
class Step:
    key: str
    title: str
    state: str
    summary: str = ""
    # CLI arguments with PROJECT (and NAME for the reviewer) as placeholders; "" if none.
    command: str = ""
    explanation: str = ""


@dataclass(frozen=True)
class Workflow:
    steps: tuple[Step, ...]
    status: dict[str, Any]
    finished: bool
    # Status notes other than the sample-sheet problem, which the Samples step explains.
    notes: tuple[str, ...] = ()

    @property
    def current(self) -> Step | None:
        return next((step for step in self.steps if step.state == CURRENT), None)

    def step(self, key: str) -> Step | None:
        return next((step for step in self.steps if step.key == key), None)


def _step_key(command: str) -> str:
    words = command.split()
    if words[0] == "review" and "--blind" in words:
        return "label"
    return words[0]


def _model_name(project: ProjectFolder) -> str:
    if project.no_model:
        return "none (build one from your labels)"
    if not project.model:
        return "default"
    return Path(project.model).stem if "/" in project.model else project.model


def plural(count: int, one: str, many: str) -> str:
    return f"{count} {one if count == 1 else many}"


def _samples(project: ProjectFolder) -> tuple[bool, str, str]:
    """(usable, one-line summary, the status note it replaces) for the sample sheet."""
    try:
        check_sample_sheet(project)
    except ProjectError as exc:
        note = str(exc)
        if project.sample_sheet.is_file():
            sheet = pd.read_csv(project.sample_sheet, dtype=str, keep_default_na=False)
            if "mouse_id" in sheet:
                blank = sheet.loc[sheet["mouse_id"].str.strip().eq(""), "image_id"].tolist()
                if blank:
                    shown = ", ".join(blank[:3]) + (" …" if len(blank) > 3 else "")
                    return (
                        False,
                        (
                            f"{plural(len(blank), 'image still needs', 'images still need')} a "
                            f"mouse ID ({shown}). Use Edit samples."
                        ),
                        note,
                    )
        return False, note, note
    sheet = pd.read_csv(project.sample_sheet, dtype=str, keep_default_na=False)
    mice = sheet["mouse_id"].str.strip().nunique()
    return True, f"{plural(len(sheet), 'image', 'images')} · {plural(mice, 'mouse', 'mice')}", ""


def _panel_name(project: ProjectFolder) -> str:
    try:
        markers = load_channel_config(project.panel).available_markers
    except (OSError, ValueError):
        return project.panel.name
    names = {"i": "I", "iia": "IIa", "iib": "IIb", "iix": "IIx", "emhc": "eMHC"}
    return "laminin + " + ", ".join(names.get(marker, marker) for marker in markers)


def _summaries(project: ProjectFolder, status: dict[str, Any]) -> dict[str, str]:
    found = {"setup": f"stains {_panel_name(project)} · model {_model_name(project)}"}
    if "sections" in status:
        failed = f", {status['failed']} failed" if status["failed"] else ""
        found["run"] = f"{status['sections']} section(s) succeeded{failed}"
    if "project_sections" in status and status["stage"] != "project_built":
        found["prepare"] = f"{status['project_sections']} section(s) ready"
    if "fiber_decisions" in status:
        found["review"] = f"{status['fiber_decisions']} fiber decision(s)"
    if status.get("blind_labels"):
        found["label"] = ", ".join(
            f"{name}: {count}" for name, count in status["blind_labels"].items()
        )
    if "build" in status:
        found["build"] = f"{status['build']}: recommended {status['recommended'] or 'none'}"
    if status.get("evaluation"):
        found["evaluate"] = {
            "current": "evaluated on the test mice",
            "stale": "evaluation out of date",
            "missing": "not evaluated yet",
            "no_test_mice": "no test mice in this project, so no independent check",
        }[status["evaluation"]]
    if "report" in status:
        found["finalize"] = "report ready"
    return found


def workflow(project: ProjectFolder) -> Workflow:
    """The steps for this project, with the next one marked current."""
    status = project_status(project)
    command, explanation = next_step(project, status)
    path = BUILDER_PATH if project.no_model else MODEL_PATH
    finished = status["stage"] == "reported"
    routed = None if finished else _step_key(command)
    # Steps before the routed one are done, even when the sample sheet takes over as current.
    position = path.index(routed) if routed in path else len(path)
    current = routed
    problem = model_problem(project) if routed == "run" else ""
    if problem:
        # Run would stop at once; say why here instead, before anything starts.
        current, command = "setup", ""
        missing = problem.removeprefix("--model file not found: ")
        explanation = (
            f"The model file is not where the project expects it:\n{missing}\n"
            if missing != problem
            else f"The model cannot be used: {problem}\n"
        ) + "Choose the model file, or put it there and press Refresh."
    samples_ok, samples_summary, samples_note = _samples(project)
    if not samples_ok and current not in (None, "run"):
        # Everything after segmentation needs a usable sample sheet.
        current, explanation, command = "samples", "", ""
    summaries = _summaries(project, status)
    summaries["samples"] = samples_summary

    steps = []
    for index, key in enumerate(path):
        if key == "setup":
            state = CURRENT if current == "setup" else DONE
        elif key == "samples":
            state = CURRENT if current == "samples" else (DONE if samples_ok else ATTENTION)
        elif key == current:
            state = CURRENT
        elif key == "evaluate" and status.get("evaluation") == "no_test_mice":
            state = SKIPPED
        else:
            state = DONE if index < position else UPCOMING
        step_command = command if state == CURRENT and key == current else COMMANDS.get(key, "")
        if not step_command and key in ("evaluate", "promote") and status.get("recommended"):
            step_command = f"{key} PROJECT {status['recommended']}"
        steps.append(
            Step(
                key=key,
                title="Segment and measure" if key == "run" and project.no_model else TITLES[key],
                state=state,
                summary=summaries.get(key, ""),
                command=step_command,
                explanation=explanation if state == CURRENT else "",
            )
        )
    notes = tuple(note for note in status["notes"] if note != samples_note)
    return Workflow(steps=tuple(steps), status=status, finished=finished, notes=notes)


def model_problem(project: ProjectFolder) -> str:
    """Why ``run`` could not use the project's model (checked as run_batch does), or ""."""
    if project.no_model:
        return ""
    try:
        resolve_model_argument(project.model or default_model_id("fiber_identity") or "")
    except ValueError as exc:
        return str(exc)
    return ""


def set_model(project: ProjectFolder, model: str) -> None:
    """Point the project at another model, keeping the rest of the config file as written."""
    config = project.root / CONFIG_NAME
    line = f"model: {json.dumps(model)}"
    text = config.read_text(encoding="utf-8")
    updated, count = re.subn(r"^model:.*$", lambda _match: line, text, flags=re.MULTILINE)
    if not count:
        updated = text.rstrip("\n") + f"\n{line}\n"
    config.write_text(updated, encoding="utf-8")


def command_arguments(command: str, project: ProjectFolder, reviewer: str = "") -> list[str]:
    """CLI arguments for a step command, with the placeholders filled in."""
    filled = {"PROJECT": str(project.root), "NAME": reviewer}
    return [filled.get(word, word) for word in command.split()]
