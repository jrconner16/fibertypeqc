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
import yaml

from fibertypeqc.config import load_channel_config
from fibertypeqc.model_resolution import default_model_id, resolve_model_argument
from fibertypeqc.panel_presets import write_panel
from src.cli import (
    CONFIG_NAME,
    ProjectError,
    ProjectFolder,
    check_sample_sheet,
    image_channel_count,
    image_files,
    next_step,
    project_images,
    project_status,
    sample_rows,
)
from src.make_review_project import USE_COLUMN, used_rows

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
    # For Project set up when it blocks Run: "panel" or "model", whichever needs fixing.
    fix: str = ""


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
                try:
                    sheet = used_rows(sheet)
                except ValueError:
                    return False, note, note
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
    used = used_rows(sheet)
    mice = used["mouse_id"].str.strip().nunique()
    summary = f"{plural(len(used), 'image', 'images')} · {plural(mice, 'mouse', 'mice')}"
    if len(used) < len(sheet):
        summary += f" · {len(sheet) - len(used)} not used"
    return True, summary, ""


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
    # Run would stop, or fail on every image; say why here instead, before anything starts.
    fix = ""
    problem = panel_problem(project) if routed == "run" else ""
    if problem:
        current, command, explanation, fix = "setup", "", problem, "panel"
    problem = "" if problem or routed != "run" else model_problem(project)
    if problem:
        current, command, fix = "setup", "", "model"
        explanation = friendly_model_problem(problem)
    samples_ok, samples_summary, samples_note = _samples(project)
    if not samples_ok and current == routed and routed not in (None, "run"):
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
                fix=fix if key == "setup" else "",
            )
        )
    notes = tuple(note for note in status["notes"] if note != samples_note)
    return Workflow(steps=tuple(steps), status=status, finished=finished, notes=notes)


def friendly_model_problem(problem: str) -> str:
    """The run_batch message about a model, as the launcher says it."""
    missing = problem.removeprefix("--model file not found: ")
    if missing != problem:
        return (
            f"The model file is not where the project expects it:\n{missing}\n"
            "Choose the model file, or put it there and press Refresh."
        )
    private = re.match(r"Model '([^']+)' is distributed privately", problem)
    if private:
        return (
            f"The model {private.group(1)} is downloaded separately and is not on this computer "
            f"yet. Download {private.group(1)}.joblib from the FiberTypeQC releases page, then "
            "choose it here."
        )
    return f"The model cannot be used: {problem}\nChoose a model file."


def model_problem(project: ProjectFolder) -> str:
    """Why ``run`` could not use the project's model (checked as run_batch does), or ""."""
    if project.no_model:
        return ""
    try:
        resolve_model_argument(project.model or default_model_id("fiber_identity") or "")
    except ValueError as exc:
        return str(exc)
    return ""


def set_config_value(project: ProjectFolder, key: str, value: str) -> None:
    """Set one top-level config value, keeping the rest of the file and its comments."""
    config = project.root / CONFIG_NAME
    line = f"{key}: {json.dumps(value)}"
    text = config.read_text(encoding="utf-8")
    pattern = rf"^{re.escape(key)}:.*$"
    updated, count = re.subn(pattern, lambda _match: line, text, flags=re.MULTILINE)
    if not count:
        updated = text.rstrip("\n") + f"\n{line}\n"
    config.write_text(updated, encoding="utf-8")


def set_model(project: ProjectFolder, model: str) -> None:
    """Point the project at another model."""
    set_config_value(project, "model", model)


def can_edit_setup(project: ProjectFolder) -> bool:
    """Panel, images, and model may change until an image has been segmented successfully."""
    summary = project.batch_dir / "batch_summary.csv"
    if not summary.is_file():
        return True
    table = pd.read_csv(summary, dtype=str, keep_default_na=False)
    return not table["status"].eq("success").any()


STAIN_NAMES = {
    "laminin": "laminin",
    "dapi": "DAPI",
    "type_i": "Type I",
    "type_iia": "Type IIa",
    "type_iib": "Type IIb",
    "type_iix": "Type IIx",
    "emhc": "eMHC",
}


def channels_beyond(panel: dict, channel_count: int | None) -> list[str]:
    """Stains the panel puts in a channel the images do not have, e.g. "Type IIb in channel 3"."""
    if not channel_count:
        return []
    return [
        f"{STAIN_NAMES.get(name, name)} in channel {channel}"
        for name, channel in (panel.get("channels") or {}).items()
        if isinstance(channel, int) and channel >= channel_count
    ]


def panel_problem(project: ProjectFolder) -> str:
    """Why the panel does not fit the images (from the first image's metadata), or ""."""
    try:
        images = project_images(project)
        panel = yaml.safe_load(project.panel.read_text(encoding="utf-8")) or {}
    except (ProjectError, OSError, yaml.YAMLError):
        return ""  # status already notes a missing images folder or panel file
    count = image_channel_count(images[0])
    beyond = channels_beyond(panel, count)
    if not beyond:
        return ""
    return (
        f"The stain panel does not fit these images: it expects {', '.join(beyond)}, but "
        f"{images[0].name} has {count} channels (numbered 0 to {count - 1}). Use Edit to "
        "choose the right panel or channels."
    )


def update_setup(
    project: ProjectFolder, images_dir: Path, panel: dict, panel_source: str, model: str
) -> None:
    """Change a project's images folder, panel, and model before anything has run.

    The sample sheet keeps each image's mouse ID and group columns; images new to the folder get
    blank rows (``use = yes`` when the sheet has that column), and images no longer in it are
    dropped. The project's include and exclude patterns apply to the new folder too.
    """
    if not can_edit_setup(project):
        raise ProjectError(
            "Segmentation has already run with these settings. Create a new project to use "
            "another panel or images."
        )
    images = image_files(images_dir, project.include, project.exclude)
    rows = pd.DataFrame(sample_rows(images, None, images_dir if project.include else None))
    if project.sample_sheet.is_file():
        old = pd.read_csv(project.sample_sheet, dtype=str, keep_default_na=False)
        kept = [name for name in old.columns if name not in ("raw_image_path",)]
        rows = rows[["image_id", "raw_image_path"]].merge(old[kept], on="image_id", how="left")
        rows = rows.fillna("")
        if USE_COLUMN in rows:
            rows.loc[rows[USE_COLUMN].eq(""), USE_COLUMN] = "yes"
        rows = rows[[name for name in old.columns if name in rows.columns]]
    write_panel(project.root / "panel.yaml", panel, f"Panel for this project ({panel_source}).")
    rows.to_csv(project.sample_sheet, index=False)
    set_config_value(project, "images", str(images_dir))
    set_config_value(project, "panel", "panel.yaml")
    set_model(project, model)


def command_arguments(command: str, project: ProjectFolder, reviewer: str = "") -> list[str]:
    """CLI arguments for a step command, with the placeholders filled in."""
    filled = {"PROJECT": str(project.root), "NAME": reviewer}
    return [filled.get(word, word) for word in command.split()]
