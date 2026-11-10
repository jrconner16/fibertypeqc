"""One command for a FiberTypeQC project folder.

A project folder holds one small config file and everything the workflow writes::

    myproject/
      fibertypeqc_project.yaml   images folder, panel, model, sample sheet
      panel.yaml  samples.csv
      batch/    run outputs (masks, fiber tables, per-image QC)
      review/   review project, QC tables, saved review decisions
      final/    finalized fiber tables
      results/  summary tables and cohort_report.html

Usage::

    python -m fibertypeqc init     myproject/ --images /path/to/images
    python -m fibertypeqc status   myproject/
    python -m fibertypeqc run      myproject/
    python -m fibertypeqc prepare  myproject/
    python -m fibertypeqc review   myproject/
    python -m fibertypeqc finalize myproject/
    python -m fibertypeqc app      myproject/   (the same steps in a window)

Each step calls the corresponding ``scripts.*`` command with paths taken from the project folder.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from fibertypeqc.config import load_channel_config
from fibertypeqc.model_resolution import default_model_id
from fibertypeqc.panel_presets import (
    MARKER_LABELS,
    MARKERS,
    channel_label,
    channel_range,
    list_presets,
    panel_from_channels,
    preset_path,
    write_panel,
)
from src.fiber_type_labels import default_classes, negative_display_label, validate_classes
from src.make_review_project import USE_COLUMN, used_rows
from src.reference_snapshot import (
    DEFAULT_COMPOSITION_TOLERANCE,
    DEFAULT_COUNT_TOLERANCE,
)
from src.reference_snapshot import check as check_snapshot
from src.reference_snapshot import freeze as freeze_snapshot

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "fibertypeqc_project.yaml"
CONFIG_SCHEMA = "fibertypeqc_project.v1"


class ProjectError(ValueError):
    """The project folder or its config is not usable as given."""


@dataclass(frozen=True)
class ProjectFolder:
    root: Path
    images: Path
    panel: Path
    sample_sheet: Path
    model: str | None
    split_czi_scenes: bool
    reviewer: str
    display_downsample: int
    exclude_image_border_fibers: bool = False
    # ``model: none``: fibers are measured but not typed, and are labelled into ``classes``.
    no_model: bool = False
    classes: tuple[str, ...] = ()
    negative_label: str = ""
    # Glob patterns (relative to ``images``) for images in subfolders; empty: files directly
    # inside ``images``, as before patterns existed.
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    @property
    def batch_dir(self) -> Path:
        return self.root / "batch"

    @property
    def review_dir(self) -> Path:
        return self.root / "review"

    @property
    def review_project(self) -> Path:
        return self.review_dir / "project.yaml"

    @property
    def qc_dir(self) -> Path:
        return self.review_dir / "qc"

    @property
    def review_state(self) -> Path:
        return self.review_dir / "review" / "review_state.json"

    @property
    def final_dir(self) -> Path:
        return self.root / "final"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"


def _resolve(root: Path, value: Any, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ProjectError(f"{CONFIG_NAME}: '{key}' must be a path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _patterns(raw: dict, key: str) -> tuple[str, ...]:
    value = raw.get(key) or []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ProjectError(f"{CONFIG_NAME}: '{key}' must be a list of patterns, e.g. ['**/*.czi']")
    return tuple(value)


def load_project_folder(folder: Path) -> ProjectFolder:
    root = folder.expanduser().resolve()
    config_path = root / CONFIG_NAME
    if not config_path.is_file():
        raise ProjectError(
            f"{root} is not a FiberTypeQC project folder (no {CONFIG_NAME}). "
            "Create one with: python -m fibertypeqc init"
        )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != CONFIG_SCHEMA:
        raise ProjectError(f"{config_path} must set schema_version: {CONFIG_SCHEMA}")
    model = raw.get("model")
    if model is not None and not isinstance(model, str):
        raise ProjectError(f"{CONFIG_NAME}: 'model' must be a model ID or a file path")
    no_model = isinstance(model, str) and model.strip().lower() == "none"
    if no_model:
        model = None
    elif isinstance(model, str) and _is_path_like(model):
        model = str(_resolve(root, model, "model"))
    panel = preset_path(str(raw.get("panel"))) or _resolve(root, raw.get("panel"), "panel")
    classes: tuple[str, ...] = ()
    negative_label = ""
    if "classes" in raw and not no_model:
        raise ProjectError(
            f"{CONFIG_NAME}: 'classes' is only used with 'model: none'; "
            "a model sets its own classes"
        )
    if no_model and panel.is_file():
        try:
            markers = load_channel_config(panel).available_markers
            classes = tuple(
                validate_classes(raw["classes"], markers)
                if "classes" in raw
                else default_classes(markers)
            )
        except ValueError as exc:
            raise ProjectError(f"{CONFIG_NAME}: {exc}") from exc
        negative_label = negative_display_label(markers)
    return ProjectFolder(
        root=root,
        images=_resolve(root, raw.get("images"), "images"),
        # A preset name may be used directly instead of a panel file in the project folder.
        panel=panel,
        sample_sheet=_resolve(root, raw.get("sample_sheet"), "sample_sheet"),
        model=model,
        split_czi_scenes=bool(raw.get("split_czi_scenes", True)),
        reviewer=str(raw.get("reviewer", "") or ""),
        display_downsample=int(raw.get("display_downsample", 2)),
        exclude_image_border_fibers=bool(raw.get("exclude_image_border_fibers", False)),
        no_model=no_model,
        classes=classes,
        negative_label=negative_label,
        include=_patterns(raw, "include"),
        exclude=_patterns(raw, "exclude"),
    )


def _is_path_like(value: str) -> bool:
    return "/" in value or value.endswith((".joblib", ".pkl", ".pickle"))


def _run(module: str, arguments: list[str]) -> int:
    """Run one ``scripts.*`` command from the repository root."""
    command = [sys.executable, "-m", f"scripts.{module}", *arguments]
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


# --- status ---------------------------------------------------------------------------------


def project_status(project: ProjectFolder) -> dict[str, Any]:
    """Describe how far the project has progressed, without modifying anything."""
    status: dict[str, Any] = {"stage": "new", "notes": []}
    if not project.images.is_dir():
        status["notes"].append(f"images folder not found: {project.images}")
    if not project.panel.is_file():
        status["notes"].append(f"panel file not found: {project.panel}")
    summary_path = project.batch_dir / "batch_summary.csv"
    if not summary_path.is_file():
        return status
    summary = pd.read_csv(summary_path, dtype=str, keep_default_na=False)
    succeeded = int(summary["status"].eq("success").sum())
    status.update(stage="batch_done", sections=succeeded, failed=len(summary) - succeeded)
    if succeeded == 0:
        status["stage"] = "batch_failed"
        return status
    try:
        check_sample_sheet(project)
    except ProjectError as exc:
        status["notes"].append(str(exc))
    if not project.review_project.is_file():
        return status
    review = yaml.safe_load(project.review_project.read_text(encoding="utf-8"))
    status.update(stage="project_built", project_sections=len(review.get("images", [])))
    if not (project.qc_dir / "fiber_qc.csv").is_file():
        return status
    status["stage"] = "ready_for_review"
    if project.review_state.is_file():
        state = json.loads(project.review_state.read_text(encoding="utf-8"))
        status.update(
            fiber_decisions=len(state.get("object_decisions", [])),
            regions=len(state.get("regions", [])),
            sections_opened=len(state.get("input_fingerprints", {})),
        )
    if project.no_model:
        status["blind_labels"] = _blind_label_counts(project)
        split = mouse_split_summary(project)
        if split is not None:
            status["test_mice"], status["mice"] = split
            note = split_changed_note(project)
            if note:
                status["notes"].append(note)
        status.update(builder_status(project))
        return status
    manifest_path = project.final_dir / "finalization_manifest.json"
    if not manifest_path.is_file():
        return status
    status["stage"] = "finalized"
    if project.review_state.is_file() and (
        project.review_state.stat().st_mtime > manifest_path.stat().st_mtime
    ):
        status["stage"] = "finalize_outdated"
        return status
    report = project.results_dir / "cohort_report.html"
    if report.is_file() and report.stat().st_mtime >= manifest_path.stat().st_mtime:
        status.update(stage="reported", report=str(report))
    return status


def builder_status(project: ProjectFolder) -> dict[str, Any]:
    """The latest build and the independent evaluation of its recommended candidate (read only)."""
    builds = sorted((project.root / "models").glob("build_*/build_summary.json"))
    if not builds:
        return {}
    from src.model_evaluation import evaluation_status, find_build_candidate
    from src.review.project import load_project

    summary = json.loads(builds[-1].read_text(encoding="utf-8"))
    recommended = summary["selection"]["decision"].get("recommended")
    found: dict[str, Any] = {"build": builds[-1].parent.name, "recommended": recommended}
    if recommended:
        review = load_project(project.review_project)
        artifact = find_build_candidate(project.root, recommended)
        found["evaluation"] = evaluation_status(project.root, review, artifact)["status"]
        history = project.root / "models" / "evaluation_history.csv"
        if history.is_file():
            uses = pd.read_csv(history, dtype=str, keep_default_na=False)["test_set_uses"]
            found["test_set_uses"] = int(uses.astype(int).max()) if len(uses) else 0
    return found


def no_model_next_step(status: dict[str, Any]) -> tuple[str, str] | None:
    """The next builder step for a project without a model, as (command arguments, why)."""
    if status["stage"] != "ready_for_review":
        return None
    label = (
        "review PROJECT --blind --reviewer NAME",
        "Label a sample of fibers blind, in every mouse (the test mice too: the model never "
        "learns from them, it is checked on them).",
    )
    if not any(status.get("blind_labels", {}).values()):
        return label
    if "build" not in status:
        return ("build PROJECT", "Build candidate models from the blind labels of training mice.")
    chosen = status["recommended"]
    if not chosen:
        return (
            label[0],
            "No candidate passed the recipe in the latest build; label more fibers, then build "
            "again. (A candidate can still be evaluated and promoted with --override.)",
        )
    evaluation = status["evaluation"]
    if evaluation == "no_test_mice":
        return (
            f"promote PROJECT {chosen}",
            "No test mice are declared, so there is no independent test; promotion is recorded "
            "with a warning.",
        )
    if evaluation == "current":
        return (f"promote PROJECT {chosen}", "Switch the project to the evaluated candidate.")
    why = (
        "Its evaluation is out of date (labels, masks, or measurements changed); evaluating "
        "again uses the test mice again."
        if evaluation == "stale"
        else "Score the recommended candidate on the test mice before promoting it."
    )
    return (f"evaluate PROJECT {chosen}", why)


NEXT_STEP = {
    "new": ("run", "Segment and type every image (slow; use a GPU node or a batch job)."),
    "batch_failed": ("run", "No image succeeded; see batch/batch_run.log, fix, and run again."),
    "batch_done": ("prepare", "Build the review project and run QC."),
    "project_built": ("prepare", "Run QC."),
    "ready_for_review": ("review", "Review in Napari, then run: finalize."),
    "finalize_outdated": ("finalize", "Review changed since the last finalize; finalize again."),
    "finalized": ("finalize", "Build the result tables and report."),
    "reported": ("review", "Done. Review more and finalize again, or open the report."),
}
# A project without a model is measured, not typed; see no_model_next_step for later stages.
NO_MODEL_NEXT_STEP = {
    "new": ("run", "Segment and measure every image (no model: fibers are not typed)."),
}


def next_step(project: ProjectFolder, status: dict[str, Any]) -> tuple[str, str]:
    """The next command (arguments, with ``PROJECT`` for the folder) and why it comes next.

    Shared by ``status`` and the launcher so they always agree.
    """
    builder = no_model_next_step(status) if project.no_model else None
    if builder is not None:
        return builder
    step, explanation = (
        NO_MODEL_NEXT_STEP.get(status["stage"]) if project.no_model else None
    ) or NEXT_STEP[status["stage"]]
    return f"{step} PROJECT", explanation


def command_status(project: ProjectFolder) -> int:
    status = project_status(project)
    print(f"Project: {project.root}")
    print(f"Stage:   {status['stage'].replace('_', ' ')}")
    if "sections" in status:
        failed = f", {status['failed']} failed" if status["failed"] else ""
        print(f"Batch:   {status['sections']} section(s) succeeded{failed}")
    if "project_sections" in status:
        print(f"Review project: {status['project_sections']} section(s)")
    if "fiber_decisions" in status:
        print(
            f"Review:  {status['fiber_decisions']} fiber decision(s), {status['regions']} "
            f"region(s), {status['sections_opened']} section(s) opened"
        )
    if status.get("blind_labels"):
        counts = ", ".join(f"{name} {count}" for name, count in status["blind_labels"].items())
        print(f"Labels:  blind labels by reviewer: {counts}")
    if "test_mice" in status:
        print(f"Mice:    {mouse_split_message(status['test_mice'], status['mice'])}")
    if "report" in status:
        print(f"Report:  {status['report']}")
    for note in status["notes"]:
        print(f"Note:    {note}")
    if project.no_model:
        print(f"Model:   none; classes {', '.join(project.classes) or '(panel file missing)'}")
    if "build" in status:
        print(f"Build:   {status['build']}: recommended {status['recommended'] or 'none'}")
    if status.get("evaluation"):
        state = {
            "current": "evaluated on the test mice (current)",
            "stale": "evaluation out of date",
            "missing": "not evaluated on the test mice yet",
            "no_test_mice": "no test mice declared",
        }[status["evaluation"]]
        uses = status.get("test_set_uses")
        print(
            f"Test:    {status['recommended']}: {state}"
            + (f"; test mice used {uses} time(s)" if uses else "")
        )
    command, explanation = next_step(project, status)
    print(f"Next:    python -m fibertypeqc {command.replace('PROJECT', str(project.root))}")
    print(f"         {explanation}")
    return 0


# --- steps ----------------------------------------------------------------------------------


def runs_from_sample_sheet(project: ProjectFolder) -> bool:
    """Whether ``run`` takes its images from the sample sheet rather than the images folder.

    Projects with include patterns or a ``use`` column do; older flat projects keep scanning the
    folder, so their runs are unchanged.
    """
    if project.include:
        return True
    if not project.sample_sheet.is_file():
        return False
    return USE_COLUMN in pd.read_csv(project.sample_sheet, dtype=str, nrows=0).columns


def write_input_manifest(project: ProjectFolder) -> Path:
    """The ``use = yes`` rows of the sample sheet as a run_batch input manifest."""
    if not project.sample_sheet.is_file():
        raise ProjectError(f"Sample sheet not found: {project.sample_sheet}")
    sheet = pd.read_csv(project.sample_sheet, dtype=str, keep_default_na=False)
    missing = [name for name in ("image_id", "raw_image_path") if name not in sheet]
    if missing:
        raise ProjectError(f"{project.sample_sheet.name} is missing columns: {', '.join(missing)}")
    try:
        rows = used_rows(sheet)
    except ValueError as exc:
        raise ProjectError(f"{project.sample_sheet.name}: {exc}") from exc
    if rows.empty:
        raise ProjectError(f"No image in {project.sample_sheet.name} is marked use = yes.")
    paths = [
        path if path.is_absolute() else (project.sample_sheet.parent / path).resolve()
        for path in (Path(text).expanduser() for text in rows["raw_image_path"])
    ]
    absent = [str(path) for path in paths if not path.is_file()]
    if absent:
        raise ProjectError(f"Image file(s) in the sample sheet not found: {', '.join(absent[:3])}")
    manifest = project.root / "batch_inputs.csv"
    table = {"image_id": rows["image_id"].tolist(), "input_path": [str(path) for path in paths]}
    pd.DataFrame(table).to_csv(manifest, index=False)
    return manifest


def command_run(project: ProjectFolder, extra: list[str]) -> int:
    if runs_from_sample_sheet(project):
        source = ["--input-manifest", str(write_input_manifest(project))]
    else:
        source = ["--input-dir", str(project.images)]
    arguments = [
        *source,
        "--panel-config", str(project.panel),
        "--output-dir", str(project.batch_dir),
    ]  # fmt: skip
    if project.no_model:
        arguments.append("--no-model")
    elif project.model:
        arguments += ["--model", project.model]
    if project.split_czi_scenes:
        arguments.append("--split-czi-scenes")
    return _run("run_batch", [*arguments, *extra])


def command_prepare(project: ProjectFolder) -> int:
    if not (project.batch_dir / "batch_summary.csv").is_file():
        raise ProjectError("The batch has not run yet. Run: python -m fibertypeqc run")
    if project.review_project.is_file():
        print(f"Review project exists: {project.review_project} (kept)")
    else:
        check_sample_sheet(project)
        code = _run(
            "make_review_project",
            [
                "--batch-dir",
                str(project.batch_dir),
                "--sample-sheet",
                str(project.sample_sheet),
                "--panel-config",
                str(project.panel),
                "--project-dir",
                str(project.review_dir),
                "--project-id",
                project.root.name,
                *(
                    ["--classes", ",".join(project.classes),
                     "--negative-label", project.negative_label]
                    if project.no_model
                    else []
                ),
            ],  # fmt: skip
        )
        if code != 0:
            return code
    code = _run("generate_review_qc", ["--project", str(project.review_project)])
    if code == 0 and project.no_model:
        message = ensure_test_mice(project)
        if message:
            print(message)
    return code


def _blind_label_counts(project: ProjectFolder) -> dict[str, int]:
    labels = {}
    for state_path in sorted((project.review_dir / "reference").glob("*/review_state.json")):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        labels[state_path.parent.name] = len(state.get("object_decisions", []))
    return labels


def mouse_split_summary(project: ProjectFolder) -> tuple[int, int] | None:
    """(test mice, all mice) from the project's evaluation roles, or None without a roles file."""
    from src.review.roles import ROLES_FILENAME, TEST

    path = project.root / ROLES_FILENAME
    if not path.is_file():
        return None
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"mouse_id", "role"} <= set(table.columns):
        return None
    return int(table["role"].str.strip().str.lower().eq(TEST).sum()), len(table)


def split_changed_note(project: ProjectFolder) -> str:
    """A warning when the test mice no longer match the recorded automatic split, or ""."""
    from src.review.roles import RECORD_FILENAME, ROLES_FILENAME, TEST

    record_path = project.root / RECORD_FILENAME
    if not record_path.is_file():
        return ""
    recorded = set(json.loads(record_path.read_text(encoding="utf-8")).get("test_mice", []))
    table = pd.read_csv(project.root / ROLES_FILENAME, dtype=str, keep_default_na=False)
    if not {"mouse_id", "role"} <= set(table.columns):
        return ""
    current = set(table.loc[table["role"].str.strip().str.lower().eq(TEST), "mouse_id"].str.strip())
    if current == recorded:
        return ""
    return (
        f"The test mice in {ROLES_FILENAME} differ from the automatic choice recorded in "
        f"{RECORD_FILENAME}. If test labels or an evaluation already exist, the model is no "
        "longer checked independently; say so when reporting it."
    )


def ensure_test_mice(project: ProjectFolder) -> str:
    """Set test mice aside for a project without a model, before any fiber is labelled.

    Writes ``evaluation_roles.csv`` (and its record) only when there is none and no blind label
    exists yet, so an existing split, or one made by hand, is never changed. Returns what was
    done, in plain language, or "".
    """
    from src.model_builder import load_recipe
    from src.review.project import load_project
    from src.review.roles import ROLES_FILENAME, write_default_roles

    if not project.no_model or (project.root / ROLES_FILENAME).exists():
        return ""
    if not project.review_project.is_file() or any(_blind_label_counts(project).values()):
        return ""
    minimum = int(load_recipe()["minimums"]["training_mice"])
    roles = write_default_roles(load_project(project.review_project), minimum)
    return mouse_split_message(sum(role == "test" for role in roles.values()), len(roles), minimum)


def mouse_split_message(n_test: int, n_mice: int, minimum_training: int = 3) -> str:
    if n_test:
        return (
            f"{n_test} of {n_mice} mice are set aside to check the model fairly "
            f"({ROLES_NOTE}). Label them too; the model never learns from them."
        )
    return (
        f"With {n_mice} mice, none can be set aside to check the model: at least "
        f"{minimum_training + 1} are needed ({minimum_training} to learn from, 1 to test). The "
        "model can still be built, but it will not have an independent check."
    )


ROLES_NOTE = "listed in evaluation_roles.csv"


def command_review(project: ProjectFolder, extra: list[str]) -> int:
    if not (project.qc_dir / "fiber_qc.csv").is_file():
        raise ProjectError("QC has not run yet. Run: python -m fibertypeqc prepare")
    arguments = [
        "--project", str(project.review_project),
        "--display-downsample", str(project.display_downsample),
    ]  # fmt: skip
    if project.reviewer:
        arguments += ["--reviewer", project.reviewer]
    if project.no_model and "--blind" not in extra:
        raise ProjectError(
            "This project has no model, so there are no calls to review. Label a sample of fibers "
            "instead: python -m fibertypeqc review PROJECT --blind --reviewer NAME"
        )
    if project.no_model:
        message = ensure_test_mice(project)  # projects prepared before test mice were automatic
        if message:
            print(message)
    return _run("review_project_napari", [*arguments, *extra])


def _require_model(project: ProjectFolder, step: str) -> None:
    if project.no_model:
        raise ProjectError(
            f"This project has no model yet ('model: none'), so '{step}' has no fiber calls to "
            "work with. Label fibers with: review --blind --reviewer NAME."
        )


def command_finalize(project: ProjectFolder) -> int:
    _require_model(project, "finalize")
    if not project.review_project.is_file():
        raise ProjectError("The review project is not built. Run: python -m fibertypeqc prepare")
    code = _run(
        "finalize_review_project",
        [
            "--project", str(project.review_project),
            "--output-dir", str(project.final_dir),
            *(["--exclude-image-border-fibers"] if project.exclude_image_border_fibers else []),
        ],  # fmt: skip
    )
    if code != 0:
        return code
    code = _run(
        "summarize_results",
        [
            "--final-dir",
            str(project.final_dir),
            "--project",
            str(project.review_project),
            "--output-dir",
            str(project.results_dir),
        ],  # fmt: skip
    )
    if code == 0:
        print(f"Open: {project.results_dir / 'cohort_report.html'}")
    return code


# --- init -----------------------------------------------------------------------------------

IMAGE_SUFFIXES = (".czi", ".tif", ".tiff")


def image_channel_count(path: Path) -> int | None:
    """Channel count from file metadata only (no pixel data is read)."""
    try:
        if path.suffix.lower() == ".czi":
            import czifile

            with czifile.CziFile(str(path)) as czi:
                axes, shape = czi.axes, czi.shape
        else:
            import tifffile

            with tifffile.TiffFile(path) as tif:
                axes, shape = tif.series[0].axes, tif.series[0].shape
    except Exception:  # unreadable metadata only costs the hint
        return None
    for name in ("C", "S") if path.suffix.lower() != ".czi" else ("C",):
        if name in axes:
            return int(shape[axes.index(name)])
    return 1


def image_channel_names(path: Path) -> list[str]:
    """Channel names from file metadata (CZI or OME-TIFF), or [] when there are none."""
    import xml.etree.ElementTree as ET

    try:
        if path.suffix.lower() == ".czi":
            import czifile

            with czifile.CziFile(str(path)) as czi:
                root = ET.fromstring(czi.metadata())
            channels = root.findall("./Metadata/Information/Image/Dimensions/Channels/Channel")
        else:
            import tifffile

            with tifffile.TiffFile(path) as tif:
                if not tif.is_ome or not tif.ome_metadata:
                    return []
                root = ET.fromstring(tif.ome_metadata)
            channels = root.findall(".//{*}Image[1]/{*}Pixels/{*}Channel")
    except Exception:  # unreadable metadata only costs the names
        return []
    names = [(channel.get("Name") or "").strip() for channel in channels]
    return names if any(names) else []


def channel_summary(path: Path, count: int, names: list[str]) -> str:
    """The first image's channels as shown to a person, named when the metadata has names."""
    if names and len(names) == count:
        listing = ", ".join(channel_label(index, names) for index in range(count))
        return f"{path.name} has {count} channel(s): {listing}"
    return f"{path.name} has {count} channel(s), {channel_range(count)}"


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer or default


def _choose_panel(images: list[Path], interactive: bool, requested: str | None) -> tuple[dict, str]:
    """Return (panel mapping, description) from a preset name, a file, or interactive answers."""
    presets = list_presets()
    if requested:
        if requested in presets:
            path = presets[requested][0]
            return yaml.safe_load(path.read_text(encoding="utf-8")), f"preset {requested}"
        path = Path(requested).expanduser()
        if not path.is_file():
            raise ProjectError(
                f"--panel must be a preset ({', '.join(presets)}) or a panel file; got {requested}"
            )
        return yaml.safe_load(path.read_text(encoding="utf-8")), f"copied from {path.name}"
    if not interactive:
        raise ProjectError(f"--panel is required (a preset: {', '.join(presets)}; or a file)")
    channels = image_channel_count(images[0]) if images else None
    if channels:
        channel_names = image_channel_names(images[0])
        print(f"\n{channel_summary(images[0], channels, channel_names)}.")
    print("Panel presets:")
    names = list(presets)
    for number, name in enumerate(names, start=1):
        print(f"  {number}. {name}: {presets[name][1]}")
    print(f"  {len(names) + 1}. custom: I will enter the channel for each stain")
    choice = _ask("Choose a panel", "1")
    if choice.isdigit() and 1 <= int(choice) <= len(names):
        name = names[int(choice) - 1]
        return yaml.safe_load(presets[name][0].read_text(encoding="utf-8")), f"preset {name}"
    if choice in presets:
        return yaml.safe_load(presets[choice][0].read_text(encoding="utf-8")), f"preset {choice}"
    print(
        "Enter the channel number for each stain, counting from 1 as ZEN and Fiji do (C1, C2, "
        "...), or leave blank if it is not in your images."
    )
    answers: dict[str, int | None] = {}
    for marker in MARKERS:
        text = _ask(f"  {MARKER_LABELS[marker]}")
        if not text:
            answers[marker] = None
            continue
        if not text.isdigit() or int(text) < 1 or (channels and int(text) > channels):
            limit = f" to {channels}" if channels else " or more"
            raise ProjectError(f"{MARKER_LABELS[marker]}: channel must be 1{limit}; got {text}")
        answers[marker] = int(text) - 1  # panel files count from 0
    try:
        return panel_from_channels(answers), "entered interactively"
    except ValueError as exc:
        raise ProjectError(str(exc)) from exc


def command_init(args: argparse.Namespace) -> int:
    root = args.project.expanduser().resolve()
    config_path = root / CONFIG_NAME
    if config_path.exists():
        raise ProjectError(f"{config_path} already exists; this folder is already a project.")
    interactive = not args.yes and sys.stdin.isatty()
    images_text = args.images or (_ask("Folder containing your images") if interactive else "")
    if not images_text:
        raise ProjectError("--images is required")
    images_dir = Path(images_text).expanduser().resolve()
    include, exclude = tuple(args.include or ()), tuple(args.exclude or ())
    if interactive and not include and images_dir.is_dir():
        flat = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        nested = _nested_images(images_dir)
        if len(nested) > len(flat):
            print(
                f"\n{len(nested)} image(s) in {images_dir} and its subfolders, "
                f"{len(flat)} directly inside it."
            )
            print(
                "Choose which files to use with patterns ('**/' means any subfolders). Use "
                "either the original CZIs or processed TIFFs, not both, or the same sections "
                "are counted twice."
            )
            text = _ask("Include (comma-separated, blank for top folder only)", "**/*.czi")
            include = tuple(item.strip() for item in text.split(",") if item.strip())
            if include:
                text = _ask("Exclude (comma-separated)", "")
                exclude = tuple(item.strip() for item in text.split(",") if item.strip())
    images = image_files(images_dir, include, exclude)
    if include:
        print(f"\n{len(images)} image(s) match.")
    panel, panel_source = _choose_panel(images, interactive, args.panel)
    default_model = default_model_id("fiber_identity") or ""
    model = args.model or (
        _ask(
            "Model: a registered ID, the path to your model file, or 'none' if your panel has "
            "no model yet",
            default_model,
        )
        if interactive
        else default_model
    )
    rows = create_project(
        root, images_dir, panel, panel_source, model, args.mouse_id_pattern, include, exclude
    )
    missing_mice = sum(not row["mouse_id"] for row in rows)
    print(f"\nCreated project in {root}")
    print(f"  {CONFIG_NAME}: images, panel, sample sheet, model ({model or 'none'})")
    print(f"  panel.yaml: {panel_source}")
    print(f"  samples.csv: {len(rows)} image(s)")
    if include:
        print(
            "  Images come from the sample sheet: set use to 'no' to leave one out. "
            "Check that no section appears twice (e.g. as a CZI and a processed TIFF)."
        )
    if missing_mice:
        print(
            f"\nBefore 'prepare': open samples.csv and fill in mouse_id for {missing_mice} "
            "image(s). Rename or add columns (e.g. group, genotype) to define cohort groups."
        )
    print(f"\nNext: python -m fibertypeqc run {root}")
    return 0


def _pattern_regex(pattern: str) -> re.Pattern[str]:
    """A glob over relative paths: ``**/`` spans folders, ``*`` and ``?`` stay within one name.

    Matching ignores case, so ``**/*.czi`` also finds ``.CZI`` files.
    """
    out, index = "", 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            out, index = out + "(?:.*/)?", index + 3
        elif pattern.startswith("**", index):
            out, index = out + ".*", index + 2
        elif pattern[index] == "*":
            out, index = out + "[^/]*", index + 1
        elif pattern[index] == "?":
            out, index = out + "[^/]", index + 1
        else:
            out, index = out + re.escape(pattern[index]), index + 1
    return re.compile(out, re.IGNORECASE)


def _nested_images(images_dir: Path) -> list[Path]:
    """Image files at any depth, skipping hidden files and folders and other project folders."""
    found: list[Path] = []
    for folder, names, files in os.walk(images_dir):
        here = Path(folder)
        if here != images_dir and (here / CONFIG_NAME).is_file():
            names[:] = []  # a FiberTypeQC project (its scene exports are images too)
            continue
        names[:] = sorted(name for name in names if not name.startswith("."))
        found += [
            here / name
            for name in files
            if not name.startswith(".") and Path(name).suffix.lower() in IMAGE_SUFFIXES
        ]
    return found


def image_files(
    images_dir: Path, include: tuple[str, ...] = (), exclude: tuple[str, ...] = ()
) -> list[Path]:
    """The .czi/.tif images to use, or a plain-language error.

    Without ``include`` patterns: the files directly inside ``images_dir``. With them: files at
    any depth whose path relative to ``images_dir`` matches an include and no exclude pattern.
    """
    if not images_dir.is_dir():
        raise ProjectError(f"Images folder not found: {images_dir}")
    if not include:
        images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            nested = len(_nested_images(images_dir))
            hint = (
                f" ({nested} found in subfolders; choose them with include patterns, e.g. "
                "init --include '**/*.czi')"
                if nested
                else ""
            )
            raise ProjectError(f"No .czi/.tif/.tiff files found in {images_dir}{hint}")
        return images
    wanted = [_pattern_regex(pattern) for pattern in include]
    unwanted = [_pattern_regex(pattern) for pattern in exclude]
    images = []
    for path in _nested_images(images_dir):
        relative = path.relative_to(images_dir).as_posix()
        if any(r.fullmatch(relative) for r in wanted) and not any(
            r.fullmatch(relative) for r in unwanted
        ):
            images.append(path)
    if not images:
        raise ProjectError(
            f"No .czi/.tif/.tiff files in {images_dir} match include {list(include)}"
            + (f" and exclude {list(exclude)}" if exclude else "")
        )
    return sorted(images, key=lambda path: path.relative_to(images_dir).as_posix().lower())


def project_images(project: ProjectFolder) -> list[Path]:
    return image_files(project.images, project.include, project.exclude)


def _unique_ids(images: list[Path], images_dir: Path | None) -> list[str]:
    """File stems, with parent folder names added to the ones that would otherwise clash.

    ``m1/s1.czi`` and ``m2/s1.czi`` become ``m1_s1`` and ``m2_s1``; a clash that remains is
    reported, since run_batch would write both images to one output folder.
    """
    ids = [image.stem for image in images]
    if images_dir is None:
        return ids
    parents = [image.relative_to(images_dir).parent.parts for image in images]
    for depth in range(1, max((len(parts) for parts in parents), default=0) + 1):
        counts = Counter(ids)
        if all(count == 1 for count in counts.values()):
            break
        ids = [
            "_".join([*parts[-depth:], image.stem]) if counts[current] > 1 else current
            for current, image, parts in zip(ids, images, parents, strict=True)
        ]
    clashes = sorted(name for name, count in Counter(ids).items() if count > 1)
    if clashes:
        raise ProjectError(f"These images would get the same image ID: {', '.join(clashes)}")
    return ids


def sample_rows(
    images: list[Path], mouse_id_pattern: str | None, images_dir: Path | None = None
) -> list[dict[str, str]]:
    """Sample-sheet rows; the pattern's first group (or whole match) fills mouse_id.

    With ``images_dir`` (images found in subfolders), image IDs that would clash get their
    parent folder names, and each row gets ``use = yes``.
    """
    try:
        pattern = re.compile(mouse_id_pattern) if mouse_id_pattern else None
    except re.error as exc:
        raise ProjectError(f"Mouse ID pattern is not a valid regular expression: {exc}") from exc
    rows = []
    for image, image_id in zip(images, _unique_ids(images, images_dir), strict=True):
        match = pattern.search(image.stem) if pattern else None
        mouse_id = (match.group(1) if match and match.groups() else match.group(0)) if match else ""
        row = {
            "image_id": image_id,
            "mouse_id": mouse_id or "",
            "raw_image_path": str(image),
            "group": "",
        }
        if images_dir is not None:
            row[USE_COLUMN] = "yes"
        rows.append(row)
    return rows


def create_project(
    root: Path,
    images_dir: Path,
    panel: dict,
    panel_source: str,
    model: str | None,
    mouse_id_pattern: str | None = None,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> list[dict[str, str]]:
    """Write a new project folder (config, panel, sample sheet); shared by init and the launcher."""
    config_path = root / CONFIG_NAME
    if config_path.exists():
        raise ProjectError(f"{config_path} already exists; this folder is already a project.")
    if exclude and not include:
        raise ProjectError("Exclude patterns need include patterns, e.g. --include '**/*.czi'")
    images = image_files(images_dir, include, exclude)
    rows = sample_rows(images, mouse_id_pattern, images_dir if include else None)
    root.mkdir(parents=True, exist_ok=True)
    write_panel(root / "panel.yaml", panel, f"Panel for this project ({panel_source}).")
    pd.DataFrame(rows).to_csv(root / "samples.csv", index=False)
    config = {
        "schema_version": CONFIG_SCHEMA,
        "images": str(images_dir),
        "panel": "panel.yaml",
        "sample_sheet": "samples.csv",
        "model": model or None,
        "split_czi_scenes": True,
        "display_downsample": 2,
    }
    if include:
        config["include"] = list(include)
        config["exclude"] = list(exclude)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return rows


def check_sample_sheet(project: ProjectFolder) -> None:
    """Fail early, in plain language, when the sample sheet cannot be used yet."""
    if not project.sample_sheet.is_file():
        raise ProjectError(f"Sample sheet not found: {project.sample_sheet}")
    sheet = pd.read_csv(project.sample_sheet, dtype=str, keep_default_na=False)
    missing = [name for name in ("image_id", "mouse_id", "raw_image_path") if name not in sheet]
    if missing:
        raise ProjectError(f"{project.sample_sheet.name} is missing columns: {', '.join(missing)}")
    try:
        sheet = used_rows(sheet)
    except ValueError as exc:
        raise ProjectError(f"{project.sample_sheet.name}: {exc}") from exc
    if USE_COLUMN in sheet and sheet.empty:
        raise ProjectError(f"No image in {project.sample_sheet.name} is marked use = yes.")
    blank = sheet.loc[sheet["mouse_id"].str.strip().eq(""), "image_id"].tolist()
    if blank:
        shown = ", ".join(blank[:5]) + (" …" if len(blank) > 5 else "")
        raise ProjectError(
            f"Fill in mouse_id in {project.sample_sheet} for {len(blank)} image(s): {shown}"
        )


def command_snapshot(project: ProjectFolder, args: argparse.Namespace) -> int:
    path = (args.file or project.root / "reference_snapshot.json").expanduser()
    try:
        if args.freeze:
            snapshot = freeze_snapshot(project.batch_dir, path)
            print(f"Froze {len(snapshot['sections'])} section(s) to {path}")
            return 0
        checks = check_snapshot(
            project.batch_dir,
            path,
            count_tolerance=args.count_tolerance,
            composition_tolerance=args.composition_tolerance,
        )
    except (OSError, ValueError) as exc:
        raise ProjectError(str(exc)) from exc
    for item in checks:
        print(f"{'PASS' if item.passed else 'FAIL'} [{item.level}] {item.image_id}: {item.detail}")
    failed = sum(not item.passed for item in checks)
    print(f"{len(checks) - failed}/{len(checks)} section(s) match the reference")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fibertypeqc",
        description="Run FiberTypeQC steps on a project folder.",
    )
    steps = parser.add_subparsers(dest="step", required=True)
    for name, text in (
        ("status", "Show where the project is and the next command"),
        ("run", "Segment, type, and QC every image (extra options go to run_batch)"),
        ("prepare", "Build the review project and run project QC"),
        ("review", "Open the Napari review workspace (extra options go to the reviewer)"),
        ("finalize", "Finalize review decisions and build result tables and the report"),
        ("reference", "Export blind reference labels (from review --blind) for benchmarking"),
        ("improve", "Build and evaluate candidate models from your review decisions"),
        ("build", "Build and evaluate a first model from blind labels (projects with model: none)"),
    ):
        step = steps.add_parser(name, help=text)
        step.add_argument("project", type=Path, help="Project folder")
        # Declared here so they may come before or after the project folder.
        if name == "review":
            step.add_argument("--blind", action="store_true", help="Blind reference labelling")
            step.add_argument("--reviewer", help="Reviewer name")
        if name == "build":
            step.add_argument(
                "--reviewer", action="append", help="Use only this reviewer's labels (repeatable)"
            )
    promote = steps.add_parser(
        "promote",
        help=(
            "Switch the project to a candidate model from 'improve' or 'build' (or undo with "
            "--rollback)"
        ),
        description=(
            "Archive the current outputs and review state, re-type every section with the "
            "candidate on the existing masks, carry review decisions forward, and record the "
            "change. Nothing is switched unless you run this."
        ),
    )
    promote.add_argument("project", type=Path, help="Project folder")
    promote.add_argument("candidate", nargs="?", help="Candidate name (e.g. logistic) or file")
    promote.add_argument(
        "--override",
        action="store_true",
        help=(
            "Switch even though the candidate was not recommended, or (for a built model) has "
            "no current independent test evaluation (recorded)"
        ),
    )
    promote.add_argument("--rollback", action="store_true", help="Undo the most recent switch")
    evaluate = steps.add_parser(
        "evaluate",
        help="Score one chosen built model on the test mice (before promoting it)",
        description=(
            "Independent evaluation for a project with 'model: none': score the candidate you "
            "chose from 'build' on the blind labels of the test mice. Only that candidate is "
            "scored; every evaluation is logged in models/evaluation_history.csv."
        ),
    )
    evaluate.add_argument("project", type=Path, help="Project folder")
    evaluate.add_argument("candidate", help="Candidate name (e.g. logistic) or model file")
    snapshot = steps.add_parser(
        "snapshot",
        help="Freeze the batch outputs as a reference, or check this run against one",
        description=(
            "Record per-section fiber counts, class counts, and digests of masks and fiber calls "
            "(--freeze), or compare this project's batch outputs with a frozen snapshot (--check). "
            "Identical masks must give identical calls; re-segmented sections must agree within "
            "the tolerances."
        ),
    )
    snapshot.add_argument("project", type=Path, help="Project folder")
    mode = snapshot.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true", help="Write a new snapshot")
    mode.add_argument("--check", action="store_true", help="Compare with an existing snapshot")
    snapshot.add_argument(
        "--file", type=Path, help="Snapshot file (default: PROJECT/reference_snapshot.json)"
    )
    snapshot.add_argument("--count-tolerance", type=float, default=DEFAULT_COUNT_TOLERANCE)
    snapshot.add_argument(
        "--composition-tolerance", type=float, default=DEFAULT_COMPOSITION_TOLERANCE
    )
    init = steps.add_parser(
        "init",
        help="Create a project folder: config, panel file, and a sample-sheet template",
        description=(
            "Create a project folder. Without options it asks a few questions; with --images and "
            "--panel it runs unattended."
        ),
    )
    init.add_argument("project", type=Path, help="Project folder to create")
    init.add_argument("--images", help="Folder containing .czi/.tif images")
    init.add_argument(
        "--include",
        action="append",
        metavar="PATTERN",
        help=(
            "Use images in subfolders that match this pattern, relative to --images "
            "(repeatable; e.g. '**/*.czi'; '**/' means any subfolders). Without it, only files "
            "directly inside --images are used."
        ),
    )
    init.add_argument(
        "--exclude",
        action="append",
        metavar="PATTERN",
        help="Leave out images matching this pattern (repeatable; e.g. '**/Processed/**')",
    )
    init.add_argument(
        "--panel",
        help=f"Panel preset ({', '.join(list_presets())}) or a panel YAML file to copy",
    )
    init.add_argument(
        "--model",
        help="Registered model ID, path to a model file, or 'none' for a panel without a model",
    )
    init.add_argument(
        "--mouse-id-pattern",
        help="Regular expression applied to each file name; its first group fills mouse_id",
    )
    init.add_argument("--yes", action="store_true", help="Never prompt; fail if input is missing")
    app = steps.add_parser(
        "app",
        help="Open the project launcher window (experimental)",
        description=(
            "A window that shows this project's steps one at a time and runs them. It runs the "
            "same commands as the terminal, on this computer."
        ),
    )
    app.add_argument("project", type=Path, nargs="?", help="Project folder to open (optional)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args, extra = build_parser().parse_known_args(argv)
    if extra and args.step not in {"run", "review", "build"}:
        print(f"Unrecognized options for {args.step}: {' '.join(extra)}", file=sys.stderr)
        return 2
    try:
        if args.step == "init":
            return command_init(args)
        if args.step == "app":
            from src.launcher import main as launch

            return launch(args.project)
        project = load_project_folder(args.project)
        if args.step == "status":
            return command_status(project)
        if args.step == "snapshot":
            return command_snapshot(project, args)
        if args.step == "run":
            return command_run(project, extra)
        if args.step == "prepare":
            return command_prepare(project)
        if args.step == "review":
            passed = ["--blind"] if args.blind else []
            passed += ["--reviewer", args.reviewer] if args.reviewer else []
            return command_review(project, [*passed, *extra])
        if args.step == "improve":
            _require_model(project, args.step)
        if args.step == "evaluate":
            if not project.no_model:
                raise ProjectError("'evaluate' is for built models in projects with 'model: none'.")
            return _run(
                "evaluate_model",
                ["--project-root", str(project.root), "--candidate", args.candidate],
            )
        if args.step == "promote":
            arguments = ["--project-root", str(project.root), "--panel-config", str(project.panel)]
            if args.rollback:
                return _run("promote_model", [*arguments, "--rollback"])
            if args.candidate:
                arguments += ["--candidate", args.candidate]
            if args.override:
                arguments.append("--override")
            code = _run("promote_model", arguments)
            if code != 0:
                return code
            # Project QC is computed from the model's outputs, so it is rebuilt for the new model.
            return _run("generate_review_qc", ["--project", str(project.review_project)])
        if args.step == "build":
            if not project.no_model:
                raise ProjectError(
                    "This project has a model; adapt it with 'improve'. 'build' is for "
                    "projects with 'model: none'."
                )
            if not project.review_project.is_file():
                raise ProjectError("Run 'prepare' first, then label fibers with review --blind.")
            arguments = [
                "--project", str(project.review_project),
                "--panel-config", str(project.panel),
                "--output-dir", str(project.root / "models"),
            ]  # fmt: skip
            for reviewer in args.reviewer or []:
                arguments += ["--reviewer", reviewer]
            return _run("build_model", [*arguments, *extra])
        if args.step == "improve":
            arguments = [
                "--project", str(project.review_project),
                "--panel-config", str(project.panel),
                "--output-dir", str(project.root / "models"),
            ]  # fmt: skip
            if project.model:
                arguments += ["--model", project.model]
            return _run("improve_model", arguments)
        if args.step == "reference":
            return _run(
                "export_reference",
                [
                    "--project", str(project.review_project),
                    "--output-dir", str(project.root / "reference_export"),
                ],  # fmt: skip
            )
        return command_finalize(project)
    except ProjectError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
