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

Each step calls the corresponding ``scripts.*`` command with paths taken from the project folder.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from fibertypeqc.model_resolution import default_model_id
from fibertypeqc.panel_presets import (
    MARKER_LABELS,
    MARKERS,
    list_presets,
    panel_from_channels,
    preset_path,
    write_panel,
)
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
    if isinstance(model, str) and _is_path_like(model):
        model = str(_resolve(root, model, "model"))
    return ProjectFolder(
        root=root,
        images=_resolve(root, raw.get("images"), "images"),
        # A preset name may be used directly instead of a panel file in the project folder.
        panel=preset_path(str(raw.get("panel"))) or _resolve(root, raw.get("panel"), "panel"),
        sample_sheet=_resolve(root, raw.get("sample_sheet"), "sample_sheet"),
        model=model,
        split_czi_scenes=bool(raw.get("split_czi_scenes", True)),
        reviewer=str(raw.get("reviewer", "") or ""),
        display_downsample=int(raw.get("display_downsample", 2)),
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
    if "report" in status:
        print(f"Report:  {status['report']}")
    for note in status["notes"]:
        print(f"Note:    {note}")
    step, explanation = NEXT_STEP[status["stage"]]
    print(f"Next:    python -m fibertypeqc {step} {project.root}")
    print(f"         {explanation}")
    return 0


# --- steps ----------------------------------------------------------------------------------


def command_run(project: ProjectFolder, extra: list[str]) -> int:
    arguments = [
        "--input-dir", str(project.images),
        "--panel-config", str(project.panel),
        "--output-dir", str(project.batch_dir),
    ]  # fmt: skip
    if project.model:
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
            ],  # fmt: skip
        )
        if code != 0:
            return code
    return _run("generate_review_qc", ["--project", str(project.review_project)])


def command_review(project: ProjectFolder, extra: list[str]) -> int:
    if not (project.qc_dir / "fiber_qc.csv").is_file():
        raise ProjectError("QC has not run yet. Run: python -m fibertypeqc prepare")
    arguments = [
        "--project", str(project.review_project),
        "--display-downsample", str(project.display_downsample),
    ]  # fmt: skip
    if project.reviewer:
        arguments += ["--reviewer", project.reviewer]
    return _run("review_project_napari", [*arguments, *extra])


def command_finalize(project: ProjectFolder) -> int:
    if not project.review_project.is_file():
        raise ProjectError("The review project is not built. Run: python -m fibertypeqc prepare")
    code = _run(
        "finalize_review_project",
        ["--project", str(project.review_project), "--output-dir", str(project.final_dir)],
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
        print(f"\n{images[0].name} has {channels} channel(s), numbered 0 to {channels - 1}.")
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
    print("Enter the channel number for each stain, or leave blank if it is not in your images.")
    answers: dict[str, int | None] = {}
    for marker in MARKERS:
        text = _ask(f"  {MARKER_LABELS[marker]}")
        answers[marker] = int(text) if text else None
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
    if not images_dir.is_dir():
        raise ProjectError(f"Images folder not found: {images_dir}")
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ProjectError(f"No .czi/.tif/.tiff files found in {images_dir}")
    panel, panel_source = _choose_panel(images, interactive, args.panel)
    default_model = default_model_id("fiber_identity") or ""
    model = args.model or (
        _ask("Model: a registered ID or the path to your model file", default_model)
        if interactive
        else default_model
    )
    pattern = re.compile(args.mouse_id_pattern) if args.mouse_id_pattern else None
    rows = []
    for image in images:
        match = pattern.search(image.stem) if pattern else None
        mouse_id = (match.group(1) if match and match.groups() else match.group(0)) if match else ""
        rows.append(
            {
                "image_id": image.stem,
                "mouse_id": mouse_id,
                "raw_image_path": str(image),
                "group": "",
            }
        )

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
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    missing_mice = sum(not row["mouse_id"] for row in rows)
    print(f"\nCreated project in {root}")
    print(f"  {CONFIG_NAME}: images, panel, sample sheet, model ({model or 'none'})")
    print(f"  panel.yaml: {panel_source}")
    print(f"  samples.csv: {len(rows)} image(s)")
    if missing_mice:
        print(
            f"\nBefore 'prepare': open samples.csv and fill in mouse_id for {missing_mice} "
            "image(s). Rename or add columns (e.g. group, genotype) to define cohort groups."
        )
    print(f"\nNext: python -m fibertypeqc run {root}")
    return 0


def check_sample_sheet(project: ProjectFolder) -> None:
    """Fail early, in plain language, when the sample sheet cannot be used yet."""
    if not project.sample_sheet.is_file():
        raise ProjectError(f"Sample sheet not found: {project.sample_sheet}")
    sheet = pd.read_csv(project.sample_sheet, dtype=str, keep_default_na=False)
    missing = [name for name in ("image_id", "mouse_id", "raw_image_path") if name not in sheet]
    if missing:
        raise ProjectError(f"{project.sample_sheet.name} is missing columns: {', '.join(missing)}")
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
    ):
        step = steps.add_parser(name, help=text)
        step.add_argument("project", type=Path, help="Project folder")
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
        "--panel",
        help=f"Panel preset ({', '.join(list_presets())}) or a panel YAML file to copy",
    )
    init.add_argument("--model", help="Registered model ID or path to a model file")
    init.add_argument(
        "--mouse-id-pattern",
        help="Regular expression applied to each file name; its first group fills mouse_id",
    )
    init.add_argument("--yes", action="store_true", help="Never prompt; fail if input is missing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args, extra = build_parser().parse_known_args(argv)
    if extra and args.step not in {"run", "review"}:
        print(f"Unrecognized options for {args.step}: {' '.join(extra)}", file=sys.stderr)
        return 2
    try:
        if args.step == "init":
            return command_init(args)
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
            return command_review(project, extra)
        return command_finalize(project)
    except ProjectError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
