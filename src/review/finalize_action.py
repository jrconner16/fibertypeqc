"""Finalize a reviewed project and build its results and report in one call.

Used by the reviewer's "Finalize and build report" action; GUI-independent so it can be tested
headlessly. Outputs follow the project-folder convention when the review project sits inside one
(``<folder>/final`` and ``<folder>/results``); otherwise they go next to the review project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fibertypeqc.cohort_report import write_cohort_report
from src.review.finalization import finalize_project
from src.review.project import Project
from src.review.session import ReviewSession
from src.summarize_results import write_results

PROJECT_FOLDER_CONFIG = "fibertypeqc_project.yaml"


@dataclass(frozen=True)
class FinalizeOutcome:
    final_dir: Path
    results_dir: Path
    report_path: Path
    manifest: dict[str, Any]

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for image in self.manifest["images"]:
            for key, value in image["counts"].items():
                counts[key] = counts.get(key, 0) + int(value)
        return (
            f"{len(self.manifest['images'])} section(s): {counts.get('reviewed', 0)} reviewed, "
            f"{counts.get('excluded', 0)} excluded, {counts.get('unresolved', 0)} unresolved, "
            f"{counts.get('flagged_unreviewed', 0)} flagged but not reviewed"
        )


def output_directories(project: Project) -> tuple[Path, Path]:
    base = (
        project.root.parent
        if (project.root.parent / PROJECT_FOLDER_CONFIG).is_file()
        else project.root
    )
    return base / "final", base / "results"


def excludes_image_border_fibers(project: Project) -> bool:
    """The project folder's ``exclude_image_border_fibers`` option (off without a folder)."""
    config = project.root.parent / PROJECT_FOLDER_CONFIG
    if not config.is_file():
        return False
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    return bool(isinstance(raw, dict) and raw.get("exclude_image_border_fibers", False))


def finalize_and_report(
    project: Project, session: ReviewSession, qc_dir: Path | None = None
) -> FinalizeOutcome:
    """Apply the current review state, then write result tables and the cohort report."""
    final_dir, results_dir = output_directories(project)
    selection = (qc_dir or project.root / "qc") / "section_selection.csv"
    manifest = finalize_project(
        project,
        session,
        final_dir,
        section_selection_path=selection if selection.is_file() else None,
        exclude_image_border_fibers=excludes_image_border_fibers(project),
    )
    write_results(final_dir, results_dir, project.manifest_path)
    report_path = write_cohort_report(results_dir)
    return FinalizeOutcome(final_dir, results_dir, report_path, manifest)
