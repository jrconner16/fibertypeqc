"""Open the FiberTypeQC project cohort dashboard in Napari."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.generate_review_qc import load_manual_selections
from src.review.dashboard import build_dashboard_model, load_dashboard_tables
from src.review.project import load_project
from src.review.section_selection import SelectionStrategy
from src.review.storage import load_session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument(
        "--qc-dir",
        type=Path,
        help="Phase 2A QC directory; defaults to <project>/qc.",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=[strategy.value for strategy in SelectionStrategy],
        default=SelectionStrategy.ALL_PASSING.value,
    )
    parser.add_argument(
        "--manual-selection",
        type=Path,
        help="Optional manual mouse/domain/image selection YAML.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project = load_project(args.project, validate_paths=False)
    tables = load_dashboard_tables(project, args.qc_dir)
    manual = load_manual_selections(args.manual_selection)
    session = (
        load_session(project.review_state_path, expected_project_id=project.project_id)
        if project.review_state_path.is_file()
        else None
    )
    model = build_dashboard_model(
        project,
        tables,
        strategy=args.selection_strategy,
        manual_selections=manual,
        session=session,
    )

    import napari

    from src.review.dashboard_widget import CohortDashboardWidget

    viewer = napari.Viewer(title=f"FiberTypeQC project: {project.project_name}")
    widget = CohortDashboardWidget(
        project,
        tables,
        strategy=model.strategy,
        manual_selections=manual,
        session=session,
    )
    viewer.window.add_dock_widget(widget, area="right", name="Cohort Dashboard")
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
