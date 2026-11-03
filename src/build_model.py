"""Build and evaluate a first model for a panel that has none, from blind labels.

Uses the project's blind labels on training (pool) mice and the marker measurements written by
the run without a model. Candidates are evaluated, refitted, and saved under ``<output-dir>``;
the project's model is never changed.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from fibertypeqc.config import load_channel_config
from src.model_builder import BuilderError, blind_label_sessions, build, load_recipe
from src.review.project import load_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True, help="Review project YAML")
    parser.add_argument("--panel-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder for builder runs")
    parser.add_argument(
        "--reviewer",
        action="append",
        default=None,
        help="Use only this reviewer's blind labels (repeat for several; default: all)",
    )
    parser.add_argument("--recipe", type=Path, default=None, help="Builder recipe YAML")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        project = load_project(args.project)
        markers = load_channel_config(args.panel_config).available_markers
        sessions = blind_label_sessions(project, args.reviewer)
        run_dir = args.output_dir / datetime.now(UTC).strftime("build_%Y%m%dT%H%M%SZ")
        result = build(
            project,
            sessions,
            classes=list(project.review_classes),
            markers=markers,
            output_dir=run_dir,
            recipe=load_recipe(args.recipe),
            class_labels=project.class_labels,
        )
    except (BuilderError, ValueError) as exc:
        print(f"Model building stopped: {exc}", file=sys.stderr)
        return 1

    shown = project.class_labels  # panel-specific names, e.g. negative -> I⁻/IIa⁻
    labels = result["labels"]
    print(
        f"{labels['n_fibers']} labelled fibers from {labels['n_training_mice']} training mice: "
        + ", ".join(f"{shown.get(n, n)} {count}" for n, count in labels["by_class"].items())
    )
    left_out = {key: value for key, value in labels["left_out"].items() if value}
    if left_out or labels["conflicting_fibers"]:
        details = ", ".join(f"{key} {value}" for key, value in left_out.items())
        print(f"left out: {details or 'none'}; reviewer conflicts {labels['conflicting_fibers']}")
    metrics = result["selection"]["metrics"]
    classes = result["classes"]
    header = ", ".join(shown.get(c, c) for c in classes)
    print(f"\n{'model':<20}{'median mouse macro-F1':>23}  pooled recall ({header})")
    for name, values in metrics.items():
        recalls = ", ".join(f"{values['pooled_recall_by_class'][c]:.2f}" for c in classes)
        print(f"{name:<20}{values['median_mouse_macro_f1']:>23.3f}  {recalls}")
    decision = result["selection"]["decision"]
    print(f"\nRecommendation: {decision['recommended'] or 'none'}")
    print(decision["reason"])
    print(f"\n{result['caveat']}")
    print(f"\nrun folder: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
