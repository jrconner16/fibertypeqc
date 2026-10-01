"""Finalize a reviewed project into analysis-ready fiber tables.

Reads the project manifest, its predictions and label masks, and the saved review state, then
writes ``<image_id>_fibers_finalized.csv`` per image, ``final_fiber_table.csv``, and
``finalization_manifest.json``. Inputs are never modified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.review.finalization import FinalizationError, finalize_project, import_legacy_reviews
from src.review.project import load_project
from src.review.storage import load_session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True, help="Review project YAML")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--qc-dir",
        type=Path,
        default=None,
        help=(
            "Project QC directory containing section_selection.csv (default: <project>/qc when "
            "present). Fiber-typing sections not selected there are excluded."
        ),
    )
    parser.add_argument(
        "--legacy-review",
        action="append",
        default=[],
        metavar="IMAGE_ID=CSV",
        help=(
            "Import a per-image review CSV from the legacy reviewer (repeatable). Its decisions "
            "are applied like project-review decisions; conflicts with the session are refused."
        ),
    )
    parser.add_argument(
        "--require-verified",
        action="store_true",
        help="Refuse images whose review session has no recorded input fingerprints.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project = load_project(args.project)
    session = load_session(project.review_state_path, expected_project_id=project.project_id)
    qc_dir = args.qc_dir if args.qc_dir is not None else project.root / "qc"
    selection = qc_dir / "section_selection.csv"
    legacy: dict[str, Path] = {}
    for item in args.legacy_review:
        image_id, separator, path = item.partition("=")
        if not separator or not image_id or not path:
            print(f"--legacy-review must be IMAGE_ID=CSV, got {item!r}", file=sys.stderr)
            return 2
        legacy[image_id] = Path(path)
    try:
        if legacy:
            imported = import_legacy_reviews(project, session, legacy)
            print(f"imported {imported} legacy review decision(s)")
        manifest = finalize_project(
            project,
            session,
            args.output_dir,
            section_selection_path=selection if selection.is_file() else None,
            require_verified=args.require_verified,
        )
    except FinalizationError as exc:
        print(f"Finalization refused: {exc}", file=sys.stderr)
        return 2
    for image in manifest["images"]:
        counts = ", ".join(f"{key}={value}" for key, value in sorted(image["counts"].items()))
        print(f"{image['image_id']} [{image['verification']}]: {counts}")
        for warning in image["warnings"]:
            print(f"  warning: {warning}")
    print(f"finalized {len(manifest['images'])} image(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
