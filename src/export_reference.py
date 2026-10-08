"""Export blind reference labels from a review project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.review.project import load_project
from src.review.reference import export_reference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True, help="Review project YAML")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = export_reference(load_project(args.project), args.output_dir)
    except ValueError as exc:
        print(f"Reference export failed: {exc}", file=sys.stderr)
        return 1
    for reviewer, count in manifest["reviewers"].items():
        print(f"{reviewer}: {count} label(s)")
    print(
        f"{manifest['n_fibers_labelled']} fiber(s) labelled; "
        f"{manifest['n_exhaustive_fields']}/{manifest['n_fields']} drawn field(s) exhaustive"
    )
    print(f"reference: {args.output_dir / 'reference_labels.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
