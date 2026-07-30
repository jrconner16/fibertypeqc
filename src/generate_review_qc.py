"""Generate headless QC tables and section selections for a review project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.review.project import load_project
from src.review.qc import generate_project_qc
from src.review.qc_rules import DEFAULT_RULES_PATH, load_rule_config
from src.review.section_selection import SelectionStrategy, select_sections
from src.review.storage import atomic_write_dataframe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument(
        "--selection-strategy",
        choices=[strategy.value for strategy in SelectionStrategy],
        default=SelectionStrategy.ALL_PASSING.value,
    )
    parser.add_argument(
        "--manual-selection",
        type=Path,
        help="YAML mouse/domain/image mapping required for populated manual selections.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="QC output directory; defaults to <project>/qc.",
    )
    return parser


def _load_manual_selections(path: Path | None) -> dict[str, dict[str, list[str]]] | None:
    if path is None:
        return None
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Manual selection file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Manual selection file is not valid YAML: {path}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Manual selection file must contain a mouse/domain mapping")
    return raw


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project = load_project(args.project, validate_paths=False)
    rules = load_rule_config(args.rules)
    manual = _load_manual_selections(args.manual_selection)
    qc_result = generate_project_qc(project, rules)
    selections = select_sections(
        project,
        qc_result.image_qc,
        strategy=args.selection_strategy,
        manual_selections=manual,
    )
    output_directory = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project.root / "qc"
    )
    outputs = {
        "image_qc": (output_directory / "image_qc.csv", qc_result.image_qc),
        "fiber_qc": (output_directory / "fiber_qc.csv", qc_result.fiber_qc),
        "nucleus_qc": (output_directory / "nucleus_qc.csv", qc_result.nucleus_qc),
        "section_selection": (
            output_directory / "section_selection.csv",
            selections,
        ),
    }
    for name, (path, table) in outputs.items():
        atomic_write_dataframe(path, table)
        print(f"{name}: {path} ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
