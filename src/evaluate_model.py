"""Score one chosen built model on the project's test mice (independent evaluation).

Choose the candidate first (normally the one 'build' recommended); only that candidate is scored.
Nothing is re-typed and the project's model is not changed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.model_builder import BuilderError
from src.model_evaluation import EvaluationError, evaluate
from src.review.project import load_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True, help="Project folder")
    parser.add_argument("--candidate", required=True, help="Candidate name or model file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.project_root.expanduser().resolve()
    try:
        project = load_project(root / "review" / "project.yaml")
        result = evaluate(root, project, args.candidate)
    except (EvaluationError, BuilderError, ValueError) as exc:
        print(f"Evaluation stopped: {exc}", file=sys.stderr)
        return 1
    labels = result["labels"]
    print(
        f"{result['model_id']} on {len(result['test_mice'])} test mice "
        f"({labels['n_fibers']} labelled fibers; reviewer conflicts {labels['conflicting_fibers']})"
    )
    recommended = result["recommended_by_builder"]
    if recommended is False:
        print("Note: 'build' did not recommend this candidate.")
    print(f"median per-mouse macro-F1: {result['median_mouse_macro_f1']:.3f}")
    for name, value in result["pooled_recall_by_class"].items():
        print(f"  recall {name}: {'N/A' if value is None else f'{value:.2f}'}")
    uses = result["test_set_uses"]
    if uses > 1:
        print(
            f"\nWarning: these test mice have now been used for {uses} evaluations. Choosing "
            "between models by their test scores makes the test no longer independent."
        )
    print(f"\n{result['caveat']}")
    print(f"evaluation folder: {result['folder']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
