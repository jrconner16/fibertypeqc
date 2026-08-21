"""Generate a self-contained HTML results/QC report from a result bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from fibertypeqc.html_report import generate_result_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Path to *_result_bundle.json.")
    parser.add_argument("--output", type=Path, help="Optional output HTML path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = generate_result_report(args.bundle, args.output)
    print(f"saved result report: {output}")


if __name__ == "__main__":
    main()
