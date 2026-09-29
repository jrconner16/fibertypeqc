from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replace rows in a base candidate summary with trusted section rerun rows."
    )
    parser.add_argument("--base-summary", type=Path, required=True)
    parser.add_argument("--replacement-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = pd.read_csv(args.base_summary)
    repl = pd.read_csv(args.replacement_summary)
    if "image_id" not in base.columns or "image_id" not in repl.columns:
        raise SystemExit("Both summaries must contain image_id.")
    replacement_ids = set(repl["image_id"].astype(str))
    out = pd.concat(
        [base.loc[~base["image_id"].astype(str).isin(replacement_ids)].copy(), repl.copy()],
        ignore_index=True,
    )
    out = out.sort_values("image_id").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"saved: {args.output}")
    print(f"replaced rows: {len(replacement_ids)}")


if __name__ == "__main__":
    main()
