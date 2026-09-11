"""Summarize section-level locked predictions without using any test labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-dir", type=Path, required=True)
    parser.add_argument("--expected-sections", type=int, default=27)
    args = parser.parse_args()
    paths = sorted((args.scored_dir / "sections").glob("*_locked_summary.json"))
    if len(paths) != args.expected_sections:
        raise ValueError(f"Expected {args.expected_sections} section summaries; found {len(paths)}")
    sections = pd.DataFrame([json.loads(path.read_text()) for path in paths])
    pct = [column for column in sections if column.startswith("pct_")]
    group_columns = ["mouse_id", "cre_status", "analysis_eligible"]
    mice = sections.groupby(group_columns, as_index=False)[pct].mean()
    n_sections = sections.groupby(group_columns, as_index=False).size()
    mice = mice.merge(n_sections, on=group_columns, validate="one_to_one")
    sections.to_csv(args.scored_dir / "locked_section_composition.csv", index=False)
    mice.to_csv(args.scored_dir / "locked_mouse_mean_section_composition.csv", index=False)
    print(f"saved: {args.scored_dir / 'locked_mouse_mean_section_composition.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
