from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build manifest of trusted section-export inputs from frozen summary files."
    )
    parser.add_argument(
        "--trusted-root",
        type=Path,
        default=Path("outputs/myosight_section_series_exports_normalized"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/validation/section_export_trusted_input_manifest.csv"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows: list[dict[str, object]] = []
    for summary_path in sorted(args.trusted_root.glob("*/*_summary.csv")):
        df = pd.read_csv(summary_path)
        if df.empty:
            continue
        row = df.iloc[0]
        rows.append(
            {
                "image_id": summary_path.parent.name,
                "input_path": row["input"],
                "trusted_summary_path": str(summary_path),
                "trusted_pipeline_fibers_path": row.get("fibers_path", ""),
            }
        )
    if not rows:
        raise SystemExit(f"No summary files found under {args.trusted_root}")
    out = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"saved: {args.output}")
    print(out[["image_id", "input_path"]].to_string(index=False))


if __name__ == "__main__":
    main()
