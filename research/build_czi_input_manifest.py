"""Build a reviewable portable CSV manifest from explicitly filtered raw CZI files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def _image_id(relative_path: Path) -> str:
    """Create a readable stable ID while retaining parent sample identifiers."""
    value = "__".join((*relative_path.parts[:-1], relative_path.stem))
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def build_manifest_rows(
    input_root: Path, exclude_path_patterns: tuple[str, ...]
) -> list[dict[str, str]]:
    patterns = tuple(pattern.lower() for pattern in exclude_path_patterns)
    rows: list[dict[str, str]] = []
    for path in sorted(input_root.rglob("*.czi")):
        relative_path = path.relative_to(input_root)
        normalized = relative_path.as_posix().lower()
        if any(pattern in normalized for pattern in patterns):
            continue
        rows.append(
            {
                "image_id": _image_id(relative_path),
                "input_relpath": relative_path.as_posix(),
            }
        )
    if not rows:
        raise ValueError("No CZI files remain after exclusions.")
    image_ids = [row["image_id"] for row in rows]
    duplicates = sorted({image_id for image_id in image_ids if image_ids.count(image_id) > 1})
    if duplicates:
        raise ValueError(f"Generated duplicate image_id values: {', '.join(duplicates)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-path-pattern",
        action="append",
        default=[],
        help="Case-insensitive substring excluded from each relative input path; repeatable.",
    )
    args = parser.parse_args()
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")
    if not args.exclude_path_pattern:
        raise ValueError(
            "Provide at least one --exclude-path-pattern for a reviewable filtered manifest."
        )

    rows = build_manifest_rows(args.input_root, tuple(args.exclude_path_pattern))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "input_relpath"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} canonical candidates: {args.output}")
    for row in rows:
        print(f"{row['image_id']}: {row['input_relpath']}")


if __name__ == "__main__":
    main()
