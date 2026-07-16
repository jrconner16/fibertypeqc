from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import tifffile

from src.io_utils import save_dataframe
from src.quantify_classify import _feret_diameters_by_label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill Feret diameter columns into existing FiberTypeQC *_fibers.csv files "
            "using the matching *_cellpose_labels.tif masks."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing per-image FiberTypeQC outputs.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="**/*_cellpose_labels.tif",
        help="Glob used to discover label masks under --root.",
    )
    parser.add_argument(
        "--update-diagnostics",
        action="store_true",
        help="Also merge Feret columns into matching *_feature_diagnostics.csv files when present.",
    )
    return parser


def _fiber_paths_for_labels(labels_path: Path) -> tuple[Path, Path]:
    stem = labels_path.name.removesuffix("_cellpose_labels.tif")
    return (
        labels_path.with_name(f"{stem}_fibers.csv"),
        labels_path.with_name(f"{stem}_feature_diagnostics.csv"),
    )


def _feret_frame_for_fibers(labels_path: Path, fibers: pd.DataFrame) -> pd.DataFrame:
    labels = tifffile.imread(labels_path)
    label_ids = fibers["label"].astype(int).to_numpy()
    feret_max_px, feret_min_px = _feret_diameters_by_label(labels, label_ids)
    out = pd.DataFrame(
        {
            "label": label_ids,
            "feret_max_px": feret_max_px,
            "feret_min_px": feret_min_px,
        }
    )
    if {"pixel_size_x_um", "pixel_size_y_um"}.issubset(fibers.columns):
        x_um = pd.to_numeric(fibers["pixel_size_x_um"], errors="coerce")
        y_um = pd.to_numeric(fibers["pixel_size_y_um"], errors="coerce")
        if x_um.notna().all() and y_um.notna().all():
            feret_max_um, feret_min_um = _feret_diameters_by_label(
                labels,
                label_ids,
                pixel_size_x=float(x_um.iloc[0]),
                pixel_size_y=float(y_um.iloc[0]),
            )
            out["feret_max_um"] = feret_max_um
            out["feret_min_um"] = feret_min_um
    return out


def _merge_feret_columns(df: pd.DataFrame, feret: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if c not in set(feret.columns) - {"label"}]
    return df.loc[:, keep].merge(feret, on="label", how="left")


def backfill_root(root: Path, pattern: str, update_diagnostics: bool) -> tuple[int, int]:
    labels_paths = sorted(root.glob(pattern))
    updated = 0
    missing_fibers = 0
    for labels_path in labels_paths:
        fibers_path, diagnostics_path = _fiber_paths_for_labels(labels_path)
        if not fibers_path.exists():
            missing_fibers += 1
            continue
        fibers = pd.read_csv(fibers_path, low_memory=False)
        if fibers.empty or "label" not in fibers.columns:
            continue
        feret = _feret_frame_for_fibers(labels_path, fibers)
        save_dataframe(fibers_path, _merge_feret_columns(fibers, feret))
        if update_diagnostics and diagnostics_path.exists():
            diagnostics = pd.read_csv(diagnostics_path, low_memory=False)
            save_dataframe(diagnostics_path, _merge_feret_columns(diagnostics, feret))
        updated += 1
    return updated, missing_fibers


def main() -> None:
    args = build_parser().parse_args()
    updated, missing_fibers = backfill_root(
        args.root,
        args.glob,
        update_diagnostics=bool(args.update_diagnostics),
    )
    print(f"updated fiber tables: {updated}")
    print(f"labels without matching fibers csv: {missing_fibers}")


if __name__ == "__main__":
    main()
