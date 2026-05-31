from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.fiber_type_labels import normalize_review_label

TYPE_ORDER = ("iib", "iia", "iix", "hybrid", "uncertain", "exclude")


def image_id_from_pipeline_path(path: Path) -> str:
    name = path.name
    for suffix in ("_fibers_reviewed.csv", "_fibers.csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def image_id_from_myosight_path(path: Path) -> str:
    parent = path.parent
    if parent.name.lower() == "results":
        return parent.parent.name
    return path.stem


def normalize_type(value: object) -> str:
    label = normalize_review_label(value)
    label = label.lower().replace(" ", "")
    mapping = {
        "iib": "iib",
        "ii-b": "iib",
        "type1": "iib",
        "iia": "iia",
        "ii-a": "iia",
        "type2": "iia",
        "iix": "iix",
        "ii-x": "iix",
        "unknown": "iix",
        "iixcandidate": "iix",
        "iix_candidate": "iix",
        "hybrid": "hybrid",
        "mixed": "hybrid",
        "uncertain": "uncertain",
        "exclude": "exclude",
    }
    return mapping.get(label, label)


def read_myosight(path: Path) -> pd.DataFrame:
    sep = "," if path.suffix.lower() == ".csv" else "\t"
    df = pd.read_csv(path, sep=sep)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed") or str(c).strip() == ""]
    df = df.drop(columns=unnamed, errors="ignore")
    if "Label" not in df.columns:
        raise ValueError(f"{path} has no Label column")
    df["label_norm"] = df["Label"].map(normalize_type)
    return df


def pipeline_label_column(df: pd.DataFrame) -> str:
    for col in ("final_type", "fiber_type", "predicted_biological_type", "predicted_type"):
        if col in df.columns:
            return col
    raise ValueError("Pipeline fiber CSV has no recognized type column")


def read_pipeline(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    col = pipeline_label_column(df)
    df["label_norm"] = df[col].map(normalize_type)
    return df


def counts_by_type(labels: pd.Series) -> dict[str, int]:
    counts = labels.value_counts(dropna=False).to_dict()
    return {typ: int(counts.get(typ, 0)) for typ in TYPE_ORDER}


def numeric_summary(series: pd.Series, prefix: str) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_p25": float("nan"),
            f"{prefix}_p75": float("nan"),
        }
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(values.median()),
        f"{prefix}_p25": float(values.quantile(0.25)),
        f"{prefix}_p75": float(values.quantile(0.75)),
    }


def pct(n: int, total: int) -> float:
    return float(n / total) if total else 0.0


def summarize_pair(image_id: str, myo_path: Path, pipe_path: Path) -> dict[str, object]:
    myo = read_myosight(myo_path)
    pipe = read_pipeline(pipe_path)
    myo_total = int(len(myo))
    pipe_total = int(len(pipe))
    myo_counts = counts_by_type(myo["label_norm"])
    pipe_counts = counts_by_type(pipe["label_norm"])

    row: dict[str, object] = {
        "image_id": image_id,
        "myosight_path": str(myo_path),
        "pipeline_path": str(pipe_path),
        "myosight_total_fibers": myo_total,
        "pipeline_total_fibers": pipe_total,
        "fiber_count_diff": pipe_total - myo_total,
        "fiber_count_ratio_pipeline_to_myosight": float(pipe_total / myo_total)
        if myo_total
        else float("nan"),
    }
    for typ in TYPE_ORDER:
        myo_n = myo_counts[typ]
        pipe_n = pipe_counts[typ]
        row[f"myosight_{typ}_n"] = myo_n
        row[f"pipeline_{typ}_n"] = pipe_n
        row[f"myosight_{typ}_pct"] = pct(myo_n, myo_total)
        row[f"pipeline_{typ}_pct"] = pct(pipe_n, pipe_total)
        row[f"{typ}_pct_diff_pipeline_minus_myosight"] = (
            row[f"pipeline_{typ}_pct"] - row[f"myosight_{typ}_pct"]
        )

    if "Area" in myo.columns:
        row.update(numeric_summary(myo["Area"], "myosight_area"))
        if "area" in pipe.columns:
            row.update(numeric_summary(pipe["area"], "pipeline_area_px"))
            if row["pipeline_area_px_median"]:
                implied_pixel_area = row["myosight_area_median"] / row["pipeline_area_px_median"]
                row["myosight_implied_pixel_area_um2"] = implied_pixel_area
                row["myosight_implied_pixel_size_um"] = implied_pixel_area**0.5
        pipeline_area_col = "area_um2" if "area_um2" in pipe.columns else "area"
        if pipeline_area_col in pipe.columns:
            row["pipeline_area_column"] = pipeline_area_col
            row.update(numeric_summary(pipe[pipeline_area_col], "pipeline_area"))
            row["area_median_ratio_pipeline_to_myosight"] = (
                row["pipeline_area_median"] / row["myosight_area_median"]
                if row["myosight_area_median"]
                else float("nan")
            )
        for erode_px in range(1, 11):
            eroded_col = f"area_erode_{erode_px}px_um2"
            if eroded_col not in pipe.columns:
                continue
            prefix = f"pipeline_area_erode_{erode_px}px"
            row.update(numeric_summary(pipe[eroded_col], prefix))
            row[f"area_erode_{erode_px}px_median_ratio_pipeline_to_myosight"] = (
                row[f"{prefix}_median"] / row["myosight_area_median"]
                if row["myosight_area_median"]
                else float("nan")
            )
    if "MinFeret" in myo.columns:
        row.update(numeric_summary(myo["MinFeret"], "myosight_minferet"))

    row["pipeline_needs_review_n"] = int(
        pipe.get("needs_review", pd.Series(False, index=pipe.index))
        .fillna(False)
        .astype(bool)
        .sum()
    )
    row["pipeline_signal_warning_n"] = int(
        pipe.get("has_signal_qc_warning", pd.Series(False, index=pipe.index))
        .fillna(False)
        .astype(bool)
        .sum()
    )
    return row


def discover_myosight(root: Path) -> dict[str, Path]:
    paths = sorted(root.rglob("Results/Results.txt")) + sorted(root.rglob("Results/Results.csv"))
    out: dict[str, Path] = {}
    for path in paths:
        image_id = image_id_from_myosight_path(path)
        out.setdefault(image_id, path)
    return out


def discover_pipeline(root: Path) -> dict[str, Path]:
    paths = sorted(root.rglob("*_fibers_reviewed.csv")) + sorted(root.rglob("*_fibers.csv"))
    out: dict[str, Path] = {}
    for path in paths:
        if "iia_signal_qc" in path.name or path.name == "random_audit_fibers.csv":
            continue
        image_id = image_id_from_pipeline_path(path)
        current = out.get(image_id)
        if current is None or ("reviewed" in path.name and "reviewed" not in current.name):
            out[image_id] = path
    return out


def natural_key(value: str) -> list[object]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", value)]


def canonical_image_id(value: str) -> str:
    return value.lower()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build matched MyoSight vs pipeline validation summary."
    )
    parser.add_argument("--myosight-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/validation/myosight_pipeline_image_summary.csv"),
    )
    parser.add_argument(
        "--image-id", action="append", default=[], help="Restrict to an image ID; repeatable."
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    myo_raw = discover_myosight(args.myosight_root)
    pipe_raw = discover_pipeline(args.pipeline_root)
    myo = {canonical_image_id(k): v for k, v in myo_raw.items()}
    pipe = {canonical_image_id(k): v for k, v in pipe_raw.items()}
    common = sorted(set(myo) & set(pipe), key=natural_key)
    if args.image_id:
        requested = {canonical_image_id(image_id) for image_id in args.image_id}
        common = [image_id for image_id in common if image_id in requested]
    if not common:
        raise SystemExit("No matched MyoSight/pipeline image IDs found.")

    rows = []
    for image_id in common:
        rows.append(
            summarize_pair(
                image_id_from_myosight_path(myo[image_id]),
                myo[image_id],
                pipe[image_id],
            )
        )
    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"saved: {args.output}")
    print(
        out[
            [
                "image_id",
                "myosight_total_fibers",
                "pipeline_total_fibers",
                "pipeline_needs_review_n",
                "pipeline_signal_warning_n",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
