from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DISPARITY_CLASS_PAIRS: tuple[tuple[str, str], ...] = (
    ("iia", "iix"),
    ("iix", "iia"),
    ("iib", "iix"),
    ("iix", "iib"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a matched-ROI audit table that joins MyoSight ROI labels to pipeline fiber "
            "labels for images with existing ROI matching outputs."
        )
    )
    parser.add_argument(
        "--feature-table",
        type=Path,
        required=True,
        help="Candidate feature table CSV containing image_id, label, and fiber_type columns.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Validation manifest CSV with image_id and myosight_results columns.",
    )
    parser.add_argument(
        "--matched-dir",
        type=Path,
        required=True,
        help="Directory containing *_matched_fibers.csv ROI matching outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV for the full matched audit table.",
    )
    parser.add_argument(
        "--disparity-output",
        type=Path,
        default=None,
        help=(
            "Optional output CSV for a filtered disparity bucket limited to IIa<->IIx and "
            "IIb<->IIx matched-label mismatches."
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help=(
            "Optional candidate prediction CSV from train_candidate_from_feature_table. "
            "If provided, candidate predictions will be added per matched fiber."
        ),
    )
    return parser


def _normalize_text(value: str) -> str:
    return str(value).strip().lower()


def _normalize_image_id(value: str) -> str:
    return _normalize_text(value).replace(" ", "").replace("-", "_")


def _normalize_class_label(value: str) -> str:
    text = _normalize_text(value)
    aliases = {
        "iia": "iia",
        "iib": "iib",
        "iix": "iix",
        "type iia": "iia",
        "type iib": "iib",
        "type iix": "iix",
        "iia ": "iia",
        "iib ": "iib",
        "iix ": "iix",
    }
    return aliases.get(text, text)


def _resolve_manifest_results_path(manifest_path: Path, rel_path: str) -> Path:
    roots = [manifest_path.parent, *manifest_path.parents]
    seen: set[Path] = set()
    for root in roots:
        path = (root / rel_path).resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not resolve MyoSight results path: {rel_path}")


def _load_feature_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"image_id", "label", "fiber_type"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Feature table missing required columns: {sorted(missing)}")
    df = df.copy()
    df["_merge_image_id"] = df["image_id"].map(_normalize_image_id)
    return df


def _load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"image_id", "myosight_results"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    df = df.copy()
    df["_merge_image_id"] = df["image_id"].map(_normalize_image_id)
    return df


def _load_predictions(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    df = pd.read_csv(path)
    required = {"image_id", "label", "candidate_name", "predicted_fiber_type"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Predictions table missing required columns: {sorted(missing)}")
    pivot = (
        df.pivot_table(
            index=["image_id", "label"],
            columns="candidate_name",
            values="predicted_fiber_type",
            aggfunc="first",
        )
        .rename(columns=lambda c: f"candidate_pred_{c}")
        .reset_index()
    )
    pivot["_merge_image_id"] = pivot["image_id"].map(_normalize_image_id)
    return pivot.drop(columns=["image_id"])


def _matched_file_score(path: Path) -> tuple[float, int]:
    df = pd.read_csv(path)
    if df.empty or "area_ratio" not in df.columns:
        return (float("inf"), 0)
    ratios = pd.to_numeric(df["area_ratio"], errors="coerce").dropna()
    if ratios.empty:
        return (float("inf"), len(df))
    log_abs_median = float((ratios.map(lambda x: abs(__import__("math").log(x)))).median())
    return (log_abs_median, -len(df))


def _discover_best_matched_files(matched_dir: Path, manifest: pd.DataFrame) -> dict[str, Path]:
    matched_files = list(matched_dir.glob("*_matched_fibers.csv"))
    selected: dict[str, Path] = {}
    for _, row in manifest.iterrows():
        image_id = str(row["image_id"])
        norm = _normalize_image_id(image_id)
        candidates = [
            path
            for path in matched_files
            if norm in _normalize_image_id(path.stem.replace("_matched_fibers", ""))
        ]
        if not candidates:
            continue
        selected[norm] = sorted(candidates, key=_matched_file_score)[0]
    return selected


def build_matched_myosight_audit(
    feature_table: pd.DataFrame,
    manifest: pd.DataFrame,
    matched_files_by_image: dict[str, Path],
    manifest_path: Path,
    predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    feature_keep = [
        col
        for col in [
            "image_id",
            "label",
            "fiber_type",
            "fiber_type_source",
            "split",
            "needs_review",
            "model_confidence",
            "model_margin",
            "prob_iib",
            "prob_iia",
            "prob_iix",
        ]
        if col in feature_table.columns
    ]
    feature_subset = feature_table.loc[:, ["_merge_image_id", *feature_keep]].copy()

    if predictions is not None:
        feature_subset = feature_subset.merge(
            predictions,
            on=["_merge_image_id", "label"],
            how="left",
            validate="many_to_one",
        )

    for _, row in manifest.iterrows():
        norm = row["_merge_image_id"]
        matched_path = matched_files_by_image.get(norm)
        if matched_path is None:
            continue

        results_path = _resolve_manifest_results_path(manifest_path, str(row["myosight_results"]))
        results = pd.read_csv(results_path, sep="\t")
        results = results.loc[:, [c for c in results.columns if str(c).strip()]].copy()
        if "Label" not in results.columns:
            continue
        results = results.reset_index().rename(
            columns={"index": "roi_index", "Label": "myosight_label"}
        )
        results["myosight_label"] = results["myosight_label"].map(_normalize_class_label)

        matched = pd.read_csv(matched_path).copy()
        if matched.empty:
            continue
        matched["label"] = matched["label_id"].astype(int)
        matched["roi_index"] = matched["roi_index"].astype(int)
        matched["_merge_image_id"] = norm
        matched["matched_fibers_path"] = str(matched_path.resolve())

        merged = matched.merge(
            results.loc[:, ["roi_index", "myosight_label"]],
            on="roi_index",
            how="left",
            validate="many_to_one",
        ).merge(
            feature_subset,
            on=["_merge_image_id", "label"],
            how="left",
            validate="many_to_one",
        )

        merged["_merge_image_id"] = norm
        merged["image_id"] = str(row["image_id"])
        rows.append(merged)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out["pipeline_fiber_type"] = out["fiber_type"].map(_normalize_class_label)
    out["myosight_vs_pipeline_pair"] = (
        out["myosight_label"].fillna("unknown")
        + "_vs_"
        + out["pipeline_fiber_type"].fillna("unknown")
    )
    out["labels_agree"] = out["myosight_label"] == out["pipeline_fiber_type"]
    out["disparity_bucket"] = out["myosight_vs_pipeline_pair"].where(
        out.apply(
            lambda r: (r["myosight_label"], r["pipeline_fiber_type"]) in DISPARITY_CLASS_PAIRS,
            axis=1,
        ),
        other="",
    )
    ordered = [
        "image_id",
        "split",
        "roi_index",
        "label",
        "myosight_label",
        "pipeline_fiber_type",
        "myosight_vs_pipeline_pair",
        "disparity_bucket",
        "labels_agree",
        "centroid_distance_px",
        "area_ratio",
        "needs_review",
        "model_confidence",
        "model_margin",
        "prob_iib",
        "prob_iia",
        "prob_iix",
        "fiber_type_source",
    ]
    prediction_cols = sorted([c for c in out.columns if c.startswith("candidate_pred_")])
    remaining = [c for c in out.columns if c not in set(ordered) | set(prediction_cols)]
    return out.loc[:, [c for c in ordered if c in out.columns] + prediction_cols + remaining]


def main() -> None:
    args = build_parser().parse_args()
    feature_table = _load_feature_table(args.feature_table)
    manifest = _load_manifest(args.manifest)
    predictions = _load_predictions(args.predictions)
    matched_files_by_image = _discover_best_matched_files(args.matched_dir, manifest)
    audit = build_matched_myosight_audit(
        feature_table=feature_table,
        manifest=manifest,
        matched_files_by_image=matched_files_by_image,
        manifest_path=args.manifest,
        predictions=predictions,
    )
    if audit.empty:
        raise SystemExit("No matched ROI audit rows could be assembled.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output, index=False)

    disparity_output = args.disparity_output
    if disparity_output is not None:
        disparity = audit[(audit["disparity_bucket"] != "") & (~audit["labels_agree"])].copy()
        disparity_output.parent.mkdir(parents=True, exist_ok=True)
        disparity.to_csv(disparity_output, index=False)
        print(f"saved disparity bucket: {disparity_output}")
        print(f"disparity rows: {len(disparity)}")

    print(f"matched images: {audit['image_id'].nunique()}")
    print(f"rows: {len(audit)}")
    print(f"mismatches: {int((~audit['labels_agree']).sum())}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
