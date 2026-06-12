from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd

from src.build_candidate_split_manifest import _infer_genotype, _infer_timepoint
from src.sweep_candidate_weights_on_cohort import (
    _discover_section_ids,
    _load_cohort_cache,
    _reclassify_fibers,
)

DEFAULT_DIRECT_ROOT = Path("outputs/myosight_validation_all_diagnostics")
DEFAULT_TRUSTED_SECTION_ROOT = Path(
    "outputs/myosight_validation_candidate_sections_trusted_baseline_gb_soft_iia_q001"
)
DEFAULT_REVIEWED = Path("outputs/validation/reviewed_audit_all.csv")
DEFAULT_MANIFEST = Path("outputs/validation/candidate_split_manifest.csv")

DEFAULT_CLASS_QUOTAS = {
    "iix": 80,
    "iib": 70,
    "iia": 50,
}
DEFAULT_BAND_FRACTIONS = {
    "low": 0.25,
    "threshold_band": 0.50,
    "high": 0.25,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a manual audit set for review-threshold calibration from the trusted "
            "32-image cohort and a candidate classifier."
        )
    )
    parser.add_argument("--classifier-path", type=Path, required=True)
    parser.add_argument("--myosight-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional summary text output. Defaults next to --output.",
    )
    parser.add_argument("--direct-root", type=Path, default=DEFAULT_DIRECT_ROOT)
    parser.add_argument(
        "--trusted-section-root",
        type=Path,
        default=DEFAULT_TRUSTED_SECTION_ROOT,
    )
    parser.add_argument("--reviewed", type=Path, default=DEFAULT_REVIEWED)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--display-output-root",
        type=Path,
        default=Path("outputs/myosight_validation_candidate_manual_high_myo_medium_baseline_light_reclass"),
        help=(
            "Existing output root with labels/fibers for Napari review. This can differ from "
            "the candidate being audited as long as segmentation/image IDs match."
        ),
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--margin-threshold", type=float, default=0.05)
    parser.add_argument("--low-confidence-cutoff", type=float, default=0.75)
    parser.add_argument("--high-confidence-cutoff", type=float, default=0.90)
    parser.add_argument("--margin-band-halfwidth", type=float, default=0.05)
    parser.add_argument("--confidence-band-halfwidth", type=float, default=0.05)
    parser.add_argument("--max-per-image", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--class-quotas",
        type=str,
        default="iix:80,iib:70,iia:50",
        help="Comma-separated per-class quotas, e.g. 'iix:80,iib:70,iia:50'.",
    )
    return parser


def _parse_class_quotas(raw: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(f"Expected class quota like 'iix:80', got {text!r}")
        label, quota = text.split(":", 1)
        key = label.strip().lower()
        out[key] = int(quota.strip())
    if not out:
        raise ValueError("At least one class quota is required.")
    return out


def _exclude_reviewed(df: pd.DataFrame, reviewed_path: Path | None) -> pd.DataFrame:
    if reviewed_path is None or not reviewed_path.exists():
        return df
    reviewed = pd.read_csv(reviewed_path, low_memory=False)
    if not {"image_id", "label"}.issubset(reviewed.columns):
        raise ValueError(f"{reviewed_path} must include image_id and label columns.")
    reviewed_keys = set(
        zip(
            reviewed["image_id"].astype(str),
            reviewed["label"].astype(int),
            strict=False,
        )
    )
    keep = ~df.apply(lambda row: (str(row["image_id"]), int(row["label"])) in reviewed_keys, axis=1)
    return df.loc[keep].copy()


def _quota_rows(class_quotas: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fiber_type, total in class_quotas.items():
        low = int(round(total * DEFAULT_BAND_FRACTIONS["low"]))
        band = int(round(total * DEFAULT_BAND_FRACTIONS["threshold_band"]))
        high = max(0, total - low - band)
        rows.extend(
            {
                "fiber_type": fiber_type,
                "calibration_band": calibration_band,
                "quota": quota,
            }
            for calibration_band, quota in (
                ("low", low),
                ("threshold_band", band),
                ("high", high),
            )
        )
    return pd.DataFrame(rows)


def _label_band(
    confidence: pd.Series,
    margin: pd.Series,
    *,
    confidence_threshold: float,
    margin_threshold: float,
    low_confidence_cutoff: float,
    high_confidence_cutoff: float,
    confidence_band_halfwidth: float,
    margin_band_halfwidth: float,
) -> pd.Series:
    confidence = pd.to_numeric(confidence, errors="coerce")
    margin = pd.to_numeric(margin, errors="coerce")

    low = confidence.lt(low_confidence_cutoff)
    threshold_band = confidence.sub(confidence_threshold).abs().le(confidence_band_halfwidth) | (
        margin.sub(margin_threshold).abs().le(margin_band_halfwidth)
    )
    high = confidence.ge(high_confidence_cutoff) & margin.ge(margin_threshold + margin_band_halfwidth)

    out = pd.Series("middle", index=confidence.index, dtype="object")
    out.loc[high] = "high"
    out.loc[low] = "low"
    out.loc[threshold_band & ~low & ~high] = "threshold_band"
    return out


def _sampling_priority(
    df: pd.DataFrame,
    *,
    confidence_threshold: float,
    margin_threshold: float,
) -> pd.Series:
    conf = pd.to_numeric(df["candidate_model_confidence"], errors="coerce")
    margin = pd.to_numeric(df["candidate_model_margin"], errors="coerce")
    band = df["calibration_band"].astype(str)

    distance = pd.Series(0.0, index=df.index, dtype=float)
    distance.loc[band.eq("threshold_band")] = (
        (conf.sub(confidence_threshold).abs() + margin.sub(margin_threshold).abs()).loc[band.eq("threshold_band")]
    )
    distance.loc[band.eq("low")] = (
        (confidence_threshold - conf).clip(lower=0.0) + (margin_threshold - margin).clip(lower=0.0)
    ).loc[band.eq("low")]
    distance.loc[band.eq("high")] = (
        (conf - confidence_threshold).clip(lower=0.0) + (margin - margin_threshold).clip(lower=0.0)
    ).loc[band.eq("high")]

    # Smaller is better near-threshold, larger is better for clear low/high exemplars.
    score = pd.Series(0.0, index=df.index, dtype=float)
    score.loc[band.eq("threshold_band")] = -distance.loc[band.eq("threshold_band")]
    score.loc[~band.eq("threshold_band")] = distance.loc[~band.eq("threshold_band")]
    return score


def _load_manifest_metadata(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, low_memory=False)
    if "image_id" not in manifest.columns:
        raise ValueError(f"{path} must include image_id.")
    out = manifest.copy()
    out["image_id"] = out["image_id"].astype(str)
    return out


def build_review_threshold_calibration_audit(
    *,
    classifier_path: Path,
    myosight_root: Path,
    direct_root: Path,
    trusted_section_root: Path,
    reviewed: Path | None,
    manifest_path: Path,
    display_output_root: Path,
    class_quotas: dict[str, int],
    confidence_threshold: float,
    margin_threshold: float,
    low_confidence_cutoff: float,
    high_confidence_cutoff: float,
    confidence_band_halfwidth: float,
    margin_band_halfwidth: float,
    max_per_image: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    section_ids = _discover_section_ids(trusted_section_root, [])
    cohort_cache = _load_cohort_cache(
        myosight_root=myosight_root,
        direct_root=direct_root,
        trusted_section_root=trusted_section_root,
        section_ids=section_ids,
    )
    model = joblib.load(classifier_path)
    manifest = _load_manifest_metadata(manifest_path)

    rows: list[pd.DataFrame] = []
    for image_id, entry in cohort_cache.items():
        fibers = _reclassify_fibers(entry["fibers"], entry["diagnostics"], model).copy()
        fibers["image_id"] = image_id
        fibers["input_kind"] = "section_tiff_export" if image_id in section_ids else "direct_czi"
        fibers["genotype"] = _infer_genotype(image_id)
        fibers["timepoint"] = _infer_timepoint(image_id)
        rows.append(
            fibers[
                [
                    c
                    for c in [
                        "image_id",
                        "label",
                        "fiber_type",
                        "model_confidence",
                        "model_margin",
                        "needs_review",
                        "has_signal_qc_warning",
                        "typing_signal_qc_flags",
                        "type1_signal_evidence",
                        "type2_signal_evidence",
                        "input_kind",
                        "genotype",
                        "timepoint",
                    ]
                    if c in fibers.columns
                ]
            ].copy()
        )
    pool = pd.concat(rows, ignore_index=True)
    pool = pool.rename(
        columns={
            "fiber_type": "candidate_pred_gated",
            "model_confidence": "candidate_model_confidence",
            "model_margin": "candidate_model_margin",
            "needs_review": "candidate_needs_review_default",
            "has_signal_qc_warning": "candidate_signal_warning",
        }
    )
    pool["candidate_pred_gated"] = pool["candidate_pred_gated"].fillna("").astype(str).str.lower()
    pool["label"] = pool["label"].astype(int)
    pool = pool.loc[pool["candidate_pred_gated"].isin(class_quotas)].copy()
    pool["candidate_needs_review_calibrated"] = (
        pd.to_numeric(pool["candidate_model_confidence"], errors="coerce").lt(confidence_threshold)
        | pd.to_numeric(pool["candidate_model_margin"], errors="coerce").lt(margin_threshold)
    )
    pool["calibration_band"] = _label_band(
        pool["candidate_model_confidence"],
        pool["candidate_model_margin"],
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        low_confidence_cutoff=low_confidence_cutoff,
        high_confidence_cutoff=high_confidence_cutoff,
        confidence_band_halfwidth=confidence_band_halfwidth,
        margin_band_halfwidth=margin_band_halfwidth,
    )
    pool["sampling_priority"] = _sampling_priority(
        pool,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
    )
    pool = _exclude_reviewed(pool, reviewed)
    pool = pool.merge(
        manifest[["image_id"] + [c for c in ["split", "source_image"] if c in manifest.columns]],
        on="image_id",
        how="left",
        validate="many_to_one",
    )
    if "split" not in pool.columns:
        pool["split"] = ""
    pool["review_display_output_root"] = str(display_output_root)

    quotas = _quota_rows(class_quotas)
    selected_keys: set[tuple[str, int]] = set()
    image_counts: Counter[str] = Counter()
    selected: list[pd.DataFrame] = []

    rng_offsets = {
        (row.fiber_type, row.calibration_band): idx
        for idx, row in enumerate(quotas.itertuples(index=False))
    }
    for row in quotas.itertuples(index=False):
        subset = pool.loc[
            pool["candidate_pred_gated"].eq(row.fiber_type)
            & pool["calibration_band"].eq(row.calibration_band)
        ].copy()
        if subset.empty or int(row.quota) <= 0:
            continue
        subset = subset.sample(frac=1.0, random_state=seed + rng_offsets[(row.fiber_type, row.calibration_band)])
        subset = subset.sort_values(
            ["sampling_priority", "candidate_model_confidence", "candidate_model_margin"],
            ascending=[False, True, True],
            kind="stable",
        )
        chosen_idx: list[int] = []
        for candidate in subset.itertuples():
            key = (str(candidate.image_id), int(candidate.label))
            if key in selected_keys:
                continue
            if image_counts[str(candidate.image_id)] >= max_per_image:
                continue
            chosen_idx.append(int(candidate.Index))
            selected_keys.add(key)
            image_counts[str(candidate.image_id)] += 1
            if len(chosen_idx) >= int(row.quota):
                break
        if chosen_idx:
            chosen = subset.loc[chosen_idx].copy()
            chosen["calibration_target_quota"] = int(row.quota)
            selected.append(chosen)

    if not selected:
        sampled = pool.iloc[0:0].copy()
    else:
        sampled = pd.concat(selected, ignore_index=True)

    sampled["audit_corrected_type"] = ""
    sampled["audit_is_uncertain"] = False
    sampled["audit_is_excluded"] = False
    sampled["audit_notes"] = ""
    sampled["manual_calibration_pool"] = "review_threshold_calibration"
    sampled["manual_calibration_thresholds"] = f"conf<{confidence_threshold:.2f} or margin<{margin_threshold:.2f}"
    sampled = sampled.sort_values(
        [
            "candidate_pred_gated",
            "calibration_band",
            "image_id",
            "sampling_priority",
            "label",
        ],
        ascending=[True, True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)

    summary = (
        sampled.groupby(["candidate_pred_gated", "calibration_band"], dropna=False)
        .agg(
            n_rows=("label", "size"),
            n_images=("image_id", "nunique"),
            mean_confidence=("candidate_model_confidence", "mean"),
            mean_margin=("candidate_model_margin", "mean"),
            calibrated_review_rate=("candidate_needs_review_calibrated", "mean"),
        )
        .reset_index()
    )
    return sampled, summary


def main() -> None:
    args = build_parser().parse_args()
    class_quotas = _parse_class_quotas(args.class_quotas)
    sampled, summary = build_review_threshold_calibration_audit(
        classifier_path=args.classifier_path,
        myosight_root=args.myosight_root,
        direct_root=args.direct_root,
        trusted_section_root=args.trusted_section_root,
        reviewed=args.reviewed,
        manifest_path=args.manifest,
        display_output_root=args.display_output_root,
        class_quotas=class_quotas,
        confidence_threshold=args.confidence_threshold,
        margin_threshold=args.margin_threshold,
        low_confidence_cutoff=args.low_confidence_cutoff,
        high_confidence_cutoff=args.high_confidence_cutoff,
        confidence_band_halfwidth=args.confidence_band_halfwidth,
        margin_band_halfwidth=args.margin_band_halfwidth,
        max_per_image=args.max_per_image,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(args.output, index=False)
    summary_output = (
        args.summary_output
        if args.summary_output is not None
        else args.output.with_name(args.output.stem + "_summary.txt")
    )
    with summary_output.open("w") as handle:
        handle.write(f"rows={len(sampled)}\n")
        handle.write(f"images={sampled['image_id'].nunique() if not sampled.empty else 0}\n\n")
        if not sampled.empty:
            handle.write("Counts by predicted class and calibration band:\n")
            handle.write(summary.to_string(index=False))
            handle.write("\n\nTop images:\n")
            handle.write(sampled["image_id"].value_counts().head(20).to_string())
    print(f"rows: {len(sampled)}")
    print(f"images: {sampled['image_id'].nunique() if not sampled.empty else 0}")
    print(f"saved: {args.output}")
    print(f"summary: {summary_output}")


if __name__ == "__main__":
    main()
