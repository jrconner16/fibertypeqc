"""Evaluate a conservative, grouped eMHC positive/negative/uncertain policy."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ("positive", "negative", "uncertain")
SCORE_COLUMNS = ("emhc.snr_p90", "emhc.coverage_high", "emhc.p90", "emhc.pctl")
GROUP_RE = re.compile(r"(?<!\d)(\d{6}_[A-Za-z]{1,3})(?![A-Za-z])")


def _infer_group(path: Path) -> str:
    match = GROUP_RE.search(str(path))
    if match is None:
        raise ValueError(f"Could not infer mouse/group ID from path: {path}")
    return match.group(1)


def _load_review_pair(review_path: Path) -> pd.DataFrame:
    diagnostic_paths = sorted(review_path.parent.glob("*_feature_diagnostics.csv"))
    if len(diagnostic_paths) != 1:
        raise ValueError(
            f"Expected exactly one feature diagnostics CSV beside {review_path}, "
            f"found {len(diagnostic_paths)}."
        )
    review = pd.read_csv(review_path)
    diagnostics = pd.read_csv(diagnostic_paths[0]).rename(columns={"label": "fiber_id"})
    required_review = {"fiber_id", "emhc_manual_label"}
    missing_review = required_review - set(review.columns)
    if missing_review:
        raise ValueError(f"{review_path} is missing columns: {sorted(missing_review)}")
    required_diagnostics = set(SCORE_COLUMNS) & set(diagnostics.columns)
    if not required_diagnostics:
        raise ValueError(f"{diagnostic_paths[0]} has no supported eMHC score columns")
    out = review[["fiber_id", "emhc_manual_label"]].merge(
        diagnostics[["fiber_id", *sorted(required_diagnostics)]],
        on="fiber_id",
        how="inner",
        validate="one_to_one",
    )
    out["emhc_manual_label"] = (
        out["emhc_manual_label"].fillna("").astype(str).str.strip().str.lower()
    )
    return out.loc[out["emhc_manual_label"].isin(LABELS)].copy()


def _discover(root: Path) -> pd.DataFrame:
    frames = []
    for review_path in sorted(root.rglob("manual_review.csv")):
        frame = _load_review_pair(review_path)
        try:
            group = _infer_group(review_path)
        except ValueError:
            diagnostic_paths = sorted(review_path.parent.glob("*_feature_diagnostics.csv"))
            group = _infer_group(diagnostic_paths[0])
        frame["group"] = group
        frame["review_path"] = str(review_path)
        frames.append(frame)
    if not frames:
        raise ValueError(f"No manual_review.csv files found below {root}")
    return pd.concat(frames, ignore_index=True)


def _policy(score: pd.Series, negative_cutoff: float, positive_cutoff: float) -> pd.Series:
    values = score.to_numpy(dtype=float)
    labels = np.full(values.shape, "uncertain", dtype=object)
    labels[np.isfinite(values) & (values <= negative_cutoff)] = "negative"
    labels[np.isfinite(values) & (values >= positive_cutoff)] = "positive"
    return pd.Series(labels, index=score.index)


def _metrics(frame: pd.DataFrame, prediction: pd.Series) -> dict[str, float | int]:
    gold = frame["emhc_manual_label"]
    certain = prediction.isin(("positive", "negative"))
    known = gold.isin(("positive", "negative"))
    negatives = gold.eq("negative")
    positives = gold.eq("positive")
    return {
        "n_labeled": int(len(frame)),
        "n_known_gold": int(known.sum()),
        "n_positive_gold": int(positives.sum()),
        "n_negative_gold": int(negatives.sum()),
        "n_uncertain_gold": int(gold.eq("uncertain").sum()),
        "predicted_positive": int(prediction.eq("positive").sum()),
        "predicted_negative": int(prediction.eq("negative").sum()),
        "predicted_uncertain": int(prediction.eq("uncertain").sum()),
        "uncertain_rate": float((~certain).mean()) if len(frame) else 0.0,
        "negative_false_positive_rate": float(
            (prediction.eq("positive") & negatives).sum() / max(int(negatives.sum()), 1)
        ),
        "positive_recall": float(
            (prediction.eq("positive") & positives).sum() / max(int(positives.sum()), 1)
        ),
        "negative_recall": float(
            (prediction.eq("negative") & negatives).sum() / max(int(negatives.sum()), 1)
        ),
        "certain_accuracy": float((prediction[known & certain] == gold[known & certain]).mean())
        if bool((known & certain).any())
        else 0.0,
    }


def _candidate_cutoffs(values: pd.Series) -> np.ndarray:
    finite = values[np.isfinite(values.to_numpy(dtype=float))].to_numpy(dtype=float)
    if finite.size == 0:
        return np.empty(0, dtype=float)
    return np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 101)))


def _select_policy(
    development: pd.DataFrame,
    false_positive_target: float,
) -> tuple[str, float, float, dict[str, float | int]]:
    best: tuple[tuple[float, float, float], str, float, float, dict[str, float | int]] | None = None
    for score_name in SCORE_COLUMNS:
        if score_name not in development.columns:
            continue
        cutoffs = _candidate_cutoffs(development[score_name])
        for negative_cutoff in cutoffs:
            for positive_cutoff in cutoffs:
                if positive_cutoff <= negative_cutoff:
                    continue
                prediction = _policy(development[score_name], negative_cutoff, positive_cutoff)
                metrics = _metrics(development, prediction)
                if metrics["negative_false_positive_rate"] > false_positive_target:
                    continue
                rank = (
                    float(metrics["positive_recall"]),
                    float(metrics["negative_recall"]),
                    -float(metrics["uncertain_rate"]),
                )
                if best is None or rank > best[0]:
                    best = (
                        rank,
                        score_name,
                        float(negative_cutoff),
                        float(positive_cutoff),
                        metrics,
                    )
    if best is None:
        raise ValueError("No conservative policy met the requested false-positive target.")
    _, score_name, negative_cutoff, positive_cutoff, metrics = best
    return score_name, negative_cutoff, positive_cutoff, metrics


def evaluate(
    root: Path,
    holdout_group: str,
    false_positive_target: float = 0.05,
) -> tuple[dict[str, object], pd.DataFrame]:
    frame = _discover(root)
    development = frame.loc[frame["group"] != holdout_group].copy()
    holdout = frame.loc[frame["group"] == holdout_group].copy()
    if development.empty or holdout.empty:
        groups = sorted(frame["group"].unique())
        raise ValueError(f"Need labeled development and holdout groups; found groups={groups}")
    score_name, negative_cutoff, positive_cutoff, development_metrics = _select_policy(
        development, false_positive_target
    )
    frame["policy_score"] = frame[score_name]
    frame["policy_prediction"] = _policy(frame[score_name], negative_cutoff, positive_cutoff)
    holdout_metrics = _metrics(
        holdout,
        frame.loc[holdout.index, "policy_prediction"],
    )
    report: dict[str, object] = {
        "policy_version": "emhc_conservative_v1",
        "score": score_name,
        "negative_cutoff": negative_cutoff,
        "positive_cutoff": positive_cutoff,
        "false_positive_target": false_positive_target,
        "development_groups": sorted(development["group"].unique().tolist()),
        "holdout_group": holdout_group,
        "development_metrics": development_metrics,
        "holdout_metrics": holdout_metrics,
        "groups": {
            group: {
                "n_labeled": int(len(group_frame)),
                "gold_counts": group_frame["emhc_manual_label"].value_counts().to_dict(),
            }
            for group, group_frame in frame.groupby("group")
        },
    }
    return report, frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--holdout-group", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--false-positive-target", type=float, default=0.05)
    args = parser.parse_args()
    report, predictions = evaluate(args.input_root, args.holdout_group, args.false_positive_target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.predictions_output is not None:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.predictions_output, index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
