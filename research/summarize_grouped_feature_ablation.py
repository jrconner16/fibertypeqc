from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

LOWER_IS_BETTER = frozenset(
    {
        "iix_iib_symmetric_confusion_rate",
        "multiclass_log_loss",
        "multiclass_brier_score",
        "expected_calibration_error",
        "review_rate",
    }
)
NON_METRIC_NUMERIC_COLUMNS = frozenset({"fold_index", "train_rows", "test_rows", "feature_count"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Create paired-group bootstrap evidence for two feature-ablation conditions.")
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--comparator", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--comparison-output", type=Path, required=True)
    parser.add_argument("--decision-output", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paired_tables(
    metrics: pd.DataFrame,
    *,
    comparator: str,
    candidate: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    required = {"feature_set", "heldout_group_id"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"Metrics table missing required columns: {missing}")
    selected = metrics.loc[metrics["feature_set"].isin([comparator, candidate])].copy()
    if selected.duplicated(["feature_set", "heldout_group_id"]).any():
        raise ValueError("Metrics contain duplicate feature-set/group rows.")
    available = set(selected["feature_set"].astype(str))
    if available != {comparator, candidate}:
        raise ValueError(
            f"Expected feature sets {comparator!r} and {candidate!r}; found {sorted(available)}."
        )

    metric_columns = [
        column
        for column in selected.select_dtypes(include="number").columns
        if column not in NON_METRIC_NUMERIC_COLUMNS
    ]
    if not metric_columns:
        raise ValueError("Metrics table contains no numeric comparison metrics.")
    indexed = selected.set_index(["heldout_group_id", "feature_set"])
    first = indexed.xs(comparator, level="feature_set")[metric_columns].sort_index()
    second = indexed.xs(candidate, level="feature_set")[metric_columns].sort_index()
    if not first.index.equals(second.index):
        missing_from_candidate = sorted(set(first.index).difference(second.index))
        missing_from_comparator = sorted(set(second.index).difference(first.index))
        raise ValueError(
            "Comparator/candidate group sets differ: "
            f"missing from candidate={missing_from_candidate}, "
            f"missing from comparator={missing_from_comparator}."
        )
    if (
        not np.isfinite(first.to_numpy(dtype=float)).all()
        or not np.isfinite(second.to_numpy(dtype=float)).all()
    ):
        raise ValueError("Comparison metrics contain non-finite values.")
    return first, second, metric_columns


def paired_bootstrap_summary(
    metrics: pd.DataFrame,
    *,
    comparator: str,
    candidate: str,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    if repetitions < 1:
        raise ValueError("Bootstrap repetitions must be positive.")
    first, second, metric_columns = _paired_tables(
        metrics, comparator=comparator, candidate=candidate
    )
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(first), size=(repetitions, len(first)))
    rows: list[dict[str, object]] = []
    for metric in metric_columns:
        delta = second[metric].to_numpy(dtype=float) - first[metric].to_numpy(dtype=float)
        bootstrap_medians = np.median(delta[sample_indices], axis=1)
        direction = "lower" if metric in LOWER_IS_BETTER else "higher"
        favorable = delta <= 0 if direction == "lower" else delta >= 0
        rows.append(
            {
                "metric": metric,
                "direction_better": direction,
                "comparator": comparator,
                "candidate": candidate,
                "groups": len(delta),
                "comparator_median": float(first[metric].median()),
                "candidate_median": float(second[metric].median()),
                "paired_median_delta": float(np.median(delta)),
                "paired_mean_delta": float(np.mean(delta)),
                "paired_median_delta_ci025": float(np.quantile(bootstrap_medians, 0.025)),
                "paired_median_delta_ci975": float(np.quantile(bootstrap_medians, 0.975)),
                "favorable_groups": int(favorable.sum()),
            }
        )
    return pd.DataFrame(rows)


def build_decision_record(
    summary: pd.DataFrame,
    contract: dict[str, Any],
    *,
    comparator: str,
    candidate: str,
    repetitions: int,
    seed: int,
    metrics_sha256: str,
    contract_sha256: str,
) -> dict[str, Any]:
    primary_name = str(contract["evaluation"]["primary_metric"])
    if primary_name.startswith("per_mouse_"):
        primary_name = primary_name.removeprefix("per_mouse_")
    primary_rows = summary.loc[summary["metric"] == primary_name]
    if len(primary_rows) != 1:
        raise ValueError(f"Primary metric {primary_name!r} is absent or duplicated.")
    primary = primary_rows.iloc[0]
    return {
        "schema_version": 1,
        "contract_id": contract["contract_id"],
        "comparison": {"comparator": comparator, "candidate": candidate},
        "evidence_scope": "primary_development_grouped",
        "status": "baseline_characterization",
        "candidate_advancement_assessed": False,
        "candidate_advances": None,
        "reason": (
            "This record summarizes grouped development evidence only. The contract requires "
            "the declared transfer-development benchmark before an advancement decision."
        ),
        "protected_evaluation_read": False,
        "bootstrap": {"repetitions": repetitions, "seed": seed, "unit": "heldout_group"},
        "primary_metric": {
            "name": primary_name,
            "paired_median_delta": float(primary["paired_median_delta"]),
            "paired_median_delta_ci025": float(primary["paired_median_delta_ci025"]),
            "paired_median_delta_ci975": float(primary["paired_median_delta_ci975"]),
            "favorable_groups": int(primary["favorable_groups"]),
            "groups": int(primary["groups"]),
        },
        "input_sha256": {
            "metrics": metrics_sha256,
            "contract": contract_sha256,
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    for output in [args.comparison_output, args.decision_output]:
        if output.exists():
            raise SystemExit(f"Refusing to overwrite existing output: {output}")
    contract = yaml.safe_load(args.contract.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("Contract must be a YAML mapping.")
    bootstrap = contract["evaluation"]["uncertainty"]
    repetitions = int(bootstrap["repetitions"])
    seed = int(bootstrap["seed"])
    summary = paired_bootstrap_summary(
        pd.read_csv(args.metrics),
        comparator=args.comparator,
        candidate=args.candidate,
        repetitions=repetitions,
        seed=seed,
    )
    decision = build_decision_record(
        summary,
        contract,
        comparator=args.comparator,
        candidate=args.candidate,
        repetitions=repetitions,
        seed=seed,
        metrics_sha256=_sha256(args.metrics),
        contract_sha256=_sha256(args.contract),
    )
    args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
    args.decision_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.comparison_output, index=False)
    args.decision_output.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"paired groups: {int(summary['groups'].iloc[0])}")
    print(f"bootstrap repetitions: {repetitions}")
    print(f"saved comparison: {args.comparison_output}")
    print(f"saved decision record: {args.decision_output}")


if __name__ == "__main__":
    main()
