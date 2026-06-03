from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.quantify_classify import (
    FROZEN_ALPHA_BASELINE_FEATURES,
    MarkerSpec,
    build_feature_table,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare frozen alpha model features against the experimental feature builder."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("data/models/rebaseline_tile_v2_p75p90_iib_iia_iix.joblib"),
        help="Path to the frozen alpha classifier.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for the comparison table.",
    )
    return parser


def _synthetic_feature_frame() -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray | float]]]:
    df = pd.DataFrame(
        {
            "area": [10, 20],
            "type1_mean": [1.0, 2.0],
            "type2_mean": [3.0, 4.0],
            "type1_p75": [1.5, 2.5],
            "type2_p75": [3.5, 4.5],
            "type1_p90": [1.8, 2.8],
            "type2_p90": [3.8, 4.8],
            "type1_pctl": [1.7, 2.7],
            "type2_pctl": [3.7, 4.7],
            "type1_coverage": [0.2, 0.3],
            "type2_coverage": [0.4, 0.5],
        }
    )
    marker_stats = {
        "iib": {
            "mean": np.array([1.0, 2.0], dtype=np.float32),
            "p75": np.array([1.5, 2.5], dtype=np.float32),
            "p90": np.array([1.8, 2.8], dtype=np.float32),
            "pctl": np.array([1.7, 2.7], dtype=np.float32),
            "coverage": np.array([0.2, 0.3], dtype=np.float32),
            "tissue_median": 0.5,
            "tissue_mad": 0.25,
        },
        "iia": {
            "mean": np.array([3.0, 4.0], dtype=np.float32),
            "p75": np.array([3.5, 4.5], dtype=np.float32),
            "p90": np.array([3.8, 4.8], dtype=np.float32),
            "pctl": np.array([3.7, 4.7], dtype=np.float32),
            "coverage": np.array([0.4, 0.5], dtype=np.float32),
            "tissue_median": 1.0,
            "tissue_mad": 0.5,
        },
        "i": {
            "mean": np.array([5.0, 6.0], dtype=np.float32),
            "p75": np.array([5.5, 6.5], dtype=np.float32),
            "p90": np.array([5.8, 6.8], dtype=np.float32),
            "pctl": np.array([5.7, 6.7], dtype=np.float32),
            "coverage": np.array([0.1, 0.2], dtype=np.float32),
            "tissue_median": 0.8,
            "tissue_mad": 0.4,
        },
        "iix": {
            "mean": np.array([7.0, 8.0], dtype=np.float32),
            "p75": np.array([7.5, 8.5], dtype=np.float32),
            "p90": np.array([7.8, 8.8], dtype=np.float32),
            "pctl": np.array([7.7, 8.7], dtype=np.float32),
            "coverage": np.array([0.6, 0.7], dtype=np.float32),
            "tissue_median": 1.2,
            "tissue_mad": 0.6,
        },
    }
    return df, marker_stats


def main() -> None:
    args = build_parser().parse_args()

    model = joblib.load(args.model_path)
    model_features = tuple(getattr(model, "feature_names_in_", ()))
    synthetic_df, synthetic_marker_stats = _synthetic_feature_frame()
    experimental = build_feature_table(
        synthetic_df,
        marker_specs=(
            MarkerSpec(marker_name="iib", legacy_prefix="type1", channel_index=0),
            MarkerSpec(marker_name="iia", legacy_prefix="type2", channel_index=1),
        ),
        marker_stats_metadata=synthetic_marker_stats,
    )

    frozen_model_set = set(model_features)
    frozen_code_set = set(FROZEN_ALPHA_BASELINE_FEATURES)
    experimental_set = set(experimental.columns)
    all_features = sorted(frozen_model_set | frozen_code_set | experimental_set)

    rows = []
    for name in all_features:
        rows.append(
            {
                "feature_name": name,
                "in_frozen_model": name in frozen_model_set,
                "in_frozen_code_list": name in frozen_code_set,
                "in_experimental_builder": name in experimental_set,
                "status": (
                    "baseline"
                    if name in frozen_model_set
                    else "experimental_only"
                    if name in experimental_set
                    else "code_only"
                ),
            }
        )

    out = pd.DataFrame(rows)
    print(f"Frozen model features: {len(model_features)}")
    print(f"Frozen code baseline features: {len(FROZEN_ALPHA_BASELINE_FEATURES)}")
    print(f"Experimental builder features: {len(experimental.columns)}")
    print(
        f"Experimental-only features: "
        f"{int((out['status'] == 'experimental_only').sum())}"
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
        print(f"saved: {args.output}")
    else:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
