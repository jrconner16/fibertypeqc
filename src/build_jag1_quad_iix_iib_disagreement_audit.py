"""Build a blinded development re-review queue for observed IIx/IIb disagreements."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.build_jag1_quad_review_setup import QUEUE_COLUMNS, _queue_row

QUEUE_ID = "jamie_development_iix_iib_disagreement_audit"
STRATUM = "development_iix_iib_disagreement_audit"


def build_queue(
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    mouse_id: str = "351545_L",
    maximum: int = 36,
) -> pd.DataFrame:
    """Select fixed development disagreements without exposing either label in the queue."""
    required_predictions = {"mouse_id", "image_id", "fiber_id", "label", "lomo_prediction"}
    missing = sorted(required_predictions - set(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing columns: {missing}")
    required_manifest = {
        "mouse_id",
        "image_id",
        "split",
        "section_id",
        "raw_image_path",
        "fiber_labels_path",
    }
    missing = sorted(required_manifest - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    candidates = predictions.loc[
        predictions["mouse_id"].eq(mouse_id)
        & predictions["label"].astype(str).str.lower().eq("iix")
        & predictions["lomo_prediction"].astype(str).str.lower().eq("iib")
    ].copy()
    if candidates.empty:
        raise ValueError("No requested IIx-to-IIb development disagreements were found")
    sections = manifest.loc[
        manifest["split"].eq("development") & manifest["mouse_id"].eq(mouse_id)
    ].set_index("image_id", drop=False)
    if sections.empty:
        raise ValueError(f"No development manifest sections found for {mouse_id}")
    records: list[dict[str, object]] = []
    for row in (
        candidates.sort_values(["image_id", "fiber_id"], kind="stable")
        .head(maximum)
        .itertuples(index=False)
    ):
        if str(row.image_id) not in sections.index:
            raise ValueError(f"Disagreement has no eligible manifest section: {row.image_id}")
        records.append(
            _queue_row(sections.loc[str(row.image_id)], int(row.fiber_id), STRATUM, QUEUE_ID)
        )
    queue = pd.DataFrame(records, columns=QUEUE_COLUMNS)
    if queue.duplicated(["image_id", "fiber_id"]).any():
        raise AssertionError("Disagreement audit queue contains duplicate fibers")
    queue["provenance"] = "blinded_development_re-review; prior_label_and_model_prediction_hidden"
    return queue


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build blinded Jag1 Quad IIx/IIb disagreement audit"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lomo-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mouse-id", default="351545_L")
    parser.add_argument("--max-fibers", type=int, default=36)
    args = parser.parse_args()
    if args.max_fibers < 1:
        raise ValueError("--max-fibers must be at least 1")
    queue = build_queue(
        pd.read_csv(args.manifest, dtype=str).fillna(""),
        pd.read_csv(args.lomo_predictions, dtype={"fiber_id": "Int64"}).fillna(""),
        mouse_id=args.mouse_id,
        maximum=args.max_fibers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(args.output, index=False)
    print(
        queue.groupby(["mouse_id", "image_id", "sampling_stratum"]).size().to_string(name="count")
    )
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
