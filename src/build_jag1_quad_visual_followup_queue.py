"""Build a blinded, assay-signal candidate follow-up for Jag1 Quad development.

This does not fit, score, or alter a classifier.  It produces a new development
queue after the initial random and model-informed review is complete.  Candidate
selection uses only the observed Type-I and Type-IIa assay channels, ranks
within each section, and is intentionally hidden from the reviewer.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.segmentation import find_boundaries

from src.build_jag1_quad_review_setup import QUEUE_COLUMNS, _legacy_ids, _queue_row
from src.io_utils import load_multichannel_image

ASSAY_CHANNELS = {"type_i": 0, "type_iia": 1}
QUEUE_ID = "jamie_development_visual_followup"
STRATUM = "development_targeted_visual_followup"


def _reviewed_keys(path: Path) -> set[tuple[str, int]]:
    table = pd.read_csv(path, dtype={"fiber_id": "Int64"}).fillna("")
    required = {"image_id", "fiber_id"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"--reviewed-decisions is missing columns: {missing}")
    return {
        (str(row.image_id), int(row.fiber_id))
        for row in table.itertuples(index=False)
        if pd.notna(row.fiber_id)
    }


def _signal_ranked_ids(
    section: pd.Series, channel: int, excluded: set[tuple[str, int]]
) -> list[int]:
    """Return section-local assay-signal candidates, strongest first.

    The one-pixel inner boundary is discarded so laminin and adjacent-fiber
    bleed-through do not determine a fiber's assay rank.  Ranking is within
    section only, avoiding cross-image exposure or intensity-scale effects.
    """
    raw = load_multichannel_image(Path(str(section.raw_image_path)))
    if raw.shape[0] != 4:
        raise ValueError(
            f"Expected four observed channels for {section.image_id}; got {raw.shape[0]}"
        )
    labels = np.asarray(tifffile.imread(str(section.fiber_labels_path)), dtype=np.int32)
    if raw.shape[1:] != labels.shape:
        raise ValueError(f"Raw/label shape mismatch for {section.image_id}")
    core = labels.copy()
    core[find_boundaries(labels, mode="inner")] = 0
    counts = np.bincount(core.ravel())
    sums = np.bincount(core.ravel(), weights=raw[channel].ravel())
    eligible = np.flatnonzero(counts > 0)
    eligible = eligible[eligible > 0]
    values = sums[eligible] / counts[eligible]
    ranked = pd.DataFrame({"fiber_id": eligible, "signal": values})
    ranked = ranked[~ranked.fiber_id.map(lambda fid: (str(section.image_id), int(fid)) in excluded)]
    return (
        ranked.sort_values(["signal", "fiber_id"], ascending=[False, True])
        .fiber_id.astype(int)
        .tolist()
    )


def _round_robin_take(candidates: dict[str, list[int]], count: int) -> list[tuple[str, int]]:
    """Take high-ranked candidates while spreading them across available sections."""
    queues = {image_id: deque(ids) for image_id, ids in candidates.items()}
    selected: list[tuple[str, int]] = []
    while len(selected) < count:
        made_progress = False
        for image_id in sorted(queues):
            if not queues[image_id] or len(selected) == count:
                continue
            selected.append((image_id, queues[image_id].popleft()))
            made_progress = True
        if not made_progress:
            break
    return selected


def build_visual_followup_queue(
    manifest: pd.DataFrame,
    reviewed: set[tuple[str, int]],
    per_mouse_per_assay: int = 20,
) -> pd.DataFrame:
    """Build the post-initial-review development queue with no model inputs."""
    if per_mouse_per_assay < 1:
        raise ValueError("--per-mouse-per-assay must be at least 1")
    selected: list[dict[str, object]] = []
    for _mouse, sections in manifest[manifest.split.eq("development")].groupby(
        "mouse_id", sort=True
    ):
        available = sections[
            sections.fiber_labels_path.astype(str).map(lambda p: Path(p).is_file())
        ]
        excluded = set(reviewed)
        for _, section in available.iterrows():
            excluded.update(
                (str(section.image_id), fid)
                for fid in _legacy_ids(Path(str(section.legacy_review_path)))
            )
        by_image = {str(section.image_id): section for _, section in available.iterrows()}
        for _assay, channel in ASSAY_CHANNELS.items():
            ranked = {
                image_id: _signal_ranked_ids(section, channel, excluded)
                for image_id, section in by_image.items()
            }
            for image_id, fiber_id in _round_robin_take(ranked, per_mouse_per_assay):
                section = by_image[image_id]
                record = _queue_row(section, fiber_id, STRATUM, QUEUE_ID)
                # Keep this out of the reviewer queue.  It is only an audit trail
                # showing why a fiber was sampled, not a candidate label.
                record["candidate_assay"] = _assay
                selected.append(record)
                excluded.add((image_id, fiber_id))
    queue = pd.DataFrame(selected).sort_values(["mouse_id", "image_id", "fiber_id"], kind="stable")
    if queue.duplicated(["image_id", "fiber_id"]).any():
        raise AssertionError("Visual follow-up queue contains duplicate fibers")
    queue["provenance"] = (
        "development_only; observed_assay_signal_ranked_within_section; "
        "candidate_assay_hidden_from_reviewer"
    )
    audit = queue.loc[
        :, ["mouse_id", "cre_status", "image_id", "section_id", "fiber_id", "candidate_assay"]
    ].copy()
    queue = queue.loc[:, QUEUE_COLUMNS].copy()
    queue.attrs["selection_audit"] = audit
    return queue


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Jag1 Quad visual follow-up queue")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reviewed-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-mouse-per-assay", type=int, default=20)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, dtype=str).fillna("")
    required = {
        "mouse_id",
        "image_id",
        "split",
        "fiber_labels_path",
        "raw_image_path",
        "legacy_review_path",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"--manifest is missing columns: {missing}")
    queue = build_visual_followup_queue(
        manifest,
        _reviewed_keys(args.reviewed_decisions),
        args.per_mouse_per_assay,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(args.output, index=False)
    audit_path = args.output.with_name(f"{args.output.stem}_selection_audit.csv")
    queue.attrs["selection_audit"].to_csv(audit_path, index=False)
    print(
        queue.groupby(["mouse_id", "cre_status", "sampling_stratum"]).size().to_string(name="count")
    )
    print(f"saved: {args.output}")
    print(f"private candidate-selection audit: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
