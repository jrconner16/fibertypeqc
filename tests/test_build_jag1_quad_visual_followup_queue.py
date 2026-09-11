from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from src.build_jag1_quad_visual_followup_queue import (
    STRATUM,
    build_visual_followup_queue,
)


def _section(tmp_path: Path, mouse: str, section: int, signal_channel: int) -> dict[str, str]:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[1:4, 1:4] = 1
    labels[1:4, 6:9] = 2
    labels[6:9, 1:4] = 3
    labels[6:9, 6:9] = 4
    raw = np.zeros((4, 10, 10), dtype=np.uint16)
    raw[signal_channel][labels == 1] = 100
    raw[signal_channel][labels == 2] = 80
    raw[signal_channel][labels == 3] = 60
    raw[signal_channel][labels == 4] = 40
    raw_path = tmp_path / f"{mouse}_{section}.tif"
    labels_path = tmp_path / f"{mouse}_{section}_labels.tif"
    tifffile.imwrite(raw_path, raw)
    tifffile.imwrite(labels_path, labels)
    return {
        "mouse_id": mouse,
        "cre_status": "cre_negative_mdx",
        "image_id": f"{mouse}_section_{section:02d}",
        "section_id": f"section-{section:02d}",
        "split": "development",
        "raw_image_path": str(raw_path),
        "fiber_labels_path": str(labels_path),
        "legacy_review_path": "",
    }


def test_visual_followup_is_section_spread_and_excludes_reviewed(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        [
            _section(tmp_path, "mouse_a", 1, 0),
            _section(tmp_path, "mouse_a", 2, 1),
            _section(tmp_path, "mouse_b", 1, 0),
            _section(tmp_path, "mouse_b", 2, 1),
        ]
    )
    queue = build_visual_followup_queue(
        manifest,
        {("mouse_a_section_01", 1), ("mouse_b_section_02", 1)},
        per_mouse_per_assay=2,
    )
    assert set(queue.sampling_stratum) == {STRATUM}
    assert "candidate_assay" not in queue
    assert set(queue.attrs["selection_audit"].candidate_assay) == {"type_i", "type_iia"}
    assert len(queue) == 8
    assert not set(zip(queue.image_id, queue.fiber_id, strict=False)).intersection(
        {("mouse_a_section_01", 1), ("mouse_b_section_02", 1)}
    )
    assert (queue.groupby("mouse_id").size() == 4).all()
    assert (queue.groupby(["mouse_id", "image_id"]).size() >= 1).all()
    assert queue.duplicated(["image_id", "fiber_id"]).sum() == 0
