from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.freeze_jag1_quad_candidate import freeze_candidate


def test_freeze_copies_candidate_and_records_no_final_test_input(tmp_path: Path) -> None:
    model = tmp_path / "model.joblib"
    metadata = tmp_path / "metadata.json"
    manifest = tmp_path / "manifest.csv"
    script = tmp_path / "train.py"
    review = tmp_path / "review.csv"
    for path, content in (
        (model, "model"),
        (metadata, "{}"),
        (manifest, "image_id\nimage\n"),
        (script, "print('train')\n"),
        (review, "image_id,fiber_id\nimage,1\n"),
    ):
        path.write_text(content)
    output = tmp_path / "lock"
    lock_path = freeze_candidate(
        model=model,
        metadata=metadata,
        manifest=manifest,
        training_script=script,
        review_sources=[review],
        output_dir=output,
    )
    lock = json.loads(lock_path.read_text())
    assert lock["final_test_status"] == "not read or used by this freeze operation"
    assert (output / "jag1_quad_four_class_random_forest_v2.joblib").is_file()
    with pytest.raises(FileExistsError):
        freeze_candidate(
            model=model,
            metadata=metadata,
            manifest=manifest,
            training_script=script,
            review_sources=[review],
            output_dir=output,
        )
