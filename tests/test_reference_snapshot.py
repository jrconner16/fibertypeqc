from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from src import cli
from src.reference_snapshot import build_snapshot, check, compare_snapshots, freeze


def _section(batch: Path, image_id: str, calls: list[str], *, shift: int = 0, model="m" * 64):
    out = batch / image_id
    out.mkdir(parents=True, exist_ok=True)
    labels = np.zeros((40, 40), dtype=np.int32)
    for index in range(len(calls)):
        labels[1 + shift + (index // 19) * 2, 1 + (index % 19) * 2] = index + 1
    tifffile.imwrite(out / f"{image_id}_cellpose_labels.tif", labels)
    pd.DataFrame({"label": range(1, len(calls) + 1), "fiber_type": calls}).to_csv(
        out / f"{image_id}_fibers.csv", index=False
    )
    (out / f"{image_id}_run.json").write_text(
        json.dumps({"classifier_sha256": model, "segmentation": {"device": "cpu"}})
    )


def _batch(tmp_path: Path, name: str, sections: dict[str, dict]) -> Path:
    batch = tmp_path / name
    for image_id, options in sections.items():
        _section(batch, image_id, **options)
    pd.DataFrame({"image_name": list(sections), "status": "success"}).to_csv(
        batch / "batch_summary.csv", index=False
    )
    return batch


CALLS = ["iib"] * 80 + ["iix"] * 15 + ["iia"] * 5


def test_identical_run_matches_exactly_and_snapshot_is_not_overwritten(tmp_path):
    batch = _batch(tmp_path, "a", {"s1": {"calls": CALLS}})
    path = tmp_path / "snapshot.json"

    snapshot = freeze(batch, path)

    assert snapshot["sections"][0]["class_counts"] == {"iia": 5, "iib": 80, "iix": 15}
    results = check(batch, path)
    assert [(r.level, r.passed) for r in results] == [("exact", True)]
    with pytest.raises(ValueError, match="not overwritten"):
        freeze(batch, path)


def test_same_mask_with_different_calls_fails_exactly(tmp_path):
    reference = build_snapshot(_batch(tmp_path, "a", {"s1": {"calls": CALLS}}))
    changed = CALLS.copy()
    changed[0] = "iix"
    current = build_snapshot(_batch(tmp_path, "b", {"s1": {"calls": changed}}))

    (result,) = compare_snapshots(reference, current)

    assert (result.level, result.passed) == ("exact", False)
    assert "fiber calls differ" in result.detail


def test_resegmented_sections_use_tolerances(tmp_path):
    reference = build_snapshot(_batch(tmp_path, "a", {"s1": {"calls": CALLS}}))
    close = ["iib"] * 80 + ["iix"] * 16 + ["iia"] * 5  # one extra fiber, about one point shift
    far = ["iib"] * 60 + ["iix"] * 35 + ["iia"] * 5
    near_run = build_snapshot(_batch(tmp_path, "b", {"s1": {"calls": close, "shift": 1}}))
    far_run = build_snapshot(_batch(tmp_path, "c", {"s1": {"calls": far, "shift": 1}}))

    (near,) = compare_snapshots(reference, near_run)
    (far_result,) = compare_snapshots(reference, far_run)
    (strict,) = compare_snapshots(reference, near_run, count_tolerance=0.001)

    assert (near.level, near.passed) == ("tolerant", True)
    assert (far_result.level, far_result.passed) == ("tolerant", False)
    assert not strict.passed


def test_missing_section_and_different_model_fail(tmp_path):
    reference = build_snapshot(
        _batch(tmp_path, "a", {"s1": {"calls": CALLS}, "s2": {"calls": CALLS}})
    )
    current = build_snapshot(_batch(tmp_path, "b", {"s1": {"calls": CALLS, "model": "x" * 64}}))

    results = {r.image_id: r for r in compare_snapshots(reference, current)}

    assert not results["s1"].passed and "different model" in results["s1"].detail
    assert (results["s2"].level, results["s2"].passed) == ("missing", False)


def test_snapshot_command_freezes_and_checks_a_project(tmp_path, capsys):
    root = tmp_path / "proj"
    (root / "images").mkdir(parents=True)
    (root / "panel.yaml").write_text("channels: {}\n")
    (root / "samples.csv").write_text("image_id,mouse_id,raw_image_path\n")
    (root / cli.CONFIG_NAME).write_text(
        "schema_version: fibertypeqc_project.v1\nimages: images\npanel: panel.yaml\n"
        "sample_sheet: samples.csv\n"
    )
    _batch(root, "batch", {"s1": {"calls": CALLS}})

    assert cli.main(["snapshot", str(root), "--freeze"]) == 0
    assert cli.main(["snapshot", str(root), "--check"]) == 0
    assert "1/1 section(s) match the reference" in capsys.readouterr().out
    _section(root / "batch", "s1", ["iix"] + CALLS[1:])
    assert cli.main(["snapshot", str(root), "--check"]) == 1
    assert cli.main(["snapshot", str(root), "--check", "--file", str(tmp_path / "none.json")]) == 2
