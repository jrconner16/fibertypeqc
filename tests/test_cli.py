from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src import cli


def _project(tmp_path: Path, **overrides) -> Path:
    root = tmp_path / "proj"
    (root / "images").mkdir(parents=True)
    (root / "panel.yaml").write_text("channels: {}\n")
    (root / "samples.csv").write_text("image_id,mouse_id,raw_image_path\n")
    config = {
        "schema_version": "fibertypeqc_project.v1",
        "images": "images",
        "panel": "panel.yaml",
        "sample_sheet": "samples.csv",
        "model": "models/my_model.joblib",
    }
    config.update(overrides)
    (root / cli.CONFIG_NAME).write_text(yaml.safe_dump(config))
    return root


def _stage(root: Path) -> str:
    return cli.project_status(cli.load_project_folder(root))["stage"]


def test_config_paths_resolve_relative_to_the_project_folder(tmp_path):
    root = _project(tmp_path)

    project = cli.load_project_folder(root)

    assert project.images == root / "images"
    assert project.model == str(root / "models/my_model.joblib")
    assert project.batch_dir == root / "batch"
    assert project.review_project == root / "review" / "project.yaml"
    assert cli.load_project_folder(_project(tmp_path / "b", model="some_model_id")).model == (
        "some_model_id"
    )


def test_missing_or_invalid_config_is_reported(tmp_path, capsys):
    assert cli.main(["status", str(tmp_path)]) == 2
    assert "not a FiberTypeQC project folder" in capsys.readouterr().err
    root = _project(tmp_path, schema_version="other")
    assert cli.main(["status", str(root)]) == 2
    assert "schema_version" in capsys.readouterr().err


def test_status_follows_the_workflow_stages(tmp_path, capsys):
    root = _project(tmp_path)
    assert _stage(root) == "new"

    (root / "batch").mkdir()
    pd.DataFrame({"image_name": ["a", "b"], "status": ["failed", "failed"]}).to_csv(
        root / "batch/batch_summary.csv", index=False
    )
    assert _stage(root) == "batch_failed"
    pd.DataFrame({"image_name": ["a", "b"], "status": ["success", "failed"]}).to_csv(
        root / "batch/batch_summary.csv", index=False
    )
    assert _stage(root) == "batch_done"

    (root / "review/qc").mkdir(parents=True)
    (root / "review/project.yaml").write_text(yaml.safe_dump({"images": [{"image_id": "a"}]}))
    assert _stage(root) == "project_built"
    (root / "review/qc/fiber_qc.csv").write_text("fiber_id\n")
    assert _stage(root) == "ready_for_review"

    (root / "review/review").mkdir()
    state = root / "review/review/review_state.json"
    state.write_text(json.dumps({"object_decisions": [{}, {}], "regions": [{}]}))
    (root / "final").mkdir()
    manifest = root / "final/finalization_manifest.json"
    manifest.write_text("{}")
    now = time.time()
    os.utime(state, (now - 100, now - 100))
    os.utime(manifest, (now - 50, now - 50))
    assert _stage(root) == "finalized"
    (root / "results").mkdir()
    (root / "results/cohort_report.html").write_text("<html></html>")
    assert _stage(root) == "reported"
    os.utime(state, (now + 100, now + 100))  # reviewed again after finalizing
    assert _stage(root) == "finalize_outdated"

    assert cli.main(["status", str(root)]) == 0
    output = capsys.readouterr().out
    assert "2 fiber decision(s), 1 region(s)" in output
    assert f"python -m fibertypeqc finalize {root}" in output


def test_steps_call_scripts_with_project_paths(tmp_path, monkeypatch):
    root = _project(tmp_path)
    calls: list[tuple[str, list[str]]] = []

    def fake_run(module, arguments):
        calls.append((module, arguments))
        if module == "make_review_project":
            (root / "review").mkdir(exist_ok=True)
            (root / "review/project.yaml").write_text("images: []\n")
        return 0

    monkeypatch.setattr(cli, "_run", fake_run)

    assert cli.main(["run", str(root), "--reuse-artifacts", "auto"]) == 0
    module, arguments = calls[-1]
    assert module == "run_batch"
    assert arguments[arguments.index("--output-dir") + 1] == str(root / "batch")
    assert arguments[arguments.index("--model") + 1] == str(root / "models/my_model.joblib")
    assert "--split-czi-scenes" in arguments and arguments[-2:] == ["--reuse-artifacts", "auto"]

    assert cli.main(["prepare", str(root)]) == 2  # batch has not run
    (root / "batch").mkdir()
    (root / "batch/batch_summary.csv").write_text("image_name,status\na,success\n")
    assert cli.main(["prepare", str(root)]) == 0
    assert [name for name, _ in calls[-2:]] == ["make_review_project", "generate_review_qc"]
    assert cli.main(["prepare", str(root)]) == 0  # existing project is kept
    assert calls[-1][0] == "generate_review_qc" and calls[-2][0] == "generate_review_qc"

    assert cli.main(["finalize", str(root)]) == 0
    assert [name for name, _ in calls[-2:]] == ["finalize_review_project", "summarize_results"]
    assert cli.main(["finalize", str(root), "--unknown"]) == 2


@pytest.mark.parametrize("step", ["review"])
def test_review_requires_qc(tmp_path, capsys, step):
    root = _project(tmp_path)

    assert cli.main([step, str(root)]) == 2
    assert "QC has not run yet" in capsys.readouterr().err
