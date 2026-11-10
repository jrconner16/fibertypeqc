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


def _images(tmp_path: Path, names=("m1_s1", "m2_s1")) -> Path:
    import numpy as np
    import tifffile

    folder = tmp_path / "raw"
    folder.mkdir()
    for name in names:
        tifffile.imwrite(
            folder / f"{name}.tif",
            np.zeros((4, 8, 8), dtype=np.uint16),
            imagej=True,
            metadata={"axes": "CYX"},
        )
    (folder / "notes.txt").write_text("not an image")
    return folder


def test_init_unattended_writes_config_panel_and_sample_sheet(tmp_path, capsys):
    images = _images(tmp_path)
    root = tmp_path / "study"

    code = cli.main(
        ["init", str(root), "--images", str(images), "--panel", "four_marker_i_iia_laminin_iib",
         "--model", "quad_four_class_rf_v1", "--mouse-id-pattern", r"^(m\d+)_", "--yes"]
    )  # fmt: skip

    assert code == 0
    project = cli.load_project_folder(root)
    assert project.images == images and project.model == "quad_four_class_rf_v1"
    panel = yaml.safe_load((root / "panel.yaml").read_text())
    assert panel["channels"]["type_iib"] == 3
    assert panel["classification"]["residual_inference"]["target_class"] == "iix"
    sheet = pd.read_csv(root / "samples.csv", keep_default_na=False)
    assert sheet["image_id"].tolist() == ["m1_s1", "m2_s1"]
    assert sheet["mouse_id"].tolist() == ["m1", "m2"]
    assert "fill in mouse_id" not in capsys.readouterr().out
    assert cli.main(["init", str(root), "--images", str(images), "--yes"]) == 2  # already exists


def test_init_interactive_custom_panel(tmp_path, monkeypatch, capsys):
    images = _images(tmp_path, names=("slide",))
    answers = iter(
        [str(images), "3", "3", "1", "2", "4", "", "", "", "/models/quad.joblib"]
    )  # images, custom panel, laminin, I, IIa, IIb (counted from 1), IIx, DAPI, eMHC, model
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    root = tmp_path / "study"

    assert cli.main(["init", str(root)]) == 0

    output = capsys.readouterr().out
    assert "slide.tif has 4 channel(s), numbered 1 to 4" in output
    assert "numbered 0" not in output
    assert "fill in mouse_id for 1 image(s)" in output
    panel = yaml.safe_load((root / "panel.yaml").read_text())
    assert panel["channels"] == {
        "laminin": 2, "dapi": None, "type_i": 0, "type_iia": 1, "type_iib": 3, "type_iix": None,
        "emhc": None,
    }  # fmt: skip
    assert panel["classification"]["residual_inference"]["requires_negative_markers"] == [
        "i", "iia", "iib",
    ]  # fmt: skip
    assert cli.load_project_folder(root).model == "/models/quad.joblib"


def test_init_reports_missing_inputs(tmp_path, capsys):
    assert cli.main(["init", str(tmp_path / "a"), "--yes"]) == 2
    assert "--images is required" in capsys.readouterr().err
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.main(["init", str(tmp_path / "b"), "--images", str(empty), "--yes"]) == 2
    assert "No .czi/.tif/.tiff files" in capsys.readouterr().err
    images = _images(tmp_path)
    assert cli.main(["init", str(tmp_path / "c"), "--images", str(images), "--yes"]) == 2
    assert "--panel is required" in capsys.readouterr().err


def test_prepare_stops_until_mouse_ids_are_filled(tmp_path, capsys, monkeypatch):
    images = _images(tmp_path)
    root = tmp_path / "study"
    cli.main(["init", str(root), "--images", str(images), "--panel",
              "four_marker_i_iia_laminin_iib", "--yes"])  # fmt: skip
    (root / "batch").mkdir()
    (root / "batch/batch_summary.csv").write_text("image_name,status\nm1_s1,success\n")
    monkeypatch.setattr(cli, "_run", lambda module, arguments: 0)
    capsys.readouterr()

    assert cli.main(["prepare", str(root)]) == 2
    assert "Fill in mouse_id" in capsys.readouterr().err
    assert cli.main(["status", str(root)]) == 0
    assert "Fill in mouse_id" in capsys.readouterr().out


def test_config_may_name_a_panel_preset(tmp_path):
    root = _project(tmp_path, panel="three_marker_iib_iia_laminin")

    assert cli.load_project_folder(root).panel.name == "three_marker_iib_iia_laminin.yaml"


def _nested_study(tmp_path: Path) -> Path:
    """genotype/treatment/mouse folders with originals, processed copies, and clutter."""
    import numpy as np
    import tifffile

    study = tmp_path / "study"
    files = [
        "geno_a/treated/m01/sections.tif",
        "geno_a/treated/m01/Processed/001/sections.tif",
        "geno_a/treated/m01/Split Scenes/section_001.tif",
        "geno_a/untreated/m02/sections.TIF",
        "geno_b/treated/m03/other.tif",
    ]
    for name in files:
        path = study / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(
            path, np.zeros((3, 8, 8), dtype=np.uint16), imagej=True, metadata={"axes": "CYX"}
        )
    (study / "geno_a/treated/m01/._sections.tif").write_bytes(b"macOS metadata")
    (study / ".hidden").mkdir()
    (study / ".hidden/x.tif").write_bytes(b"")
    (study / "old_project").mkdir()
    (study / "old_project" / cli.CONFIG_NAME).write_text("schema_version: x\n")
    (study / "old_project/batch.tif").write_bytes(b"")
    return study


def test_include_patterns_find_images_in_subfolders(tmp_path):
    study = _nested_study(tmp_path)
    relative = lambda paths: [p.relative_to(study).as_posix() for p in paths]  # noqa: E731

    with pytest.raises(cli.ProjectError, match=r"5 found in subfolders.*--include"):
        cli.image_files(study)
    assert relative(
        cli.image_files(study, ("**/*.tif",), ("**/Processed/**", "**/Split Scenes/**"))
    ) == [
        "geno_a/treated/m01/sections.tif",
        "geno_a/untreated/m02/sections.TIF",
        "geno_b/treated/m03/other.tif",
    ]
    assert relative(cli.image_files(study, ("**/Processed/**/*.tif",))) == [
        "geno_a/treated/m01/Processed/001/sections.tif"
    ]
    with pytest.raises(cli.ProjectError, match="match include"):
        cli.image_files(study, ("*.tif",))  # one star stays in the top folder


def test_init_with_patterns_writes_them_and_a_sample_sheet_with_unique_ids(tmp_path, capsys):
    study = _nested_study(tmp_path)
    root = tmp_path / "proj"
    arguments = [
        "init",
        str(root),
        "--images",
        str(study),
        "--panel",
        "three_marker_iib_iia_laminin",
    ]
    arguments += ["--model", "none", "--include", "**/*.tif", "--exclude", "**/Processed/**"]
    assert cli.main([*arguments, "--exclude", "**/Split Scenes/**", "--yes"]) == 0

    config = yaml.safe_load((root / cli.CONFIG_NAME).read_text())
    assert config["include"] == ["**/*.tif"]
    assert config["exclude"] == ["**/Processed/**", "**/Split Scenes/**"]
    sheet = pd.read_csv(root / "samples.csv", dtype=str, keep_default_na=False)
    assert sheet["image_id"].tolist() == ["m01_sections", "m02_sections", "other"]
    assert sheet["use"].tolist() == ["yes", "yes", "yes"]
    assert "use to 'no'" in capsys.readouterr().out
    project = cli.load_project_folder(root)
    assert project.include == ("**/*.tif",) and cli.runs_from_sample_sheet(project)


def test_unique_ids_go_up_as_many_folders_as_needed(tmp_path):
    images = [tmp_path / "a/x/m1/s.tif", tmp_path / "b/x/m1/s.tif", tmp_path / "c/t.tif"]
    assert cli._unique_ids(images, tmp_path) == ["a_x_m1_s", "b_x_m1_s", "t"]
    with pytest.raises(cli.ProjectError, match="same image ID"):
        cli._unique_ids([tmp_path / "s.tif", tmp_path / "s.tiff"], tmp_path)


def test_run_uses_only_sample_sheet_rows_marked_use_yes(tmp_path, monkeypatch):
    study = _nested_study(tmp_path)
    root = tmp_path / "proj"
    cli.main(
        ["init", str(root), "--images", str(study), "--panel", "three_marker_iib_iia_laminin",
         "--model", "none", "--include", "**/*.tif", "--exclude", "**/Processed/**",
         "--exclude", "**/Split Scenes/**", "--yes"]
    )  # fmt: skip
    sheet = pd.read_csv(root / "samples.csv", dtype=str, keep_default_na=False)
    sheet.loc[sheet["image_id"].eq("other"), "use"] = "no"
    sheet.to_csv(root / "samples.csv", index=False)
    calls: list[list[str]] = []
    monkeypatch.setattr(cli, "_run", lambda module, arguments: calls.append(arguments) or 0)

    assert cli.main(["run", str(root)]) == 0
    arguments = calls[-1]
    assert "--input-dir" not in arguments
    manifest = Path(arguments[arguments.index("--input-manifest") + 1])
    table = pd.read_csv(manifest, dtype=str)
    assert table["image_id"].tolist() == ["m01_sections", "m02_sections"]
    assert table["input_path"].tolist() == [
        str(study / "geno_a/treated/m01/sections.tif"),
        str(study / "geno_a/untreated/m02/sections.TIF"),
    ]

    sheet.loc[0, "use"] = "maybe"
    sheet.to_csv(root / "samples.csv", index=False)
    assert cli.main(["run", str(root)]) == 2


def test_flat_projects_still_run_from_the_images_folder(tmp_path, monkeypatch):
    images = _images(tmp_path)
    root = tmp_path / "flat"
    cli.main(
        ["init", str(root), "--images", str(images), "--panel", "three_marker_iib_iia_laminin",
         "--model", "none", "--yes"]
    )  # fmt: skip
    assert "include" not in yaml.safe_load((root / cli.CONFIG_NAME).read_text())
    assert "use" not in pd.read_csv(root / "samples.csv").columns
    calls: list[list[str]] = []
    monkeypatch.setattr(cli, "_run", lambda module, arguments: calls.append(arguments) or 0)
    assert cli.main(["run", str(root)]) == 0
    assert calls[-1][:2] == ["--input-dir", str(images)]


def test_prepare_ignores_blank_mouse_ids_on_rows_not_used(tmp_path):
    root = _project(tmp_path)
    pd.DataFrame(
        {"image_id": ["a", "b"], "mouse_id": ["m1", ""], "raw_image_path": ["a.tif", "b.tif"],
         "use": ["yes", "no"]}
    ).to_csv(root / "samples.csv", index=False)  # fmt: skip
    cli.check_sample_sheet(cli.load_project_folder(root))
    pd.DataFrame(
        {"image_id": ["a"], "mouse_id": ["m1"], "raw_image_path": ["a.tif"], "use": ["no"]}
    ).to_csv(root / "samples.csv", index=False)
    with pytest.raises(cli.ProjectError, match="marked use = yes"):
        cli.check_sample_sheet(cli.load_project_folder(root))


def test_init_rejects_channel_zero_and_channels_beyond_the_images(tmp_path, monkeypatch):
    images = _images(tmp_path, names=("slide",))  # four channels
    for laminin in ("0", "5"):
        answers = iter([str(images), "3", laminin])
        monkeypatch.setattr("builtins.input", lambda prompt="", answers=answers: next(answers))
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        assert cli.main(["init", str(tmp_path / f"study{laminin}")]) == 2


def test_channel_names_come_from_ome_metadata(tmp_path):
    import numpy as np
    import tifffile

    from fibertypeqc.panel_presets import channel_label

    named = tmp_path / "named.ome.tif"
    tifffile.imwrite(
        named,
        np.zeros((2, 8, 8), dtype=np.uint16),
        ome=True,
        metadata={"axes": "CYX", "Channel": {"Name": ["DAPI", "AF488"]}},
    )
    assert cli.image_channel_names(named) == ["DAPI", "AF488"]
    assert cli.channel_summary(named, 2, ["DAPI", "AF488"]) == (
        "named.ome.tif has 2 channel(s): Channel 1 · DAPI, Channel 2 · AF488"
    )
    plain = _images(tmp_path, names=("plain",)) / "plain.tif"
    assert cli.image_channel_names(plain) == []
    assert channel_label(0) == "Channel 1"
    assert channel_label(1, ["DAPI", ""]) == "Channel 2"
