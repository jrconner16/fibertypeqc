from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.run_pipeline as run_pipeline


def _mock_pipeline_primitives(monkeypatch):
    image = np.zeros((4, 2, 2), dtype=np.float32)
    image[0, 0, 0] = 10.0
    image[1, 1, 1] = 12.0
    image[2, :, :] = 1.0
    labels = np.array([[1, 0], [0, 2]], dtype=np.int32)

    monkeypatch.setattr(run_pipeline, "load_multichannel_image", lambda _: image)
    monkeypatch.setattr(run_pipeline, "extract_pixel_size_um", lambda _: (None, None))
    monkeypatch.setattr(
        run_pipeline,
        "preprocess_membrane_channel",
        lambda membrane, cfg: SimpleNamespace(
            membrane_model_input=membrane,
            membrane_crop=membrane,
            membrane_full=membrane,
            crop_slices=(slice(0, 2), slice(0, 2)),
        ),
    )
    monkeypatch.setattr(run_pipeline, "run_cellpose", lambda arr, cfg: (labels, 0.01))
    monkeypatch.setattr(run_pipeline, "upsample_labels_nearest", lambda arr, **kwargs: arr)
    monkeypatch.setattr(run_pipeline, "paste_crop_labels", lambda arr, shape, crop: arr)


@pytest.mark.integration
def test_run_pipeline_export_diagnostics_flag_controls_output(tmp_path, monkeypatch):
    _mock_pipeline_primitives(monkeypatch)
    input_path = tmp_path / "image.tif"
    input_path.write_bytes(b"fake")

    base_args = [
        "run_pipeline",
        "--input",
        str(input_path),
        "--output-dir",
        str(tmp_path / "out"),
        "--iib-channel",
        "0",
        "--iia-channel",
        "1",
        "--membrane-channel",
        "2",
        "--typing-preprocess",
        "raw",
        "--typing-smooth-sigma",
        "0",
        "--typing-erode-px",
        "0",
        "--classifier-path",
        "",
    ]

    monkeypatch.setattr(run_pipeline.sys, "argv", base_args)
    run_pipeline.main()

    stem = input_path.stem
    run_manifest = json.loads((tmp_path / "out" / f"{stem}_run.json").read_text())
    assert run_manifest["panel"]["channels"]["laminin"] == 2
    assert run_manifest["output_schema_version"] == "legacy_fibers.v1"
    diagnostics_path = tmp_path / "out" / f"{stem}_feature_diagnostics.csv"
    assert not diagnostics_path.exists()

    monkeypatch.setattr(
        run_pipeline.sys,
        "argv",
        [*base_args, "--output-dir", str(tmp_path / "out_diag"), "--export-diagnostics"],
    )
    run_pipeline.main()

    diagnostics_path = tmp_path / "out_diag" / f"{stem}_feature_diagnostics.csv"
    assert diagnostics_path.exists()
    diagnostics = pd.read_csv(diagnostics_path)
    assert {"label", "fiber_type", "type1_mean", "type1_snr_mean", "type_cov_sum"}.issubset(
        diagnostics.columns
    )


@pytest.mark.integration
def test_run_pipeline_writes_summary_for_diagnostics_only_panel(tmp_path, monkeypatch):
    _mock_pipeline_primitives(monkeypatch)
    input_path = tmp_path / "image.tif"
    input_path.write_bytes(b"fake")
    panel_path = tmp_path / "panel.yaml"
    panel_path.write_text(
        "channels:\n"
        "  laminin: 2\n"
        "  dapi: 3\n"
        "  type_i: 0\n"
        "  type_iia: 1\n"
        "  type_iib: null\n"
        "  type_iix: null\n"
        "  emhc: null\n"
        "classification:\n"
        "  residual_inference:\n"
        "    enabled: false\n"
        "    target_class: iix\n"
        "    requires_negative_markers: []\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        run_pipeline.sys,
        "argv",
        [
            "run_pipeline",
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--panel-config",
            str(panel_path),
            "--export-diagnostics",
            "--typing-preprocess",
            "raw",
            "--typing-smooth-sigma",
            "0",
            "--typing-erode-px",
            "0",
        ],
    )
    run_pipeline.main()

    summary = pd.read_csv(tmp_path / "out" / f"{input_path.stem}_summary.csv")
    assert pd.isna(summary.loc[0, "type1_channel"])
    nuclear_dir = tmp_path / "out" / "nuclear"
    assert nuclear_dir.is_dir()
    assert (nuclear_dir / f"{input_path.stem}_nuclear_run.json").exists()
    assert summary.loc[0, "nuclear_manifest_path"] == str(
        nuclear_dir / f"{input_path.stem}_nuclear_run.json"
    )


def test_cleanup_outputs_for_retain_mode_tables_removes_labels_only(tmp_path):
    labels_path = tmp_path / "image_cellpose_labels.tif"
    fibers_path = tmp_path / "image_fibers.csv"
    diagnostics_path = tmp_path / "image_feature_diagnostics.csv"
    summary_path = tmp_path / "image_summary.csv"
    for path in (labels_path, fibers_path, diagnostics_path, summary_path):
        path.write_text("x")

    removed = run_pipeline._cleanup_outputs_for_retain_mode(
        retain_mode="tables",
        labels_path=labels_path,
        fibers_path=fibers_path,
        diagnostics_path=diagnostics_path,
        summary_path=summary_path,
    )

    assert removed == [labels_path]
    assert not labels_path.exists()
    assert fibers_path.exists()
    assert diagnostics_path.exists()
    assert summary_path.exists()


def test_cleanup_outputs_for_retain_mode_summary_keeps_only_summary(tmp_path):
    labels_path = tmp_path / "image_cellpose_labels.tif"
    fibers_path = tmp_path / "image_fibers.csv"
    diagnostics_path = tmp_path / "image_feature_diagnostics.csv"
    summary_path = tmp_path / "image_summary.csv"
    for path in (labels_path, fibers_path, diagnostics_path, summary_path):
        path.write_text("x")

    removed = run_pipeline._cleanup_outputs_for_retain_mode(
        retain_mode="summary",
        labels_path=labels_path,
        fibers_path=fibers_path,
        diagnostics_path=diagnostics_path,
        summary_path=summary_path,
    )

    assert removed == [labels_path, fibers_path, diagnostics_path]
    assert not labels_path.exists()
    assert not fibers_path.exists()
    assert not diagnostics_path.exists()
    assert summary_path.exists()
