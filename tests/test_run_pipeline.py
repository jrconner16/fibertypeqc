from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.run_pipeline as run_pipeline


def _mock_pipeline_primitives(monkeypatch):
    image = np.zeros((3, 2, 2), dtype=np.float32)
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
