from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest

from src.run_fixed_mask_transfer_task import (
    REQUIRED_SEMANTIC_FEATURES,
    _canonicalize_diagnostics,
    build_diagnostics_command,
    build_matching_command,
    load_manifest,
)


def _manifest(tmp_path: Path) -> Path:
    image = tmp_path / "image.tif"
    labels = tmp_path / "labels.tif"
    results = tmp_path / "Results"
    image.write_bytes(b"image")
    labels.write_bytes(b"labels")
    results.mkdir()
    (results / "Results.txt").write_text("Label\nIIb\n")
    (results / "ROISet.zip").write_bytes(b"roi")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "image_id": ["image_a"],
            "input_path": [image],
            "labels_path": [labels],
            "myosight_results_dir": [results],
        }
    ).to_csv(manifest, index=False)
    return manifest


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        typing_preprocess="global_subtract",
        typing_bg_quantile=0.02,
        typing_tile_size=512,
        typing_bg_sigma=24.0,
        typing_smooth_sigma=0.8,
        typing_erode_px=2,
        coverage_quantile=0.85,
        sensitivity=0.5,
        mixed_strictness=0.7,
    )


def test_fixed_mask_command_explicitly_reuses_labels_and_disables_auto_crop(tmp_path):
    row = load_manifest(_manifest(tmp_path))[0]
    command = build_diagnostics_command(
        row,
        output_dir=tmp_path / "output",
        channel_config=tmp_path / "channels.yaml",
        args=_args(),
    )

    assert "--labels-path" in command
    assert command[command.index("--labels-path") + 1] == str(row.labels_path)
    assert "--no-crop-auto" in command
    assert "--export-diagnostics" in command
    assert "--retain-mode" in command
    assert "src.run_pipeline" in command


def test_matching_command_uses_same_cached_labels_and_exhaustive_candidate_mode(tmp_path):
    row = load_manifest(_manifest(tmp_path))[0]
    command = build_matching_command(
        row,
        output_dir=tmp_path / "matches",
        max_centroid_distance=25.0,
    )

    assert command[command.index("--pipeline-labels") + 1] == str(row.labels_path)
    assert command[command.index("--myosight-results-dir") + 1] == str(row.myosight_results_dir)
    assert "--no-panel" in command


def test_canonicalize_diagnostics_requires_complete_current_schema(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    diagnostics = pd.DataFrame({column: [1.0] for column in REQUIRED_SEMANTIC_FEATURES})
    diagnostics["label"] = [1]
    diagnostics.to_csv(output / "source_stem_feature_diagnostics.csv", index=False)
    (output / "source_stem_summary.csv").write_text("metric,value\ncount,1\n")

    path = _canonicalize_diagnostics(output, "canonical_image")

    assert path.name == "canonical_image_feature_diagnostics.csv"
    assert (output / "canonical_image_summary.csv").is_file()


def test_canonicalize_diagnostics_handles_case_only_rename(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    diagnostics = pd.DataFrame({column: [1.0] for column in REQUIRED_SEMANTIC_FEATURES})
    diagnostics["label"] = [1]
    diagnostics.to_csv(output / "section001_feature_diagnostics.csv", index=False)

    path = _canonicalize_diagnostics(output, "Section001")

    assert path.is_file()
    assert path.name == "Section001_feature_diagnostics.csv"


def test_canonicalize_diagnostics_rejects_missing_f1_features(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    pd.DataFrame({"label": [1], "type_iia.mean": [1.0]}).to_csv(
        output / "source_feature_diagnostics.csv", index=False
    )

    with pytest.raises(ValueError, match="missing required transfer features"):
        _canonicalize_diagnostics(output, "canonical_image")
