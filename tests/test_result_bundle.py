import json

import pytest

from fibertypeqc.result_bundle import (
    RESULT_BUNDLE_SCHEMA_VERSION,
    build_result_bundle,
    write_result_bundle,
)


def test_result_bundle_indexes_only_retained_artifacts_with_relative_paths(tmp_path):
    output_dir = tmp_path / "image"
    output_dir.mkdir()
    fibers = output_dir / "image_fibers.csv"
    summary = output_dir / "image_summary.csv"
    fibers.write_text("label\n1\n")
    summary.write_text("input\nimage.tif\n")

    bundle = build_result_bundle(
        output_dir=output_dir,
        image_id="image",
        retain_mode="tables",
        artifact_paths={
            "fiber_labels": output_dir / "image_cellpose_labels.tif",
            "fiber_table": fibers,
            "image_summary": summary,
        },
    )

    assert bundle["schema_version"] == RESULT_BUNDLE_SCHEMA_VERSION
    assert bundle["image_id"] == "image"
    assert bundle["retain_mode"] == "tables"
    assert set(bundle["artifacts"]) == {"fiber_table", "image_summary"}
    assert bundle["artifacts"]["fiber_table"]["path"] == "image_fibers.csv"
    assert bundle["artifacts"]["fiber_table"]["join_keys"] == ["label"]
    assert bundle["artifacts"]["fiber_table"]["cardinality"] == "one_row_per_fiber"


def test_result_bundle_can_mark_emhc_diagnostics_as_regeneration_domain(tmp_path):
    diagnostics = tmp_path / "image_feature_diagnostics.csv"
    diagnostics.write_text("label,emhc.mean\n1,2.0\n")

    bundle = build_result_bundle(
        output_dir=tmp_path,
        image_id="image",
        retain_mode="full",
        artifact_paths={"feature_diagnostics": diagnostics},
        additional_domains={"feature_diagnostics": ["regeneration"]},
    )

    assert bundle["artifacts"]["feature_diagnostics"]["domains"] == [
        "fiber_identity",
        "regeneration",
    ]


def test_result_bundle_rejects_artifacts_outside_output_directory(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    outside = tmp_path / "fibers.csv"
    outside.write_text("label\n1\n")

    with pytest.raises(ValueError, match="inside output_dir"):
        build_result_bundle(
            output_dir=output_dir,
            image_id="image",
            retain_mode="full",
            artifact_paths={"fiber_table": outside},
        )


def test_result_bundle_rejects_unknown_retain_mode(tmp_path):
    with pytest.raises(ValueError, match="Unsupported retain mode"):
        build_result_bundle(
            output_dir=tmp_path,
            image_id="image",
            retain_mode="everything",
            artifact_paths={},
        )


def test_write_result_bundle_round_trips_json(tmp_path):
    path = tmp_path / "image_result_bundle.json"
    bundle = {
        "schema_version": RESULT_BUNDLE_SCHEMA_VERSION,
        "image_id": "image",
        "retain_mode": "summary",
        "artifacts": {},
    }

    write_result_bundle(path, bundle)

    assert json.loads(path.read_text()) == bundle
