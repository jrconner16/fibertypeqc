import hashlib
from pathlib import Path

from fibertypeqc.artifacts import (
    build_run_manifest,
    can_reuse_fiber_labels,
    decide_artifact_reuse,
    file_sha256,
    portable_path,
)


def test_classifier_change_reuses_labels_but_recomputes_classification():
    assert decide_artifact_reuse(classifier_changed=True) == {
        "reuse_fiber_labels": True,
        "reuse_nuclei_labels": True,
        "recompute_features_or_links": True,
    }


def test_fiber_segmentation_change_invalidates_only_fiber_labels():
    assert decide_artifact_reuse(fiber_segmentation_changed=True) == {
        "reuse_fiber_labels": False,
        "reuse_nuclei_labels": True,
        "recompute_features_or_links": True,
    }


def test_fiber_label_reuse_requires_matching_stage_fingerprint():
    previous = {"schema_version": 2, "stage_fingerprints": {"fiber_segmentation": "same"}}
    assert can_reuse_fiber_labels(previous, previous)
    assert not can_reuse_fiber_labels(
        previous, {"schema_version": 2, "stage_fingerprints": {"fiber_segmentation": "new"}}
    )
    # Records from before the schema bump never satisfy reuse.
    old = {"schema_version": 1, "stage_fingerprints": {"fiber_segmentation": "same"}}
    assert not can_reuse_fiber_labels(old, previous)


def _manifest(**overrides):
    kwargs = {
        "input_path": Path("images/sample.tif"),
        "input_sha256": "a" * 64,
        "image_shape": (3, 10, 10),
        "pixel_size_um": (0.5, 0.5),
        "panel_fingerprint": "panel",
        "panel_channels": {"laminin": 2},
        "segmentation": {"model": "cpsam", "device": "cpu", "labels_source": "cellpose"},
        "preprocessing": {"downsample_factor": 2},
        "classifier_path": "data/models/model.joblib",
        "model_manifest_path": None,
        "classifier_sha256": "b" * 64,
    }
    kwargs.update(overrides)
    return build_run_manifest(**kwargs)


def test_segmentation_fingerprint_depends_on_image_content_and_labels_source():
    base = _manifest()
    fingerprints = base["stage_fingerprints"]

    assert _manifest()["stage_fingerprints"] == fingerprints
    assert (
        _manifest(input_sha256="c" * 64)["stage_fingerprints"]["fiber_segmentation"]
        != fingerprints["fiber_segmentation"]
    )
    provided = _manifest(
        segmentation={"model": "cpsam", "device": "not_used", "labels_source": "provided:x"}
    )
    provided_fingerprint = provided["stage_fingerprints"]["fiber_segmentation"]
    assert provided_fingerprint != fingerprints["fiber_segmentation"]


def test_classification_fingerprint_uses_model_digest_not_path():
    moved = _manifest(classifier_path="/elsewhere/model.joblib")
    retrained = _manifest(classifier_sha256="d" * 64)

    assert (
        moved["stage_fingerprints"]["classification"]
        == (_manifest()["stage_fingerprints"]["classification"])
    )
    assert (
        retrained["stage_fingerprints"]["classification"]
        != (_manifest()["stage_fingerprints"]["classification"])
    )


def test_run_manifest_records_portable_paths_and_library_versions(tmp_path):
    manifest = _manifest(input_path=tmp_path / "private" / "sample.czi")

    assert manifest["source_image"] == "sample.czi"
    assert manifest["classifier_path"] == "data/models/model.joblib"
    assert {"numpy", "scikit-learn", "cellpose"} <= set(manifest["dependency_versions"])


def test_portable_path_keeps_relative_and_strips_outside_directories(tmp_path):
    inside = tmp_path / "project" / "data" / "x.tif"

    assert portable_path("images/a.tif") == "images/a.tif"
    assert portable_path(inside, base=tmp_path / "project") == "data/x.tif"
    assert portable_path(tmp_path / "elsewhere" / "b.czi", base=tmp_path / "project") == "b.czi"
    assert portable_path(None) == ""


def test_file_sha256_streams_file_content(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"fibertypeqc")

    assert file_sha256(path, chunk_size=3) == hashlib.sha256(b"fibertypeqc").hexdigest()
