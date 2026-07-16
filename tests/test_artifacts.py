from fibertypeqc.artifacts import can_reuse_fiber_labels, decide_artifact_reuse


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
    previous = {"schema_version": 1, "stage_fingerprints": {"fiber_segmentation": "same"}}
    assert can_reuse_fiber_labels(previous, previous)
    assert not can_reuse_fiber_labels(
        previous, {"schema_version": 1, "stage_fingerprints": {"fiber_segmentation": "new"}}
    )
