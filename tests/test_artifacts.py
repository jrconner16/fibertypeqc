from fibertypeqc.artifacts import decide_artifact_reuse


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
