"""Model builder on panels other than Type I + IIa: classes, features, and channel mapping follow
the panel's marker identities, never fixed channel positions."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml

from fibertypeqc.config import load_channel_config
from fibertypeqc.model_manifest import load_model_manifest
from fibertypeqc.semantic_model import predict_semantic_candidate
from src import cli
from src.fiber_type_labels import default_classes, negative_display_label, validate_classes
from src.make_review_project import build_project
from src.model_builder import blind_label_sessions, build, training_features
from src.model_evaluation import evaluate, find_build_candidate
from src.promote_model import promote, rollback
from src.review.project import load_project
from tests.test_model_builder import _label, _tiled
from tests.test_no_model_project import FOUR_CLASS

# --- three-marker panel (unit) --------------------------------------------------------------

# Channels deliberately out of marker order: IIa first, Type I last.
THREE_MARKER_PANEL = (
    "channels:\n  type_iia: 0\n  laminin: 1\n  type_iib: 2\n  type_i: 3\n  dapi: null\n"
    "  type_iix: null\n  emhc: null\nclassification:\n  residual_inference:\n"
    "    enabled: false\n    target_class: null\n    requires_negative_markers: []\n"
)


def test_three_marker_panel_classes_labels_and_features_follow_marker_identity(tmp_path):
    panel = tmp_path / "panel.yaml"
    panel.write_text(THREE_MARKER_PANEL)
    config = load_channel_config(panel)
    # Channel positions are read from the panel; markers come back in their fixed biological order.
    assert (config.iia_channel, config.membrane_channel, config.iib_channel, config.i_channel) == (
        0, 1, 2, 3,
    )  # fmt: skip
    markers = config.available_markers
    assert markers == ("i", "iia", "iib")

    assert default_classes(markers) == ["i", "iia", "iib", "negative"]
    assert negative_display_label(markers) == "I⁻/IIa⁻/IIb⁻"
    assert validate_classes(["iib", "I", "hybrid", "negative"], markers) == [
        "iib", "i", "hybrid", "negative",
    ]  # fmt: skip
    with pytest.raises(ValueError, match="iix"):
        validate_classes(["i", "iix"], markers)  # IIx is not measured by this panel

    features = training_features(markers)
    per_marker = len(features) // 3
    assert len(features) == 27
    assert [name.split(".")[0] for name in features[::per_marker]] == [
        "type_i", "type_iia", "type_iib",
    ]  # fmt: skip
    # The same markers in another channel order give the same features in the same order.
    assert training_features(("iib", "i", "iia")) == features


def test_three_marker_project_defaults(tmp_path):
    root = tmp_path / "proj"
    (root / "images").mkdir(parents=True)
    (root / "panel.yaml").write_text(THREE_MARKER_PANEL)
    (root / "samples.csv").write_text("image_id,mouse_id,raw_image_path\n")
    (root / cli.CONFIG_NAME).write_text(
        yaml.safe_dump(
            {
                "schema_version": "fibertypeqc_project.v1",
                "images": "images",
                "panel": "panel.yaml",
                "sample_sheet": "samples.csv",
                "model": "none",
            }
        )
    )
    project = cli.load_project_folder(root)
    assert project.classes == ("i", "iia", "iib", "negative")
    assert project.negative_label == "I⁻/IIa⁻/IIb⁻"


# --- IIa + IIb panel, end to end ------------------------------------------------------------

# Laminin in the middle and IIb before IIa, unlike the fixture's own channel order.
IIA_IIB_PANEL = (
    "channels:\n  type_iib: 0\n  laminin: 1\n  type_iia: 2\n  type_i: null\n  dapi: null\n"
    "  type_iix: null\n  emhc: null\nclassification:\n  residual_inference:\n"
    "    enabled: false\n    target_class: null\n    requires_negative_markers: []\n"
)
CLASSES = ["iia", "iib", "hybrid", "negative"]
NEGATIVE = "IIa⁻/IIb⁻"
# On this panel the fixture's Type I fibers are made IIa+IIb co-expressing (hybrid), and its
# IIx fibers (no I, IIa, or IIb signal) are negative.
PANEL_TRUTH = {"i": "hybrid", "iia": "iia", "iib": "iib", "iix": "negative"}
TRAINING_MICE = ("m1", "m2", "m3")
TEST_MOUSE = "m4"


FOUR_CLASS_TRUTH = {
    int(label): kind
    for label, kind in json.loads((FOUR_CLASS / "expected_fiber_types.json").read_text())[
        "fiber_type_by_label"
    ].items()
}


def _truth(fiber_id: int) -> str:
    return PANEL_TRUTH[FOUR_CLASS_TRUTH[(fiber_id - 1) % 16 + 1]]


def _iia_iib_image(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The tiled fixture as a IIa/IIb section in panel order (IIb, laminin, IIa)."""
    image, labels = _tiled(seed)
    type_i, iia, laminin, iib = image
    ids = [label for label in np.unique(labels) if label and _truth(label) == "hybrid"]
    hybrid = np.isin(labels, ids)
    iia = np.where(hybrid, type_i, iia)
    iib = np.where(hybrid, type_i, iib)
    return np.stack([iib, laminin, iia]), labels


def _measure(root, image_id, image, labels, panel, batch):
    from src import run_pipeline

    tifffile.imwrite(root / "images" / f"{image_id}.tif", image, photometric="minisblack")
    tifffile.imwrite(root / "images" / f"{image_id}_labels.tif", labels)
    argv = sys.argv
    sys.argv = [
        "run_pipeline",
        "--input", str(root / "images" / f"{image_id}.tif"),
        "--labels-path", str(root / "images" / f"{image_id}_labels.tif"),
        "--output-dir", str(batch / image_id),
        "--image-id", image_id,
        "--panel-config", str(panel),
        "--no-model", "--cpu", "--bootstrap-reps", "10",
    ]  # fmt: skip
    try:
        run_pipeline.main()
    finally:
        sys.argv = argv
    return pd.read_csv(batch / image_id / f"{image_id}_feature_diagnostics.csv")


@pytest.fixture(scope="module")
def iia_iib_project(tmp_path_factory):
    root = tmp_path_factory.mktemp("iia_iib")
    (root / "images").mkdir()
    panel = root / "panel.yaml"
    panel.write_text(IIA_IIB_PANEL)
    batch = root / "batch"
    sheet_rows, summary_rows = [], []
    for seed, mouse in enumerate((*TRAINING_MICE, TEST_MOUSE)):
        image_id = f"syn_{mouse}"
        image, labels = _iia_iib_image(seed)
        _measure(root, image_id, image, labels, panel, batch)
        sheet_rows.append(f"{image_id},{mouse},images/{image_id}.tif")
        summary_rows.append(f"{image_id},success")
    (batch / "batch_summary.csv").write_text("image_name,status\n" + "\n".join(summary_rows))
    sheet = root / "samples.csv"
    sheet.write_text("image_id,mouse_id,raw_image_path\n" + "\n".join(sheet_rows))
    review = root / "review"
    review.mkdir()
    manifest, _ = build_project(
        batch, sheet, panel, review, project_id="iia_iib", project_name="iia_iib",
        classes=CLASSES, negative_label=NEGATIVE,
    )  # fmt: skip
    (review / "project.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    roles = [f"{mouse},pool" for mouse in TRAINING_MICE] + [f"{TEST_MOUSE},test"]
    (root / "evaluation_roles.csv").write_text("mouse_id,role\n" + "\n".join(roles))
    (root / cli.CONFIG_NAME).write_text(
        yaml.safe_dump(
            {
                "schema_version": "fibertypeqc_project.v1",
                "images": "images",
                "panel": "panel.yaml",
                "sample_sheet": "samples.csv",
                "model": "none",
                "classes": CLASSES,
            }
        )
    )
    project = load_project(review / "project.yaml")
    _label(
        project,
        "a",
        {
            (f"syn_{mouse}", fiber): _truth(fiber)
            for mouse in (*TRAINING_MICE, TEST_MOUSE)
            for fiber in range(1, 145)
        },
    )
    return root


def _tree(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).parts[0] != "models"
    }


def test_iia_iib_panel_with_hybrid_builds_evaluates_promotes_and_rolls_back(
    iia_iib_project, tmp_path
):
    root = tmp_path / "project"
    shutil.copytree(iia_iib_project, root)
    project = load_project(root / "review" / "project.yaml")
    assert project.review_classes == tuple(CLASSES)
    assert project.class_labels == {"negative": NEGATIVE}

    markers = load_channel_config(root / "panel.yaml").available_markers
    assert markers == ("iia", "iib")
    result = build(
        project, blind_label_sessions(project), classes=CLASSES, markers=markers,
        output_dir=root / "models" / "build_20260101T000000Z", class_labels=project.class_labels,
    )  # fmt: skip
    assert result["labels"]["by_class"] == {name: 108 for name in CLASSES}
    chosen = result["selection"]["decision"]["recommended"]
    assert chosen, result["selection"]["decision"]["reason"]

    # The manifest names markers and features by identity; nothing refers to channel positions.
    artifact = find_build_candidate(root, chosen)
    raw = yaml.safe_load(artifact.with_suffix(".yaml").read_text())
    assert raw["features"] == training_features(markers)
    assert {name.split(".")[0] for name in raw["features"]} == {"type_iia", "type_iib"}
    assert set(raw["required_markers"]) == {"laminin", "type_iia", "type_iib"}
    assert raw["outputs"] == CLASSES
    assert raw["class_labels"] == {"negative": NEGATIVE}
    assert "channel" not in yaml.safe_dump(raw["features"])

    evaluation = evaluate(root, project, chosen)
    assert set(evaluation["pooled_recall_by_class"]) == set(CLASSES)
    assert evaluation["pooled_recall_by_class"]["hybrid"] >= 0.7

    before = _tree(root)
    row = promote(root, chosen, root / "panel.yaml")
    assert row["independent_test"]["status"] == "current"
    promoted = load_project(root / "review" / "project.yaml")
    assert promoted.review_classes == tuple(CLASSES)
    assert promoted.class_labels == {"negative": NEGATIVE}
    calls = pd.concat(
        pd.read_csv(root / f"batch/syn_{mouse}/syn_{mouse}_fibers.csv")
        for mouse in (*TRAINING_MICE, TEST_MOUSE)
    )
    assert set(calls["fiber_type"]) == set(CLASSES)  # hybrid and negative are called

    rollback(root)
    assert _tree(root) == before


def test_features_and_predictions_do_not_depend_on_channel_order(iia_iib_project, tmp_path):
    """The same section stored in another channel order, with a panel that maps it, measures
    the same features and gets the same calls from a built model."""
    root = tmp_path / "project"
    shutil.copytree(iia_iib_project, root)
    project = load_project(root / "review" / "project.yaml")
    result = build(
        project, blind_label_sessions(project), classes=CLASSES, markers=("iia", "iib"),
        output_dir=root / "models" / "build_20260101T000000Z", class_labels=project.class_labels,
    )  # fmt: skip
    artifact = find_build_candidate(root, "logistic")
    assert result["labels"]["n_fibers"] > 0
    manifest = load_model_manifest(artifact.with_suffix(".yaml"))

    image_id = f"syn_{TEST_MOUSE}"
    image, labels = _iia_iib_image(3)
    original = pd.read_csv(root / "batch" / image_id / f"{image_id}_feature_diagnostics.csv")
    # Stored as (IIa, IIb, laminin) instead of (IIb, laminin, IIa), with a panel that says so.
    reordered_panel = tmp_path / "reordered.yaml"
    reordered_panel.write_text(
        IIA_IIB_PANEL.replace("type_iib: 0", "type_iib: 1")
        .replace("laminin: 1", "laminin: 2")
        .replace("type_iia: 2", "type_iia: 0")
    )
    (tmp_path / "images").mkdir()
    reordered = _measure(
        tmp_path, image_id, image[[2, 0, 1]], labels, reordered_panel, tmp_path / "batch"
    )

    features = training_features(("iia", "iib"))
    pd.testing.assert_frame_equal(original[["label", *features]], reordered[["label", *features]])
    calls = predict_semantic_candidate(original, artifact, manifest)["model_prediction"]
    again = predict_semantic_candidate(reordered, artifact, manifest)["model_prediction"]
    assert calls.tolist() == again.tolist()
