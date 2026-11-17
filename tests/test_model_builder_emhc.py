"""Model builder on a regeneration panel (eMHC + laminin, no myosin marker): the classes are eMHC⁺
and eMHC⁻, the model reads the eMHC stain only, and fiber-type models never see eMHC."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from fibertypeqc.config import load_channel_config
from src import cli
from src.fiber_type_labels import (
    class_labels_for,
    default_classes,
    negative_display_label,
    validate_classes,
)
from src.make_review_project import build_project
from src.model_builder import blind_label_sessions, build, training_features
from src.model_evaluation import evaluate, find_build_candidate
from src.promote_model import promote, rollback
from src.review.project import load_project
from src.summarize_results import ordered_classes
from tests.test_model_builder import _label, _tiled
from tests.test_model_builder_other_panels import FOUR_CLASS_TRUTH, _measure, _tree

EMHC_PANEL = (
    "channels:\n  emhc: 0\n  laminin: 1\n  type_i: null\n  type_iia: null\n  type_iib: null\n"
    "  type_iix: null\n  dapi: null\nclassification:\n  residual_inference:\n"
    "    enabled: false\n    target_class: null\n    requires_negative_markers: []\n"
)
CLASSES = ["emhc", "negative"]
LABELS = {"negative": "eMHC⁻", "emhc": "eMHC⁺"}
TRAINING_MICE = ("m1", "m2", "m3")
TEST_MOUSE = "m4"


def _truth(fiber_id: int) -> str:
    # The fixture's Type I fibers carry the stain that this panel reads as eMHC.
    return "emhc" if FOUR_CLASS_TRUTH[(fiber_id - 1) % 16 + 1] == "i" else "negative"


def _emhc_image(seed: int) -> tuple[np.ndarray, np.ndarray]:
    image, labels = _tiled(seed)
    type_i, _iia, laminin, _iib = image
    return np.stack([type_i, laminin]), labels


# --- classes and features (unit) ------------------------------------------------------------


def test_emhc_panel_gets_emhc_classes_and_labels():
    markers = ("emhc",)
    assert default_classes(markers) == CLASSES
    assert negative_display_label(markers) == "eMHC⁻"
    assert class_labels_for(CLASSES, "eMHC⁻") == LABELS
    assert validate_classes(["eMHC", "negative"], markers) == CLASSES


def test_emhc_class_is_rejected_without_the_stain_or_with_myosin_classes():
    with pytest.raises(ValueError, match="no eMHC channel"):
        validate_classes(["emhc", "negative"], ("i", "iia"))
    with pytest.raises(ValueError, match="on its own"):
        validate_classes(["emhc", "i", "negative"], ("i", "emhc"))
    with pytest.raises(ValueError, match="on its own"):
        validate_classes(["emhc", "hybrid"], ("emhc",))


def test_myosin_panels_with_emhc_keep_fiber_type_classes_and_features():
    markers = ("i", "iia", "emhc")
    assert default_classes(markers) == ["i", "iia", "negative"]
    assert negative_display_label(markers) == "I⁻/IIa⁻"
    assert class_labels_for(["i", "iia", "negative"], "I⁻/IIa⁻") == {"negative": "I⁻/IIa⁻"}
    features = training_features(markers, ["i", "iia", "negative"])
    assert features == training_features(("i", "iia"))
    assert not any(name.startswith("emhc.") for name in features)


def test_emhc_features_are_the_emhc_stain_only():
    features = training_features(("emhc",), CLASSES)
    assert len(features) == 9
    assert all(name.startswith("emhc.") for name in features)
    assert not any(token in name for name in features for token in ("area", "dapi", "nucle"))
    assert training_features(("emhc",)) == []  # without eMHC classes nothing is typed from eMHC
    assert ordered_classes({"negative", "emhc"}) == ["emhc", "negative"]


def test_project_folder_on_an_emhc_panel(tmp_path):
    root = tmp_path / "proj"
    (root / "images").mkdir(parents=True)
    (root / "panel.yaml").write_text(EMHC_PANEL)
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
    assert project.classes == tuple(CLASSES)
    assert project.negative_label == "eMHC⁻"


# --- end to end -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def emhc_project(tmp_path_factory):
    root = tmp_path_factory.mktemp("emhc")
    (root / "images").mkdir()
    panel = root / "panel.yaml"
    panel.write_text(EMHC_PANEL)
    batch = root / "batch"
    sheet_rows, summary_rows = [], []
    for seed, mouse in enumerate((*TRAINING_MICE, TEST_MOUSE)):
        image_id = f"syn_{mouse}"
        image, labels = _emhc_image(seed)
        _measure(root, image_id, image, labels, panel, batch)
        sheet_rows.append(f"{image_id},{mouse},images/{image_id}.tif")
        summary_rows.append(f"{image_id},success")
    (batch / "batch_summary.csv").write_text("image_name,status\n" + "\n".join(summary_rows))
    sheet = root / "samples.csv"
    sheet.write_text("image_id,mouse_id,raw_image_path\n" + "\n".join(sheet_rows))
    review = root / "review"
    review.mkdir()
    manifest, _ = build_project(
        batch, sheet, panel, review, project_id="emhc", project_name="emhc",
        classes=CLASSES, negative_label="eMHC⁻",
    )  # fmt: skip
    (review / "project.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
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


def test_emhc_panel_builds_evaluates_promotes_and_rolls_back(emhc_project, tmp_path):
    root = tmp_path / "project"
    shutil.copytree(emhc_project, root)
    project = load_project(root / "review" / "project.yaml")
    assert project.review_classes == tuple(CLASSES)
    assert project.class_labels == LABELS

    config = load_channel_config(root / "panel.yaml")
    assert config.available_markers == ()  # no myosin marker
    markers = config.panel_markers
    assert markers == ("emhc",)
    result = build(
        project, blind_label_sessions(project), classes=CLASSES, markers=markers,
        output_dir=root / "models" / "build_20260101T000000Z", class_labels=project.class_labels,
    )  # fmt: skip
    assert set(result["labels"]["by_class"]) == set(CLASSES)
    chosen = result["selection"]["decision"]["recommended"]
    assert chosen, result["selection"]["decision"]["reason"]

    artifact = find_build_candidate(root, chosen)
    raw = yaml.safe_load(artifact.with_suffix(".yaml").read_text(encoding="utf-8"))
    assert raw["features"] == training_features(markers, CLASSES)
    assert {name.split(".")[0] for name in raw["features"]} == {"emhc"}
    assert set(raw["required_markers"]) == {"laminin", "emhc"}
    assert raw["outputs"] == CLASSES
    assert raw["class_labels"] == LABELS

    evaluation = evaluate(root, project, chosen)
    assert min(evaluation["pooled_recall_by_class"].values()) >= 0.7

    before = _tree(root)
    row = promote(root, chosen, root / "panel.yaml")
    assert row["independent_test"]["status"] == "current"
    promoted = load_project(root / "review" / "project.yaml")
    assert promoted.review_classes == tuple(CLASSES)
    assert promoted.class_labels == LABELS
    calls = pd.concat(
        pd.read_csv(root / f"batch/syn_{mouse}/syn_{mouse}_fibers.csv")
        for mouse in (*TRAINING_MICE, TEST_MOUSE)
    )
    assert set(calls["fiber_type"]) == set(CLASSES)
    truth = calls["label"].map(_truth) if "label" in calls else calls["fiber_id"].map(_truth)
    assert (calls["fiber_type"] == truth).mean() >= 0.9

    rollback(root)
    assert _tree(root) == before
