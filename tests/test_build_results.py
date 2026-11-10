"""What Build will use and why a build recommended a model or none (launcher, issue #84)."""

from __future__ import annotations

import sys

import pandas as pd
import pytest

from src import cli
from src import workflow_steps as ws
from src.model_builder import BuilderError, check_minimums, label_progress, load_recipe
from tests.test_model_builder import (  # noqa: F401
    TEST_MOUSE,
    TRAINING_MICE,
    built_project,
    labelled,
)
from tests.test_model_evaluation_promotion import built  # noqa: F401

CLASSES = ["i", "iia", "negative"]


def _table(counts: dict[str, dict[str, int]]) -> pd.DataFrame:
    """Combined labels: {mouse: {class: n}}."""
    rows = [
        {"mouse_id": mouse, "label": name}
        for mouse, per_class in counts.items()
        for name, n in per_class.items()
        for _ in range(n)
    ]
    return pd.DataFrame(rows, columns=["mouse_id", "label"])


def test_progress_matches_what_build_checks():
    recipe = load_recipe()
    short = _table({"m1": {"i": 20, "iia": 3}, "m2": {"i": 20, "iia": 30, "negative": 4}})
    progress = label_progress(short, CLASSES, recipe)
    assert progress["training_mice"] == 2 and not progress["ready"]
    rows = {row["class"]: row for row in progress["classes"]}
    assert (rows["i"]["labels"], rows["i"]["mice"]) == (40, 2)
    assert (rows["iia"]["labels"], rows["iia"]["mice"]) == (33, 1)  # m1 has fewer than 5
    assert progress["by_mouse"]["m1"] == {"i": 20, "iia": 3, "negative": 0}
    with pytest.raises(BuilderError):
        check_minimums(short, CLASSES, recipe)

    hints = ws.label_hints(progress, {"negative": "I⁻/IIa⁻"})
    assert hints[0].startswith("Label fibers in 1 more training mouse")
    assert "IIa: at least 5 labels in 1 more mouse" in " ".join(hints)
    assert "I⁻/IIa⁻: 26 more labelled fibers (4 of 30)." in hints
    assert not any(hint.startswith("Type I:") for hint in hints)  # Type I is complete

    enough = _table({m: {"i": 10, "iia": 10, "negative": 10} for m in ("a", "b", "c")})
    progress = label_progress(enough, CLASSES, recipe)
    assert progress["ready"] and ws.label_hints(progress) == []
    check_minimums(enough, CLASSES, recipe)  # does not raise


def _summary(recalls: dict[str, dict[str, float]], gains: dict[str, float], passes=()):
    return {
        "selection": {
            "metrics": {
                name: {"median_mouse_macro_f1": 0.5, "pooled_recall_by_class": values}
                for name, values in recalls.items()
            },
            "decision": {
                "recommended": passes[0] if passes else None,
                "reason": "none passed",
                "checks": {
                    name: {
                        "median_mouse_macro_f1_gain_over_baseline": gains[name],
                        "lowest_pooled_class_recall": min(recalls[name].values()),
                        "passes": name in passes,
                    }
                    for name in gains
                },
            },
        }
    }


def test_build_hint_names_the_class_no_model_recognised():
    recipe = load_recipe()
    summary = _summary(
        {"logistic": {"i": 0.9, "iia": 0.6}, "forest": {"i": 0.95, "iia": 0.65}},
        {"logistic": 0.3, "forest": 0.25},
    )
    hint = ws.build_hint(summary, recipe)
    assert "No model recognised IIa well enough" in hint and "Type I" not in hint
    assert "beat always guessing" not in hint

    flat = _summary({"logistic": {"i": 0.8, "iia": 0.8}}, {"logistic": 0.02})
    assert "beat always guessing the most common class by 0.10" in ws.build_hint(flat, recipe)
    passed = _summary({"logistic": {"i": 0.9, "iia": 0.9}}, {"logistic": 0.3}, ("logistic",))
    assert ws.build_hint(passed, recipe) == ""


@pytest.fixture
def qapp(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("qtpy.QtWidgets")
    return widgets.QApplication.instance() or widgets.QApplication([])


def test_dialog_shows_labels_by_mouse_and_the_latest_build(qapp, built):  # noqa: F811
    from src.launcher import BuildResultsDialog

    root, _project, chosen = built
    dialog = BuildResultsDialog(cli.load_project_folder(root))
    progress = dialog.progress
    assert progress["ready"] and set(progress["by_mouse"]) == set(TRAINING_MICE)
    assert progress["audit"]["left_out"]["test_mice"] > 0  # test-mouse labels are not counted
    table = dialog.labels_table
    assert table.rowCount() == 3 and table.columnCount() == 2 + len(TRAINING_MICE)
    assert TEST_MOUSE not in [table.horizontalHeaderItem(c).text() for c in range(4)]
    assert dialog.hint_label.text() == "Enough labels to build."
    rows = [dialog.build_table.verticalHeaderItem(r).text() for r in range(4)]
    assert rows[0] == "majority baseline" and chosen.replace("_", " ") in rows
    assert dialog.build_label.text().startswith(f"Recommended: {chosen.replace('_', ' ')}")
    dialog.close()


def test_builder_rows_offer_the_details_and_a_failed_build_opens_them(qapp, tmp_path):
    from src.launcher import LauncherWindow
    from tests.test_launcher import _prepared, _project

    root = _project(tmp_path, model="none")
    _prepared(root)
    window = LauncherWindow(root)
    label = next(row for row in window.rows if row.step.key == "label")
    assert label.details_button.text() == "Label counts…"
    assert next(row for row in window.rows if row.step.key == "build").details_button is None

    opened = []
    window.show_build_results = lambda: opened.append(True)
    window.process_command = lambda _a: (sys.executable, ["-c", "raise SystemExit(1)"])
    window.start_arguments(["build", str(root)])
    assert window.process.waitForFinished(20000)
    qapp.processEvents()
    assert opened == [True]
    window.close()
