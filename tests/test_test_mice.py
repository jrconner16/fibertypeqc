"""Automatic test mice for projects without a model, and review of every mouse after promotion."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from src import cli
from src.review.roles import (
    POOL,
    RECORD_FILENAME,
    ROLES_FILENAME,
    TEST,
    default_split,
    held_out_mice_released,
    load_roles,
    write_default_roles,
)
from tests.test_model_builder import (  # noqa: F401
    TEST_MOUSE,
    TRAINING_MICE,
    built_project,
    labelled,
)
from tests.test_model_evaluation_promotion import _skip_retyping, built  # noqa: F401
from tests.test_no_model_project import _project_folder


def _mice(n: int, groups: int = 1) -> dict[str, str]:
    return {f"mouse_{i:02d}": f"group_{i % groups}" for i in range(n)}


def test_split_sets_aside_about_a_quarter_and_keeps_the_training_minimum():
    for n, expected in ((3, 0), (4, 1), (8, 2), (12, 3), (20, 5)):
        roles = default_split(_mice(n), minimum_training=3)
        assert sum(role == TEST for role in roles.values()) == expected
        assert sum(role == POOL for role in roles.values()) >= 3
    assert set(default_split(_mice(2), 3).values()) == {POOL}  # too few: no test mice


def test_split_is_repeatable_and_ignores_input_order():
    mice = _mice(12, groups=3)
    reversed_mice = dict(reversed(list(mice.items())))
    assert default_split(mice, 3) == default_split(reversed_mice, 3)


def test_test_mice_are_spread_over_groups_and_each_group_keeps_a_training_mouse():
    mice = {**{f"wt_{i}": "wt" for i in range(6)}, **{f"ko_{i}": "ko" for i in range(6)}}
    roles = default_split(mice, 3)
    test = [mouse for mouse, role in roles.items() if role == TEST]
    assert len(test) == 3
    assert {mouse.split("_")[0] for mouse in test} == {"wt", "ko"}
    # A group of one is never the only source of a test mouse while others can give one.
    roles = default_split({"a1": "a", "a2": "a", "a3": "a", "a4": "a", "b1": "b"}, 3)
    assert roles["b1"] == POOL


def _review_project(folder, mice: dict[str, dict]):
    images = tuple(
        SimpleNamespace(image_id=f"img_{mouse}", mouse_id=mouse, condition=condition)
        for mouse, condition in mice.items()
    )
    return SimpleNamespace(root=folder / "review", images=images, model_version="none")


def test_written_roles_are_recorded_readable_and_never_overwritten(tmp_path):
    project = _review_project(
        tmp_path, {f"m{i}": {"genotype": "wt" if i % 2 else "ko"} for i in range(8)}
    )
    roles = write_default_roles(project, 3)
    assert load_roles(project) == roles
    table = pd.read_csv(tmp_path / ROLES_FILENAME)
    assert list(table.columns) == ["mouse_id", "role", "group"]
    record = json.loads((tmp_path / RECORD_FILENAME).read_text())
    assert record["rule"] == "default_mouse_split.v1"
    assert record["test_mice"] == sorted(m for m, r in roles.items() if r == TEST)
    assert record["inputs"] == {"mice": 8, "groups": ["genotype=ko", "genotype=wt"]}
    with pytest.raises(FileExistsError):
        write_default_roles(project, 3)


def _prepared(tmp_path, model="none", mice=6):
    root = _project_folder(tmp_path, model=model)
    images = [{"image_id": f"img_{i}", "mouse_id": f"m{i}", "condition": {}} for i in range(mice)]
    (root / "review").mkdir()
    (root / "review" / "project.yaml").write_text(yaml.safe_dump({"images": images}))
    return cli.load_project_folder(root)


@pytest.fixture()
def stub_review_project(monkeypatch):
    """load_project without a full review project (only images are read)."""
    from src.review import project as review_project

    def load(path):
        raw = yaml.safe_load(path.read_text())
        images = tuple(SimpleNamespace(**image) for image in raw["images"])
        return SimpleNamespace(root=path.parent, images=images, model_version="none")

    monkeypatch.setattr(review_project, "load_project", load)


def test_prepare_sets_test_mice_aside_once_in_a_no_model_project(
    tmp_path, stub_review_project, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "_run", lambda *args: 0)
    project = _prepared(tmp_path, mice=8)
    message = cli.ensure_test_mice(project)
    assert message.startswith("2 of 8 mice are set aside")
    first = (project.root / ROLES_FILENAME).read_bytes()
    assert cli.ensure_test_mice(project) == ""  # an existing split is never changed
    assert (project.root / ROLES_FILENAME).read_bytes() == first
    assert cli.mouse_split_summary(project) == (2, 8)


def test_no_split_is_written_with_a_model_by_hand_or_after_labelling(tmp_path, stub_review_project):
    typed = _prepared(tmp_path / "typed", model="some_model_id")
    assert cli.ensure_test_mice(typed) == ""
    assert not (typed.root / ROLES_FILENAME).exists()

    by_hand = _prepared(tmp_path / "hand")
    (by_hand.root / ROLES_FILENAME).write_text("mouse_id,role\nm0,test\n")
    assert cli.ensure_test_mice(by_hand) == ""
    assert (by_hand.root / ROLES_FILENAME).read_text() == "mouse_id,role\nm0,test\n"

    labelled_already = _prepared(tmp_path / "labelled")
    state = labelled_already.review_dir / "reference" / "a" / "review_state.json"
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"object_decisions": [{}]}))
    assert cli.ensure_test_mice(labelled_already) == ""
    assert not (labelled_already.root / ROLES_FILENAME).exists()


def test_too_few_mice_says_independent_testing_is_not_possible(tmp_path, stub_review_project):
    project = _prepared(tmp_path, mice=3)
    message = cli.ensure_test_mice(project)
    assert "none can be set aside" in message and "at least 4" in message
    assert set(pd.read_csv(project.root / ROLES_FILENAME)["role"]) == {POOL}


def test_test_mice_are_released_only_for_the_promoted_model(tmp_path):
    project = SimpleNamespace(root=tmp_path / "review", model_version="none")
    assert not held_out_mice_released(project)
    project.model_version = "built.v1"
    assert not held_out_mice_released(project)  # a model, but not promoted from a build here
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "independent_test_checks.csv").write_text(
        "changed_at_utc,to_model,status,override,evaluation\nt,built.v1,current,False,e\n"
    )
    assert held_out_mice_released(project)
    project.model_version = "other.v2"  # e.g. after rolling back to an earlier model
    assert not held_out_mice_released(project)


def _guided(project, blind: bool):
    from src.review.fiber_type_review import FiberTypeReviewController
    from src.review.guided_review_widget import GuidedReviewWidget
    from src.review.session import ReviewSession

    view = project.for_reference("r") if blind else project
    session = ReviewSession(project_id=project.project_id, model_version=project.model_version)
    return GuidedReviewWidget(view, FiberTypeReviewController(session), blind=blind)


def test_after_promotion_guided_review_offers_every_mouse(built, monkeypatch):  # noqa: F811
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    qtpy = pytest.importorskip("qtpy.QtWidgets")
    from src.model_evaluation import evaluate
    from src.promote_model import promote
    from src.review.project import load_project

    application = qtpy.QApplication.instance() or qtpy.QApplication([])
    root, project, chosen = built
    evaluate(root, project, chosen)
    _skip_retyping(monkeypatch)
    promote(root, chosen, root / "panel.yaml")
    promoted = load_project(root / "review" / "project.yaml")
    review = _guided(promoted, blind=False)
    assert review.allowed_images is None
    assert set(review.rows["mouse_id"]) == {*TRAINING_MICE, TEST_MOUSE}
    blind = _guided(promoted, blind=True)  # blind labelling stays on test mice
    assert set(blind.rows["mouse_id"]) == {TEST_MOUSE}

    # Without the recorded independent test (e.g. a model set by hand), review stays on pool mice.
    (root / "models" / "independent_test_checks.csv").unlink()
    held_back = _guided(promoted, blind=False)
    assert set(held_back.rows["mouse_id"]) == set(TRAINING_MICE)
    for widget in (review, blind, held_back):
        widget.close()
        widget.deleteLater()
    application.processEvents()


def test_status_warns_when_test_mice_are_changed_after_the_split(tmp_path, stub_review_project):
    project = _prepared(tmp_path, mice=8)
    cli.ensure_test_mice(project)
    assert cli.split_changed_note(project) == ""
    table = pd.read_csv(project.root / ROLES_FILENAME, dtype=str)
    table["role"] = POOL
    table.loc[0, "role"] = TEST
    table.to_csv(project.root / ROLES_FILENAME, index=False)
    assert "differ from the automatic choice" in cli.split_changed_note(project)
