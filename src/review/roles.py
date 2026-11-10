"""Optional evaluation roles that keep training and test mice apart (no GUI imports).

A project may contain ``evaluation_roles.csv`` (columns ``mouse_id``, ``role``) beside its
``fibertypeqc_project.yaml``. Roles are ``pool`` (mice whose review decisions may train a model)
and ``test`` (mice labelled blind and never trained on). With the file present, blind labelling
is limited to test mice and the model builder and improver to pool mice. Guided review is limited
to pool mice until the test mice are released (see ``held_out_mice_released``). Without the file,
nothing is restricted.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256

import pandas as pd

from src.review.project import Project

ROLES_FILENAME = "evaluation_roles.csv"
TEST_CHECKS_PATH = "models/independent_test_checks.csv"  # written by promote_model
POOL, TEST = "pool", "test"


def load_roles(project: Project) -> dict[str, str]:
    """Mouse ID -> role, or an empty mapping when the project declares no roles."""
    path = project.root.parent / ROLES_FILENAME
    if not path.is_file():
        return {}
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"mouse_id", "role"}.issubset(table.columns):
        raise ValueError(f"{path} needs the columns mouse_id and role")
    roles = dict(
        zip(table["mouse_id"].str.strip(), table["role"].str.strip().str.lower(), strict=True)
    )
    unknown = sorted(set(roles.values()) - {POOL, TEST})
    if unknown:
        raise ValueError(f"{path}: role must be 'pool' or 'test', not {', '.join(unknown)}")
    project_mice = {image.mouse_id for image in project.images}
    missing = sorted(project_mice - set(roles))
    if missing:
        raise ValueError(f"{path}: no role for mouse {', '.join(missing[:5])}")
    return roles


def allowed_images(project: Project, role: str) -> set[str] | None:
    """Image IDs with ``role``; None when the project declares no roles (no restriction)."""
    roles = load_roles(project)
    if not roles:
        return None
    return {image.image_id for image in project.images if roles[image.mouse_id] == role}


def held_out_mice(project: Project) -> list[str]:
    """Mouse IDs held out for testing (sorted); empty without roles."""
    return sorted(mouse for mouse, role in load_roles(project).items() if role == TEST)


def held_out_mice_released(project: Project) -> bool:
    """Whether guided review may show the test mice with the current model's calls.

    True once the current model was promoted from a project without a model with its
    independent-test check recorded (a current evaluation on the test mice, or a recorded
    override). Their blind labels are then no longer needed unbiased, so the whole cohort can be
    reviewed. This only affects review: test-mouse decisions are still never trained on.
    """
    if project.model_version in (None, "", "none"):
        return False
    path = project.root.parent / TEST_CHECKS_PATH
    if not path.is_file():
        return False
    checks = pd.read_csv(path, dtype=str, keep_default_na=False)
    return "to_model" in checks.columns and bool(checks["to_model"].eq(project.model_version).any())


# --- automatic test mice (projects without a model) -------------------------------------------

RECORD_FILENAME = "evaluation_roles_record.json"
SPLIT_RULE = "default_mouse_split.v1"
TEST_SHARE = 0.25


def _order(mouse_id: str) -> str:
    """A fixed, name-independent order: the same mice always give the same split."""
    return sha256(f"{SPLIT_RULE}:{mouse_id}".encode()).hexdigest()


def default_split(mouse_groups: dict[str, str], minimum_training: int) -> dict[str, str]:
    """Mouse ID -> ``pool``/``test``: about a quarter of mice set aside to test a built model.

    Split by mouse, never by image. At least ``minimum_training`` mice stay in the pool; with
    too few mice none are set aside. Test mice are spread over the groups (e.g. genotype x
    treatment), taking from larger groups first and leaving every group at least one training
    mouse where possible. Labels and scores are never inputs, so the split cannot be tuned.
    """
    mice = sorted(mouse_groups)
    n_test = min(max(1, round(TEST_SHARE * len(mice))), len(mice) - minimum_training)
    roles = dict.fromkeys(mice, POOL)
    if n_test <= 0:
        return roles
    groups: dict[str, list[str]] = {}
    for mouse in mice:
        groups.setdefault(mouse_groups[mouse], []).append(mouse)
    for members in groups.values():
        members.sort(key=_order)
    order = sorted(groups, key=lambda name: (-len(groups[name]), _order(name)))
    taken = dict.fromkeys(order, 0)
    chosen: list[str] = []
    for keep_one in (True, False):  # first leave each group a training mouse, then relax
        while len(chosen) < n_test:
            progressed = False
            for name in order:
                limit = len(groups[name]) - (1 if keep_one else 0)
                if len(chosen) < n_test and taken[name] < limit:
                    chosen.append(groups[name][taken[name]])
                    taken[name] += 1
                    progressed = True
            if not progressed:
                break
    for mouse in chosen:
        roles[mouse] = TEST
    return roles


def write_default_roles(project: Project, minimum_training: int) -> dict[str, str]:
    """Write ``evaluation_roles.csv`` and its record for ``project`` (never overwrites).

    Groups are the images' condition columns from the sample sheet (one group per mouse).
    """
    folder = project.root.parent
    path = folder / ROLES_FILENAME
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    mouse_groups: dict[str, str] = {}
    for image in project.images:
        condition = " / ".join(f"{k}={v}" for k, v in sorted((image.condition or {}).items()))
        mouse_groups.setdefault(image.mouse_id, condition)
    roles = default_split(mouse_groups, minimum_training)
    pd.DataFrame(
        {
            "mouse_id": list(roles),
            "role": list(roles.values()),
            "group": [mouse_groups[mouse] for mouse in roles],
        }
    ).to_csv(path, index=False)
    record = {
        "rule": SPLIT_RULE,
        "test_share": TEST_SHARE,
        "minimum_training_mice": minimum_training,
        "inputs": {"mice": len(roles), "groups": sorted(set(mouse_groups.values()))},
        "test_mice": sorted(m for m, r in roles.items() if r == TEST),
        "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": (
            "Chosen automatically before any labels, from mouse IDs and groups only. Test mice "
            "are labelled blind and never used for training. Changing roles after test labels "
            "or an evaluation exist weakens the independent test."
        ),
    }
    (folder / RECORD_FILENAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return roles
