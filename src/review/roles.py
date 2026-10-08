"""Optional evaluation roles that keep training and test mice apart (no GUI imports).

A project may contain ``evaluation_roles.csv`` (columns ``mouse_id``, ``role``) beside its
``fibertypeqc_project.yaml``. Roles are ``pool`` (mice whose review decisions may train a model)
and ``test`` (mice labelled blind and never trained on). With the file present, blind labelling
is limited to test mice, guided review and the model improver to pool mice. Without it, nothing
is restricted.
"""

from __future__ import annotations

import pandas as pd

from src.review.project import Project

ROLES_FILENAME = "evaluation_roles.csv"
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
