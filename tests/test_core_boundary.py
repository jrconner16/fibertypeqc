"""The supported core must not depend on research tooling."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_DIRECTORIES = ("src", "fibertypeqc", "scripts")


def _imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def test_core_does_not_import_research_modules():
    offenders = []
    for directory in CORE_DIRECTORIES:
        for path in sorted((REPO_ROOT / directory).rglob("*.py")):
            for module in _imported_modules(path):
                if module == "research" or module.startswith("research."):
                    offenders.append(f"{path.relative_to(REPO_ROOT)} imports {module}")
    assert offenders == []
