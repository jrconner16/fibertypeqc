"""The version recorded in outputs must match the released version metadata."""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

from fibertypeqc import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_package_version_matches_pyproject_and_citation():
    # `application_version` in every run artifact comes from fibertypeqc.__version__.
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert __version__ == pyproject["project"]["version"] == str(citation["version"])
