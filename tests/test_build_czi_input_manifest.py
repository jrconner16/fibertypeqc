from pathlib import Path

import pytest

from src.build_czi_input_manifest import build_manifest_rows


def test_build_manifest_rows_filters_explicit_duplicate_patterns(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    canonical = raw / "mdx" / "351545_L" / "001_quad.czi"
    canonical.parent.mkdir(parents=True)
    canonical.touch()
    (canonical.parent / "001_quad-Change Scaling-01.czi").touch()
    (canonical.parent / "001_quad-Scene-1-TR1.czi").touch()

    rows = build_manifest_rows(raw, ("Change Scaling", "Scene-"))

    assert rows == [
        {
            "image_id": "mdx_351545_L_001_quad",
            "input_relpath": "mdx/351545_L/001_quad.czi",
        }
    ]


def test_build_manifest_rows_rejects_empty_result(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "only-duplicate.czi").touch()

    with pytest.raises(ValueError, match="No CZI files remain"):
        build_manifest_rows(raw, ("duplicate",))
