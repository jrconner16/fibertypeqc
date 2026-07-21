from __future__ import annotations

from pathlib import Path

import pytest

from src.run_panel_array_task import _load_manifest


def test_load_manifest_requires_safe_relative_inputs(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,input_relpath\nmouse_a,cohort/image.czi\n")

    assert _load_manifest(manifest)[0].input_relpath == Path("cohort/image.czi")

    manifest.write_text("image_id,input_relpath\nmouse_a,../outside.czi\n")
    with pytest.raises(ValueError, match="unsafe input_relpath"):
        _load_manifest(manifest)
