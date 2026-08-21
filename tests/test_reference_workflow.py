from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from scripts.generate_reference_fixture import generate_fixture
from scripts.run_reference import run_reference


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_reference_fixture_generator_is_deterministic(tmp_path):
    image_path, labels_path = generate_fixture(tmp_path)

    expected_image = "2a02b24b3411fc073f81431a4e953f33867ce84a8b53ecdb47f62e4d7cf2274b"
    expected_labels = "af974e2ba9306c6aa461105226df073800df896871d78551ad88f7be4775206e"
    assert _digest(image_path) == expected_image
    assert _digest(labels_path) == expected_labels


def test_reference_workflow_runs_and_validates(tmp_path):
    run_reference(tmp_path)

    assert (tmp_path / "synthetic_reference_fibers.csv").is_file()
    assert (tmp_path / "synthetic_reference_fibers_final.csv").is_file()
