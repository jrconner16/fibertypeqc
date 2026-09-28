from __future__ import annotations

from scripts.check_repository import (
    broken_documentation_links,
    forbidden_tracked_artifacts,
    private_absolute_paths,
)


def test_documentation_link_check_requires_tracked_target(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text("See [missing](missing.md).\n")

    problems = broken_documentation_links(tmp_path, {"docs/guide.md"})

    assert problems == ["docs/guide.md:1: target is not tracked: missing.md"]


def test_documentation_link_check_ignores_external_and_fenced_links(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text(
        "See [tracked](target.md) and [web](https://example.com).\n"
        "```markdown\n"
        "[illustrative missing link](not-real.md)\n"
        "```\n"
    )
    (docs / "target.md").write_text("# Target\n")
    tracked = {"docs/guide.md", "docs/target.md"}

    assert broken_documentation_links(tmp_path, tracked) == []


def test_forbidden_artifact_check_allows_only_declared_synthetic_tiffs():
    tracked = {
        "examples/reference/synthetic_reference.tif",
        "examples/reference/synthetic_reference_labels.tif",
        "images/private_section.czi",
        "outputs/run/image_summary.csv",
    }

    problems = forbidden_tracked_artifacts(tracked)

    assert problems == [
        "forbidden tracked microscopy file: images/private_section.czi",
        "forbidden tracked output/private path: outputs/run/image_summary.csv",
    ]


def test_private_absolute_path_check_reports_local_and_hpc_paths(tmp_path):
    paths = (
        "/" + "Users/researcher/private/image.czi",
        "/" + "Volumes/private/data.csv",
        "/" + "home/researcher/private/labels.csv",
        "/" + "temp_work/researcher/private/results.csv",
    )
    (tmp_path / "notes.md").write_text("\n".join(paths) + "\n")

    problems = private_absolute_paths(tmp_path, {"notes.md"})

    assert problems == [
        "notes.md:1: private absolute path",
        "notes.md:2: private absolute path",
        "notes.md:3: private absolute path",
        "notes.md:4: private absolute path",
    ]
