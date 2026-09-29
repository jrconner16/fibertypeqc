from __future__ import annotations

from scripts.check_repository import (
    broken_documentation_links,
    forbidden_tracked_artifacts,
    load_private_denylist,
    private_absolute_paths,
    private_identifier_matches,
    unapproved_fiber_tables,
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


def test_forbidden_artifact_check_rejects_unapproved_model_artifacts():
    problems = forbidden_tracked_artifacts(
        {
            "data/models/rebaseline_tile_v2_p75p90_iib_iia_iix.joblib",
            "data/models/private_candidate.joblib",
            "outputs_elsewhere/weights.pt",
        }
    )

    assert problems == [
        "model artifact not approved for public release: data/models/private_candidate.joblib",
        "model artifact not approved for public release: outputs_elsewhere/weights.pt",
    ]


def test_fiber_table_check_allows_only_examples_and_tests(tmp_path):
    for relative in ("examples/demo.csv", "docs/results.csv", "docs/issues.csv"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "examples/demo.csv").write_text("fiber_id,fiber_type\n1,iia\n")
    (tmp_path / "docs/results.csv").write_text("image_id,fiber_id,fiber_type\n")
    (tmp_path / "docs/issues.csv").write_text("title,body\n")

    problems = unapproved_fiber_tables(
        tmp_path, {"examples/demo.csv", "docs/results.csv", "docs/issues.csv"}
    )

    assert problems == ["per-fiber table outside examples/ or tests/: docs/results.csv"]


def test_private_path_check_reports_cloud_backup_locations(tmp_path):
    (tmp_path / "notes.md").write_text(
        "backup in Google" + " Drive\nsynced to Library/Cloud" + "Storage/x\nclean line\n"
    )

    assert private_absolute_paths(tmp_path, {"notes.md"}) == [
        "notes.md:1: private absolute path",
        "notes.md:2: private absolute path",
    ]


def test_private_denylist_is_optional_and_ignores_comments(tmp_path):
    assert load_private_denylist(tmp_path / "missing.txt") is None

    denylist = tmp_path / "denylist.txt"
    denylist.write_text("# comment\n\nspecimen_[0-9]+\n")
    patterns = load_private_denylist(denylist)

    assert [pattern.pattern for pattern in patterns] == ["specimen_[0-9]+"]


def test_private_identifier_scan_reports_location_without_echoing_match(tmp_path):
    (tmp_path / "specimen_42_notes.md").write_text("safe\nsee specimen_42 here\n")
    patterns = load_private_denylist_from_lines(tmp_path, ["specimen_[0-9]+"])

    problems = private_identifier_matches(tmp_path, {"specimen_42_notes.md"}, patterns)

    assert problems == [
        "specimen_42_notes.md: path matches private denylist",
        "specimen_42_notes.md:2: matches private denylist",
    ]
    assert all("see specimen" not in problem for problem in problems)


def load_private_denylist_from_lines(tmp_path, lines):
    path = tmp_path / "denylist.txt"
    path.write_text("\n".join(lines) + "\n")
    return load_private_denylist(path)
