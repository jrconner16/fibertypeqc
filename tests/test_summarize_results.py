from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fibertypeqc.cohort_report import build_cohort_report
from src.summarize_results import main, summarize, write_results


def _fibers(image_id, mouse_id, rows):
    """rows: (model call, final type, value source, exclusion reason, roi)"""
    return pd.DataFrame(
        [
            {
                "image_id": image_id,
                "mouse_id": mouse_id,
                "section_id": "s1",
                "label": index + 1,
                "fiber_type": model,
                "area": 100.0 * (index + 1),
                "final_type": final,
                "value_source": source,
                "exclusion_reason": reason,
                "flagged_unreviewed": source == "predicted" and index == 0,
                "roi_name": roi,
                "roi_role": "half" if roi else "",
            }
            for index, (model, final, source, reason, roi) in enumerate(rows)
        ]
    )


def _final_table() -> pd.DataFrame:
    return pd.concat(
        [
            _fibers(
                "a1",
                "mouse_a",
                [
                    ("i", "i", "predicted", "", "left"),
                    ("iia", "iib", "reviewed", "", "left"),
                    ("iix", "", "unresolved", "", "right"),
                    ("iib", "", "excluded", "region:exclude_all_analysis:fold", ""),
                ],
            ),
            _fibers("a2", "mouse_a", [("iia", "iia", "predicted", "", "")]),
            _fibers(
                "b1",
                "mouse_b",
                [("i", "i", "predicted", "", ""), ("iia", "iia", "reviewed", "", "")],
            ),
            _fibers("c1", "mouse_c", [("iib", "", "excluded", "section_not_selected", "")]),
        ],
        ignore_index=True,
    )


def test_composition_denominators_follow_policy():
    image = summarize(_final_table())["image_summary"].set_index("image_id")

    a1 = image.loc["a1"]
    assert (a1.n_fibers_total, a1.n_excluded, a1.n_analysis, a1.n_resolved) == (4, 1, 3, 2)
    assert a1.n_unresolved == 1 and a1.unresolved_share == pytest.approx(1 / 3)
    # Finalized: resolved fibers only (i, iib); predicted: model calls of all analysis fibers.
    assert a1.prop_final_i == 0.5 and a1.prop_final_iib == 0.5 and a1.prop_final_iia == 0
    assert a1.prop_predicted_i == pytest.approx(1 / 3)
    assert a1.prop_predicted_iia == pytest.approx(1 / 3)
    assert a1.prop_predicted_iix == pytest.approx(1 / 3)
    assert a1.n_corrected == 1 and a1.n_reviewed == 1
    assert a1.n_excluded_region == 1 and a1.n_flagged_unreviewed == 1
    assert image.loc["c1"].n_excluded_section == 1
    assert pd.isna(image.loc["c1"].prop_final_i)


def test_mouse_level_pools_sections_and_classes_are_ordered():
    outputs = summarize(_final_table())
    mouse = outputs["mouse_summary"].set_index("mouse_id")

    assert mouse.loc["mouse_a", "n_images"] == 2
    assert mouse.loc["mouse_a", "n_resolved"] == 3
    assert mouse.loc["mouse_a", "prop_final_iia"] == pytest.approx(1 / 3)
    assert outputs["_classes"]["fiber_class"].tolist() == ["i", "iia", "iix", "iib"]


def test_roi_and_cohort_summaries():
    conditions = {
        "a1": {"genotype": "mdx"},
        "a2": {"genotype": "mdx"},
        "b1": {"genotype": "mdx"},
        "c1": {"genotype": "wt"},
    }
    outputs = summarize(_final_table(), conditions)

    roi = outputs["roi_summary"].set_index("roi_name")
    assert roi.loc["left", "n_resolved"] == 2 and roi.loc["right", "n_unresolved"] == 1
    cohort = outputs["cohort_summary"]
    mdx_i = cohort[
        (cohort.genotype == "mdx") & (cohort.fiber_class == "i") & (cohort.view == "final")
    ].iloc[0]
    assert mdx_i.n_mice == 2
    assert mdx_i.mean_proportion == pytest.approx((1 / 3 + 1 / 2) / 2)
    wt = cohort[(cohort.genotype == "wt") & (cohort.view == "final")]
    assert set(wt.n_mice) == {0}  # all fibers excluded: no mouse-level value


def test_conflicting_conditions_within_a_mouse_are_rejected():
    with pytest.raises(ValueError, match="different conditions"):
        summarize(_final_table(), {"a1": {"genotype": "mdx"}, "a2": {"genotype": "wt"}})


def test_cli_writes_tables_manifest_and_report_from_exported_tables(tmp_path: Path):
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    _final_table().to_csv(final_dir / "final_fiber_table.csv", index=False)
    (final_dir / "finalization_manifest.json").write_text(
        json.dumps({"project_id": "p", "model_version": "m", "images": []})
    )
    project = tmp_path / "project.yaml"
    project.write_text(
        yaml.safe_dump(
            {
                "images": [
                    {"image_id": i, "condition": {"genotype": g}}
                    for i, g in (("a1", "mdx"), ("a2", "mdx"), ("b1", "wt"), ("c1", "wt"))
                ]
            }
        )
    )

    assert (
        main(
            [
                "--final-dir",
                str(final_dir),
                "--output-dir",
                str(tmp_path / "r"),
                "--project",
                str(project),
            ]
        )
        == 0
    )

    results = tmp_path / "r"
    manifest = json.loads((results / "results_manifest.json").read_text())
    assert set(manifest["tables"]) == {
        "image_summary",
        "mouse_summary",
        "roi_summary",
        "cohort_summary",
    }
    assert manifest["definitions"]["mouse_level"] == "fibers pooled across sections"
    report = (results / "cohort_report.html").read_text()
    assert (
        "Composition before and after review" in report and "Cohort (mouse as the unit)" in report
    )
    # The report is a pure function of the exported tables.
    assert build_cohort_report(results).split("UTC</p>")[1] == report.split("UTC</p>")[1]


def test_rewriting_results_is_deterministic(tmp_path: Path):
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    _final_table().to_csv(final_dir / "final_fiber_table.csv", index=False)

    first = write_results(final_dir, tmp_path / "one", None)
    second = write_results(final_dir, tmp_path / "two", None)

    assert {k: v["sha256"] for k, v in first["tables"].items()} == {
        k: v["sha256"] for k, v in second["tables"].items()
    }
