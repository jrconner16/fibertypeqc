"""Nucleus positions carried from the nuclear stage into finalized tables, summaries, and report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import yaml

from fibertypeqc.cohort_report import build_cohort_report
from src.nuclear_association import associate_nuclei, summarize_fiber_nuclei
from src.review.finalization import finalize_project
from src.review.schemas import ObjectReviewStatus
from src.summarize_results import write_results
from tests.test_review_finalization import _decide, _project, _session


def _write_nuclei(directory: Path, image_id: str, legacy_names: bool = False) -> None:
    """Fiber 1 and 4 get a central nucleus, fiber 2 a peripheral one, plus one interstitial."""
    labels = tifffile.imread(directory / f"{image_id}_labels.tif").astype(np.int32)
    nuclei = np.zeros_like(labels)
    nuclei[3:5, 3:5] = 1  # centre of fiber 1
    nuclei[2, 12] = 2  # corner of fiber 2
    nuclei[9, 9] = 3  # between fibers
    nuclei[13:15, 13:15] = 4  # centre of fiber 4
    table, _links = associate_nuclei(
        nuclei, labels, boundary_distance_px=0.5, central_normalized_radius=0.6
    )
    if legacy_names:  # tables written before the positions were renamed
        table["association_category"] = table["association_category"].replace(
            {
                "central": "central_interior",
                "peripheral": "boundary_associated",
                "interstitial": "unassigned_or_interstitial",
            }
        )
    folder = directory / "nuclear"
    folder.mkdir()
    table.to_csv(folder / f"{image_id}_nuclei.csv", index=False)
    summarize_fiber_nuclei(labels, table).to_csv(
        folder / f"{image_id}_fiber_nuclei.csv", index=False
    )


def test_nuclei_reach_final_tables_summaries_and_report(tmp_path):
    project = _project(tmp_path)
    _write_nuclei(tmp_path / "one", "one")
    session = _session(project)
    _decide(session, 4, ObjectReviewStatus.EXCLUDED)  # fiber 4 leaves the analysis

    manifest = finalize_project(project, session, tmp_path / "final")

    final = pd.read_csv(tmp_path / "final" / "final_fiber_table.csv").set_index("label")
    assert final["nuclei_count"].tolist() == [1, 1, 0, 1]
    assert final["central_nuclei_count"].tolist() == [1, 0, 0, 1]
    assert final["peripheral_nuclei_count"].tolist() == [0, 1, 0, 0]
    assert final["centrally_nucleated"].tolist() == [True, False, False, True]
    assert {"fiber_nuclei_summary", "nuclei_table"} <= set(manifest["images"][0]["input_sha256"])
    counts = pd.read_csv(tmp_path / "final" / "nuclei_counts.csv").iloc[0]
    assert (
        counts["n_nuclei_central"],
        counts["n_nuclei_peripheral"],
        counts["n_nuclei_interstitial"],
        counts["n_nuclei_ambiguous"],
    ) == (2, 1, 1, 0)

    project_yaml = tmp_path / "project.yaml"
    raw = yaml.safe_load(project_yaml.read_text())
    raw["images"][0]["condition"] = {"genotype": "mdx"}
    project_yaml.write_text(yaml.safe_dump(raw))
    results = write_results(tmp_path / "final", tmp_path / "results", project_yaml)

    mouse = pd.read_csv(tmp_path / "results" / "mouse_summary.csv").iloc[0]
    assert mouse["n_fibers_nuclei_measured"] == 3  # the excluded fiber is not counted
    assert mouse["n_centrally_nucleated"] == 1
    assert mouse["prop_centrally_nucleated"] == 1 / 3
    assert mouse["prop_centrally_nucleated_iia"] == 1.0  # fiber 1, final type IIa
    assert mouse["prop_centrally_nucleated_iib"] == 0.0
    assert mouse["mean_nuclei_per_fiber"] == 2 / 3
    assert mouse["n_nuclei_interstitial"] == 1
    assert "centrally_nucleated" in results["definitions"]
    cohort = pd.read_csv(tmp_path / "results" / "nuclear_cohort_summary.csv")
    overall = cohort[cohort["fiber_class"] == "all"].iloc[0]
    assert (overall["genotype"], overall["n_mice"], overall["mean_proportion"]) == (
        "mdx",
        1,
        1 / 3,
    )

    report = build_cohort_report(tmp_path / "results")
    assert "<h2>Nuclei</h2>" in report and "<b>Interstitial</b>" in report
    assert "Centrally nucleated mean ± SD" in report and "33.3%" in report


def test_old_position_names_give_the_same_counts(tmp_path):
    project = _project(tmp_path)
    _write_nuclei(tmp_path / "one", "one", legacy_names=True)

    finalize_project(project, _session(project), tmp_path / "final")

    counts = pd.read_csv(tmp_path / "final" / "nuclei_counts.csv").iloc[0]
    assert (counts["n_nuclei_central"], counts["n_nuclei_peripheral"]) == (2, 1)
    assert counts["n_nuclei_interstitial"] == 1


def test_without_dapi_nothing_nuclear_is_added(tmp_path):
    project = _project(tmp_path)
    (tmp_path / "final").mkdir()
    (tmp_path / "final" / "nuclei_counts.csv").write_text("left over\n")

    finalize_project(project, _session(project), tmp_path / "final")
    results = write_results(tmp_path / "final", tmp_path / "results", None)

    final = pd.read_csv(tmp_path / "final" / "final_fiber_table.csv")
    assert not {"nuclei_count", "centrally_nucleated"} & set(final.columns)
    assert not (tmp_path / "final" / "nuclei_counts.csv").exists()  # stale file removed
    mouse = pd.read_csv(tmp_path / "results" / "mouse_summary.csv")
    assert not any(column.startswith(("n_nuclei", "prop_centrally")) for column in mouse.columns)
    assert "centrally_nucleated" not in results["definitions"]
    assert "<h2>Nuclei</h2>" not in build_cohort_report(tmp_path / "results")
