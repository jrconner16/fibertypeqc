"""Image, mouse, ROI, and cohort result tables from finalized fiber tables.

Inputs are a finalization output directory (``final_fiber_table.csv`` and its manifest) and,
optionally, the review project YAML for per-image ``condition`` fields. Outputs are CSV tables
that are the scientific source of truth, a ``results_manifest.json``, and (via
``fibertypeqc.cohort_report``) a self-contained HTML report built only from those CSVs.

Definitions (recorded in the manifest):

- analysis fibers: every fiber whose ``value_source`` is not ``excluded``;
- finalized composition: ``final_type`` over resolved analysis fibers (``predicted`` or
  ``reviewed``); unresolved fibers are left out of the denominator and reported separately;
- predicted composition: the model call over all analysis fibers (the same fibers, before
  review), so the two compositions compare like with like;
- mouse level: fibers pooled across the mouse's sections;
- cohort level: one value per mouse (mouse-level finalized proportion), summarized as mean, SD,
  and n per condition group.

With nuclear data (DAPI), also (definitions in the manifest when present):

- centrally nucleated fibers: analysis fibers with at least one central nucleus, over analysis
  fibers with nuclear data; by fiber type, over resolved fibers of that final type;
- nuclei per fiber: mean nuclei inside each analysis fiber;
- nucleus positions per image (central, peripheral, interstitial, ambiguous): whole image, from
  ``nuclei_counts.csv``; images excluded as a whole are left out.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from fibertypeqc.artifacts import file_sha256, git_commit
from fibertypeqc.cohort_report import ID_DTYPES
from src.fiber_type_labels import normalize_review_label

RESULTS_SCHEMA_VERSION = "fibertypeqc.results.v1"
CLASS_ORDER = ("i", "iia", "iix", "iib", "hybrid")
DEFINITIONS = {
    "analysis_fibers": "value_source != excluded",
    "finalized_composition": "final_type over resolved analysis fibers; unresolved reported "
    "separately and excluded from the denominator",
    "predicted_composition": "model call over all analysis fibers",
    "mouse_level": "fibers pooled across sections",
    "cohort_level": "mean, SD, and n of mouse-level finalized proportions per condition group",
}
MORPHOLOGY_COLUMNS = ("area", "area_um2", "feret_min_um", "feret_max_um")
NUCLEAR_DEFINITIONS = {
    "nucleus_positions": "central: inside a fiber, at least halfway from its edge to its centre "
    "(see the nuclear run settings); peripheral: inside a fiber, not central; interstitial: "
    "outside every segmented fiber; ambiguous: no fiber holds most of the nucleus",
    "centrally_nucleated": "analysis fibers with at least one central nucleus over analysis fibers "
    "with nuclear data; by type over resolved fibers of that final type",
    "nuclei_per_fiber": "mean count of nuclei inside each analysis fiber",
    "nucleus_position_counts": "whole image; images excluded as a whole are left out",
}
NUCLEUS_COUNT_COLUMNS = (
    "n_nuclei_central",
    "n_nuclei_peripheral",
    "n_nuclei_interstitial",
    "n_nuclei_ambiguous",
)


def ordered_classes(values: set[str]) -> list[str]:
    known = [name for name in CLASS_ORDER if name in values]
    return known + sorted(values - set(CLASS_ORDER))


def _prepare(final: pd.DataFrame) -> pd.DataFrame:
    table = final.copy()
    for column in ("final_type", "value_source", "exclusion_reason"):
        table[column] = table[column].fillna("").astype(str)
    table["model_class"] = table["fiber_type"].map(normalize_review_label)
    table["final_class"] = table["final_type"].map(
        lambda value: normalize_review_label(value) if value else ""
    )
    table["is_analysis"] = table["value_source"] != "excluded"
    table["is_resolved"] = table["is_analysis"] & table["value_source"].isin(
        ("predicted", "reviewed")
    )
    table["is_corrected"] = (table["value_source"] == "reviewed") & (
        table["final_class"] != table["model_class"]
    )
    table["flagged_unreviewed"] = table.get("flagged_unreviewed", False)
    table["flagged_unreviewed"] = table["flagged_unreviewed"].fillna(False).astype(bool)
    return table


def _group_summary(group: pd.DataFrame, classes: list[str]) -> dict[str, Any]:
    analysis = group[group["is_analysis"]]
    resolved = group[group["is_resolved"]]
    excluded = group[~group["is_analysis"]]
    row: dict[str, Any] = {
        "n_fibers_total": len(group),
        "n_excluded": len(excluded),
        "n_analysis": len(analysis),
        "n_resolved": len(resolved),
        "n_unresolved": int((analysis["value_source"] == "unresolved").sum()),
        "unresolved_share": (
            float((analysis["value_source"] == "unresolved").mean()) if len(analysis) else None
        ),
        "n_reviewed": int((group["value_source"] == "reviewed").sum()),
        "n_corrected": int(group["is_corrected"].sum()),
        "n_flagged_unreviewed": int((analysis["flagged_unreviewed"]).sum()),
    }
    for prefix in (
        "image",
        "section",
        "region",
        "outside_analysis_roi",
        "fiber_review",
        "edge_of_image",
    ):
        row[f"n_excluded_{prefix}"] = int(excluded["exclusion_reason"].str.startswith(prefix).sum())
    for name in classes:
        row[f"n_final_{name}"] = int((resolved["final_class"] == name).sum())
        row[f"prop_final_{name}"] = (
            row[f"n_final_{name}"] / len(resolved) if len(resolved) else None
        )
        row[f"prop_predicted_{name}"] = (
            float((analysis["model_class"] == name).mean()) if len(analysis) else None
        )
    if "centrally_nucleated" in group.columns:
        row.update(_nuclear_summary(analysis, resolved, classes))
    for column in MORPHOLOGY_COLUMNS:
        if column in resolved.columns and len(resolved):
            row[f"median_{column}"] = float(resolved[column].median())
            for name in classes:
                values = resolved.loc[resolved["final_class"] == name, column]
                row[f"median_{column}_{name}"] = float(values.median()) if len(values) else None
    return row


def _flags(values: pd.Series) -> pd.Series:
    """True/False from a CSV column that may hold booleans or their text."""
    return values.map(lambda v: str(v).strip().lower() == "true")


def _nuclear_summary(
    analysis: pd.DataFrame, resolved: pd.DataFrame, classes: list[str]
) -> dict[str, Any]:
    measured = analysis[analysis["centrally_nucleated"].notna()]
    central = _flags(measured["centrally_nucleated"])
    row: dict[str, Any] = {
        "n_fibers_nuclei_measured": len(measured),
        "n_centrally_nucleated": int(central.sum()),
        "prop_centrally_nucleated": float(central.mean()) if len(measured) else None,
        "mean_nuclei_per_fiber": (
            float(pd.to_numeric(measured["nuclei_count"]).mean()) if len(measured) else None
        ),
    }
    typed = resolved[resolved["centrally_nucleated"].notna()]
    for name in classes:
        values = _flags(typed.loc[typed["final_class"] == name, "centrally_nucleated"])
        row[f"prop_centrally_nucleated_{name}"] = float(values.mean()) if len(values) else None
    return row


def _nuclear_cohort(joined: pd.DataFrame, keys: list[str], classes: list[str]) -> pd.DataFrame:
    """Mean, SD, and n across mice of centrally nucleated proportions, per condition group."""
    rows = []
    for group_values, group in joined.groupby(keys, sort=True):
        group_values = group_values if isinstance(group_values, tuple) else (group_values,)
        base = dict(zip(keys, group_values, strict=True))
        for name in ["all", *classes]:
            column = (
                "prop_centrally_nucleated" if name == "all" else f"prop_centrally_nucleated_{name}"
            )
            values = group[column].dropna()
            rows.append(
                {
                    **base,
                    "fiber_class": name,
                    "n_mice": int(len(values)),
                    "mean_proportion": float(values.mean()) if len(values) else None,
                    "sd_proportion": float(values.std(ddof=1)) if len(values) > 1 else None,
                }
            )
    return pd.DataFrame(rows)


def summarize(
    final: pd.DataFrame,
    conditions: dict[str, dict[str, Any]] | None = None,
    nuclei_counts: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    table = _prepare(final)
    classes = ordered_classes(
        set(table.loc[table["is_resolved"], "final_class"]) | set(table["model_class"])
    )
    image = pd.DataFrame(
        [
            {
                "image_id": image_id,
                "mouse_id": group["mouse_id"].iloc[0],
                "section_id": group["section_id"].iloc[0],
                **_group_summary(group, classes),
            }
            for image_id, group in table.groupby("image_id", sort=True)
        ]
    )
    mouse = pd.DataFrame(
        [
            {
                "mouse_id": mouse_id,
                "n_images": group["image_id"].nunique(),
                **_group_summary(group, classes),
            }
            for mouse_id, group in table.groupby("mouse_id", sort=True)
        ]
    )
    if nuclei_counts is not None and not nuclei_counts.empty:
        counts = nuclei_counts[["image_id", *NUCLEUS_COUNT_COLUMNS]]
        image = image.merge(counts, on="image_id", how="left")
        per_mouse = (
            image.groupby("mouse_id")[list(NUCLEUS_COUNT_COLUMNS)].sum(min_count=1).reset_index()
        )
        mouse = mouse.merge(per_mouse, on="mouse_id", how="left")
    outputs = {"image_summary": image, "mouse_summary": mouse}
    if "roi_name" in table.columns:
        roi_rows = table[table["roi_name"].fillna("").astype(str) != ""]
        if not roi_rows.empty:
            outputs["roi_summary"] = pd.DataFrame(
                [
                    {
                        "image_id": image_id,
                        "mouse_id": group["mouse_id"].iloc[0],
                        "roi_name": roi_name,
                        "roi_role": group["roi_role"].iloc[0],
                        **_group_summary(group, classes),
                    }
                    for (image_id, roi_name), group in roi_rows.groupby(
                        ["image_id", "roi_name"], sort=True
                    )
                ]
            )
    if conditions:
        mouse_conditions = _mouse_conditions(table, conditions)
        keys = sorted({key for value in mouse_conditions.values() for key in value})
        if keys:
            joined = mouse.assign(
                **{
                    key: mouse["mouse_id"].map(lambda m, k=key: mouse_conditions[m].get(k, ""))
                    for key in keys
                }
            )
            rows = []
            for group_values, group in joined.groupby(keys, sort=True):
                group_values = group_values if isinstance(group_values, tuple) else (group_values,)
                base = dict(zip(keys, group_values, strict=True))
                for name in classes:
                    for view in ("final", "predicted"):
                        values = group[f"prop_{view}_{name}"].dropna()
                        rows.append(
                            {
                                **base,
                                "fiber_class": name,
                                "view": view,
                                "n_mice": int(len(values)),
                                "mean_proportion": float(values.mean()) if len(values) else None,
                                "sd_proportion": float(values.std(ddof=1))
                                if len(values) > 1
                                else None,
                            }
                        )
            outputs["cohort_summary"] = pd.DataFrame(rows)
            if "prop_centrally_nucleated" in joined.columns:
                outputs["nuclear_cohort_summary"] = _nuclear_cohort(joined, keys, classes)
    outputs["_classes"] = pd.DataFrame({"fiber_class": classes})
    return outputs


def _mouse_conditions(
    table: pd.DataFrame, conditions: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    by_mouse: dict[str, dict[str, Any]] = {}
    for image_id, mouse_id in table.groupby("image_id")["mouse_id"].first().items():
        condition = {key: str(value) for key, value in conditions.get(image_id, {}).items()}
        previous = by_mouse.setdefault(mouse_id, condition)
        if previous != condition:
            raise ValueError(
                f"Images of mouse {mouse_id} declare different conditions; cohort grouping needs "
                "one condition per mouse."
            )
    return by_mouse


def load_conditions(project_path: Path) -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(project_path.read_text(encoding="utf-8"))
    return {
        str(image["image_id"]): dict(image.get("condition") or {})
        for image in raw.get("images", [])
    }


def load_class_labels(project_path: Path) -> dict[str, str]:
    """Panel-specific class names from the review project (models made by the builder)."""
    raw = yaml.safe_load(project_path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in (raw.get("class_labels") or {}).items()}


def write_results(final_dir: Path, output_dir: Path, project_path: Path | None) -> dict[str, Any]:
    final_table_path = final_dir / "final_fiber_table.csv"
    final = pd.read_csv(
        final_table_path, keep_default_na=False, na_values=[""], low_memory=False, dtype=ID_DTYPES
    )
    conditions = load_conditions(project_path) if project_path is not None else None
    class_labels = load_class_labels(project_path) if project_path is not None else {}
    counts_path = final_dir / "nuclei_counts.csv"
    nuclei_counts = pd.read_csv(counts_path, dtype=ID_DTYPES) if counts_path.is_file() else None
    outputs = summarize(final, conditions, nuclei_counts)
    nuclear = "centrally_nucleated" in final.columns
    classes = outputs.pop("_classes")["fiber_class"].tolist()
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, table in outputs.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = {"path": path.name, "sha256": file_sha256(path), "rows": len(table)}
    finalization_manifest = final_dir / "finalization_manifest.json"
    manifest = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "fiber_classes": classes,
        "definitions": {**DEFINITIONS, **(NUCLEAR_DEFINITIONS if nuclear else {})},
        "inputs": {
            "final_fiber_table_sha256": file_sha256(final_table_path),
            "finalization_manifest": (
                json.loads(finalization_manifest.read_text())
                if finalization_manifest.is_file()
                else None
            ),
            "conditions_from_project": project_path is not None,
        },
        "tables": written,
    }
    shown = {name: label for name, label in class_labels.items() if name in classes}
    if shown:
        manifest["class_labels"] = shown  # e.g. negative -> "I⁻/IIa⁻"; panel-specific names
    (output_dir / "results_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", type=Path, required=True, help="Finalization output dir")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--project", type=Path, default=None, help="Project YAML providing image conditions"
    )
    parser.add_argument("--no-report", action="store_true", help="Skip the HTML report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = write_results(args.final_dir, args.output_dir, args.project)
    for name, info in manifest["tables"].items():
        print(f"{name}: {info['rows']} rows -> {info['path']}")
    if not args.no_report:
        from fibertypeqc.cohort_report import write_cohort_report

        report = write_cohort_report(args.output_dir)
        print(f"report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
