"""Create private frozen-cohort artifacts for the Jag1-regeneration QUAD review.

This is intentionally a setup/audit tool: it never fits or scores a classifier.
The output directory is expected to be ignored (normally ``private/jag1_quad``).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from fibertypeqc.czi_scenes import discover_czi_scenes

PANEL = "Type I / IIa / laminin / IIb"
SEED = 20260910
NEGATIVE = ("351537_L", "351537_LL", "351544_LR", "351545_R", "351560_L")
POSITIVE = ("351537_LR", "351544_L", "351544_R", "351545_L", "351545_LL")
FORCED_DEVELOPMENT = {"351545_R", "351545_L"}
QUEUE_COLUMNS = (
    "queue_id mouse_id cre_status image_id section_id fiber_id sampling_stratum "
    "raw_image_path fiber_labels_path label timestamp provenance"
).split()


def _split() -> pd.DataFrame:
    """Fixed, stratified split independent of all image/model information."""
    selected_negative = random.Random(f"{SEED}:cre_negative").sample(
        [mouse for mouse in NEGATIVE if mouse not in FORCED_DEVELOPMENT], 2
    )
    selected_positive = random.Random(f"{SEED}:cre_positive").sample(
        [mouse for mouse in POSITIVE if mouse not in FORCED_DEVELOPMENT], 2
    )
    development = set(selected_negative + selected_positive) | FORCED_DEVELOPMENT
    rows = []
    for status, mice in (("cre_negative_mdx", NEGATIVE), ("cre_positive_mdxJAG", POSITIVE)):
        for mouse in mice:
            rows.append(
                {
                    "mouse_id": mouse,
                    "cre_status": status,
                    "jag_status": "JAG_negative"
                    if status.startswith("cre_negative")
                    else "JAG_positive",
                    "split": "development" if mouse in development else "final_test",
                    "forced_development": mouse in FORCED_DEVELOPMENT,
                    "split_seed": SEED,
                    "split_provenance": (
                        "random.Random(f'{seed}:{cre_status}') among non-forced mice; "
                        "no model performance used"
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["cre_status", "mouse_id"], kind="stable")


def _source_containers(raw_root: Path) -> list[tuple[str, str, Path]]:
    rows = []
    for path in sorted(raw_root.rglob("*.czi")):
        name = path.name.lower()
        if "change scaling" in name or "-scene-" in name:
            continue
        relative = path.relative_to(raw_root)
        if len(relative.parts) < 3:
            continue
        group, mouse = relative.parts[:2]
        if mouse not in {*NEGATIVE, *POSITIVE}:
            continue
        rows.append((group, mouse, path.resolve()))
    return rows


def _output_map(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    table = pd.read_csv(path, dtype=str).fillna("")
    if "image_id" not in table:
        raise ValueError("--processed-map must contain image_id")
    records: dict[str, dict[str, str]] = {}
    for row in table.itertuples(index=False):
        record = row._asdict()
        for key in ("fiber_labels_path", "fiber_table_path", "summary_path", "legacy_review_path"):
            value = str(record.get(key, "")).strip()
            if value:
                record[key] = str(Path(value).expanduser().resolve())
        records[str(record["image_id"])] = record
    return records


def _manifest(raw_root: Path, processed_map: dict[str, dict[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    split = _split().set_index("mouse_id")
    next_section: dict[str, int] = {}
    for _directory_group, mouse, raw_path in _source_containers(raw_root):
        scene_count = len(discover_czi_scenes(raw_path)) or 1
        for _scene_in_container in range(1, scene_count + 1):
            scene_index = next_section.get(mouse, 0) + 1
            next_section[mouse] = scene_index
            image_id = f"{mouse}_section_{scene_index:02d}"
            mapped = processed_map.get(image_id, {})
            status = "cre_negative_mdx" if mouse in NEGATIVE else "cre_positive_mdxJAG"
            rows.append(
                {
                    "image_id": image_id,
                    "mouse_id": mouse,
                    "section_id": f"section-{scene_index:02d}",
                    "scene_index": scene_index,
                    "cre_status": status,
                    "jag_status": "JAG_negative" if mouse in NEGATIVE else "JAG_positive",
                    "muscle": "QUAD",
                    "injury_status": "uninjured",
                    "study_timepoint": "28dpi",
                    "age": "approximately_18_months",
                    "background": "mdx5cv",
                    "panel": PANEL,
                    "raw_image_path": str(raw_path),
                    "fiber_labels_path": mapped.get("fiber_labels_path", ""),
                    "fiber_table_path": mapped.get("fiber_table_path", ""),
                    "summary_path": mapped.get("summary_path", ""),
                    "legacy_review_path": mapped.get("legacy_review_path", ""),
                    "batch": mapped.get("batch", "unknown"),
                    "split": split.loc[mouse, "split"],
                    "source_provenance": "frozen_jag1_regeneration_quad",
                }
            )
    result = pd.DataFrame(rows).sort_values(["mouse_id", "scene_index"], kind="stable")
    if len(result) != 27:
        raise ValueError(f"Expected 27 sections from canonical raw containers; found {len(result)}")
    return result


def _read_rows(section: pd.Series) -> pd.DataFrame:
    path = Path(str(section.fiber_table_path))
    if not path.is_file():
        return pd.DataFrame(
            columns=["fiber_id", "prediction", "confidence", "margin", "needs_review"]
        )
    table = pd.read_csv(path, low_memory=False)
    fid = "fiber_id" if "fiber_id" in table else "label"
    prediction = next(
        (x for x in ("predicted_type", "model_prediction", "fiber_type") if x in table), None
    )
    if prediction is None or fid not in table:
        return pd.DataFrame(
            columns=["fiber_id", "prediction", "confidence", "margin", "needs_review"]
        )
    out = pd.DataFrame(
        {
            "fiber_id": pd.to_numeric(table[fid], errors="coerce"),
            "prediction": table[prediction].astype(str),
        }
    )
    out["confidence"] = pd.to_numeric(
        table.get("model_confidence", table.get("confidence")), errors="coerce"
    )
    out["margin"] = pd.to_numeric(
        table.get("model_margin", table.get("probability_margin")), errors="coerce"
    )
    out["needs_review"] = table.get("needs_review", False)
    return out.dropna(subset=["fiber_id"]).query("fiber_id > 0").astype({"fiber_id": int})


def _legacy_ids(path: Path | None) -> set[int]:
    if path is None or not path.is_file():
        return set()
    table = pd.read_csv(path, low_memory=False)
    id_column = "fiber_id" if "fiber_id" in table else "label" if "label" in table else None
    if id_column is None:
        return set()
    reviewed = (
        table.get("label_source", pd.Series("", index=table.index)).astype(str).eq("manual_gold")
    )
    return set(pd.to_numeric(table.loc[reviewed, id_column], errors="coerce").dropna().astype(int))


def _queue_row(section: pd.Series, fiber_id: int, stratum: str, queue_id: str) -> dict[str, object]:
    return {
        "queue_id": queue_id,
        "mouse_id": section.mouse_id,
        "cre_status": section.cre_status,
        "image_id": section.image_id,
        "section_id": section.section_id,
        "fiber_id": fiber_id,
        "sampling_stratum": stratum,
        "raw_image_path": section.raw_image_path,
        "fiber_labels_path": section.fiber_labels_path,
        "label": "",
        "timestamp": "",
        "provenance": f"frozen_jag1_quad_queue_seed={SEED}",
    }


def _sample_mouse(
    section_rows: pd.DataFrame, target: int, seed_text: str, excluded: set[tuple[str, int]]
) -> list[tuple[pd.Series, int]]:
    choices = [
        (section, int(fid))
        for _, section in section_rows.iterrows()
        for fid in _read_rows(section).fiber_id
        if (section.image_id, int(fid)) not in excluded
    ]
    return random.Random(seed_text).sample(choices, min(target, len(choices)))


def _queues(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    development, final = [], []
    for mouse, sections in manifest.groupby("mouse_id", sort=True):
        split = sections.iloc[0].split
        available = sections[sections.fiber_table_path.astype(str).map(lambda x: Path(x).is_file())]
        if available.empty:
            continue
        if split == "development":
            legacy = {
                (section.image_id, fid)
                for _, section in available.iterrows()
                for fid in _legacy_ids(Path(section.get("legacy_review_path", "")))
            }
            random_items = _sample_mouse(
                available, 150, f"{SEED}:development:random:{mouse}", legacy
            )
            random_keys = {(x.image_id, fid) for x, fid in random_items}
            candidates = []
            for _, section in available.iterrows():
                rows = _read_rows(section)
                excluded = random_keys | legacy
                excluded_ids = {
                    fiber_id for image_id, fiber_id in excluded if image_id == section.image_id
                }
                rows = rows[~rows.fiber_id.isin(excluded_ids)]
                # Target type-I/IIa calls first, then uncertain/low-margin evidence.
                priority = (
                    rows.prediction.str.lower()
                    .isin({"i", "type1", "type_i", "iia", "type2a", "type_iia"})
                    .astype(int)
                    * 4
                )
                priority += (
                    rows.needs_review.astype(str).str.lower().isin({"true", "1"}).astype(int) * 2
                )
                priority += (
                    rows.margin.rank(method="first", ascending=True)
                    .fillna(len(rows))
                    .le(max(1, len(rows) // 4))
                    .astype(int)
                )
                for _, row in (
                    rows.assign(_priority=priority)
                    .sort_values(["_priority", "margin", "fiber_id"], ascending=[False, True, True])
                    .head(100)
                    .iterrows()
                ):
                    candidates.append((section, int(row.fiber_id)))
            development += [
                _queue_row(s, fid, "development_random", "jamie_development")
                for s, fid in random_items
            ]
            development += [
                _queue_row(s, fid, "development_targeted", "jamie_development")
                for s, fid in candidates[:100]
            ]
        else:
            random_items = _sample_mouse(available, 250, f"{SEED}:final_test:random:{mouse}", set())
            random_keys = {(x.image_id, fid) for x, fid in random_items}
            targeted = []
            for _, section in available.iterrows():
                rows = _read_rows(section)
                random_ids = {
                    fiber_id for image_id, fiber_id in random_keys if image_id == section.image_id
                }
                rows = rows[~rows.fiber_id.isin(random_ids)]
                ranked = rows.sort_values(["margin", "confidence", "fiber_id"], kind="stable").head(
                    100
                )
                targeted += [(section, int(row.fiber_id)) for _, row in ranked.iterrows()]
            final += [
                _queue_row(s, fid, "final_test_random", "blinded_final_test")
                for s, fid in random_items
            ]
            final += [
                _queue_row(s, fid, "final_test_targeted_challenge", "blinded_final_test")
                for s, fid in targeted[:100]
            ]
    return pd.DataFrame(development, columns=QUEUE_COLUMNS), pd.DataFrame(
        final, columns=QUEUE_COLUMNS
    )


def _qc(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, section in manifest.iterrows():
        table = _read_rows(section)
        flags = []
        data_status = "available"
        if table.empty:
            data_status = "missing_or_unreadable_fiber_table"
        elif len(table) < 100:
            flags.append("very_low_segmented_fiber_count")
        rows.append(
            {
                "image_id": section.image_id,
                "mouse_id": section.mouse_id,
                "section_id": section.section_id,
                "n_fibers": len(table),
                "qc_data_status": data_status,
                "inspection_flag": "|".join(flags),
                "inspection_required": bool(flags),
            }
        )
    return pd.DataFrame(rows)


def _legacy_supervision(manifest: pd.DataFrame) -> pd.DataFrame:
    """Materialize prior manual labels as development-only, never test truth."""
    records: list[dict[str, object]] = []
    for _, section in manifest[manifest.split.eq("development")].iterrows():
        path = Path(str(section.legacy_review_path))
        if not path.is_file():
            continue
        table = pd.read_csv(path, low_memory=False)
        id_column = "fiber_id" if "fiber_id" in table else "label" if "label" in table else None
        if id_column is None:
            continue
        source = table.get("label_source", pd.Series("", index=table.index)).astype(str)
        manual = table[source.eq("manual_gold")].copy()
        label_column = "corrected_type" if "corrected_type" in manual else "final_type"
        for _, row in manual.iterrows():
            records.append(
                {
                    "mouse_id": section.mouse_id,
                    "cre_status": section.cre_status,
                    "image_id": section.image_id,
                    "section_id": section.section_id,
                    "fiber_id": int(row[id_column]),
                    "label": str(row.get(label_column, "")).strip(),
                    "legacy_status": "eligible"
                    if str(row.get(label_column, "")).strip() != "exclude"
                    else "excluded",
                    "supervision_role": "legacy_development_supervision_only",
                    "source_path": str(path),
                }
            )
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, default=Path("private/jag1_quad"))
    parser.add_argument(
        "--processed-map",
        type=Path,
        help=(
            "Private CSV mapping image_id to fiber_labels_path, fiber_table_path, "
            "summary_path, and batch."
        ),
    )
    args = parser.parse_args()
    output = args.private_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(args.raw_root.resolve(), _output_map(args.processed_map))
    split = _split()
    development, final = _queues(manifest)
    qc = _qc(manifest)
    legacy = _legacy_supervision(manifest)
    manifest.to_csv(output / "canonical_section_manifest.csv", index=False)
    split.to_csv(output / "frozen_mouse_split.csv", index=False)
    development.to_csv(output / "jamie_development_queue.csv", index=False)
    final.to_csv(output / "blinded_final_test_queue.csv", index=False)
    qc.to_csv(output / "section_qc_summary.csv", index=False)
    legacy.to_csv(output / "legacy_development_supervision.csv", index=False)
    counts = (
        pd.concat([development, final])
        .groupby(["mouse_id", "cre_status", "sampling_stratum"])
        .size()
    )
    summary = {
        "seed": SEED,
        "sections": len(manifest),
        "queue_counts": [
            {
                "mouse_id": mouse_id,
                "cre_status": cre_status,
                "sampling_stratum": stratum,
                "count": int(count),
            }
            for (mouse_id, cre_status, stratum), count in counts.items()
        ],
        "qc_flags": int(qc.inspection_required.sum()),
        "legacy_development_labels": int(legacy.legacy_status.eq("eligible").sum()),
    }
    (output / "setup_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
