"""Build protected Notch3 ROI supervision rows from fixed-mask matching outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _canonical_label(value: object) -> str:
    normalized = str(value).strip().lower().replace(" ", "")
    aliases = {"iia": "iia", "iib": "iib", "iix": "iix", "i": "i", "i/iia": "i/iia"}
    return aliases.get(normalized, normalized)


def build_all_rois(manifest: pd.DataFrame, matching_root: Path) -> pd.DataFrame:
    required = {"image_id", "group_id", "myosight_results_dir"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    frames: list[pd.DataFrame] = []
    for _, row in manifest.sort_values("image_id").iterrows():
        image_id = str(row["image_id"])
        candidate_path = matching_root / image_id / f"{image_id}_match_candidates.csv"
        if not candidate_path.is_file():
            raise FileNotFoundError(f"Missing match candidates for {image_id}: {candidate_path}")
        candidates = pd.read_csv(candidate_path)
        results_path = Path(str(row["myosight_results_dir"])) / "Results.txt"
        results = pd.read_csv(results_path, sep="\t")
        results = results.loc[:, [column for column in results.columns if str(column).strip()]]
        if "Label" not in results:
            raise ValueError(f"Results table missing Label column: {results_path}")
        if len(candidates) != len(results):
            raise ValueError(
                f"ROI/result row count differs for {image_id}: {len(candidates)} != {len(results)}"
            )
        candidates = candidates.copy()
        candidates["roi_index"] = pd.to_numeric(candidates["roi_index"], errors="raise").astype(int)
        if set(candidates["roi_index"]) != set(range(len(results))):
            raise ValueError(
                f"Candidate ROI indices are not zero-based and complete for {image_id}."
            )
        candidates["myosight_label"] = (
            results.iloc[candidates["roi_index"]]["Label"].map(_canonical_label).to_numpy()
        )
        label_counts = candidates.loc[
            candidates["candidate_status"].eq("matched"), "label_id"
        ].value_counts()
        matched_label_counts = candidates["label_id"].map(label_counts).fillna(0)
        candidates["one_to_one_eligible"] = (
            candidates["candidate_status"].eq("matched")
            & pd.to_numeric(candidates["label_id"], errors="raise").gt(0)
            & matched_label_counts.eq(1)
        )
        candidates["match_outcome"] = candidates["candidate_status"].where(
            ~candidates["one_to_one_eligible"], "matched_one_to_one"
        )
        candidates.insert(0, "image_id", image_id)
        candidates.insert(1, "group_id", str(row["group_id"]))
        frames.append(candidates)
    output = pd.concat(frames, ignore_index=True)
    if output.duplicated(["image_id", "roi_index"]).any():
        raise ValueError("Built all-ROI table has duplicate image/ROI keys.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--matching-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=22)
    args = parser.parse_args()
    if args.output.exists() or args.provenance_output.exists():
        raise FileExistsError("Refusing to overwrite an existing output or provenance file.")
    output = build_all_rois(pd.read_csv(args.manifest), args.matching_root)
    if output["image_id"].nunique() != args.expected_images:
        raise ValueError(
            f"Expected {args.expected_images} images; found {output['image_id'].nunique()}."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    provenance = {
        "images": int(output["image_id"].nunique()),
        "rows": len(output),
        "one_to_one_eligible": int(output["one_to_one_eligible"].sum()),
        "label_counts": output["myosight_label"].value_counts().sort_index().to_dict(),
    }
    args.provenance_output.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"protected images: {output['image_id'].nunique()}")
    print(f"one-to-one eligible rows: {int(output['one_to_one_eligible'].sum())}")


if __name__ == "__main__":
    main()
