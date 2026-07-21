"""Create image- and mouse-level eMHC summaries from reviewed policy predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _counts(frame: pd.DataFrame, prefix: str = "") -> dict[str, int]:
    labels = frame[f"{prefix}label"]
    return {
        f"{prefix}n_positive": int(labels.eq("positive").sum()),
        f"{prefix}n_negative": int(labels.eq("negative").sum()),
        f"{prefix}n_uncertain": int(labels.eq("uncertain").sum()),
    }


def _rate(n: int, denominator: int) -> float:
    return float(n / denominator) if denominator else 0.0


def _summarize_group(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, group_frame in frame.groupby(group_col, sort=True):
        row: dict[str, object] = {
            group_col: group,
            "n_images": int(group_frame["image_id"].nunique()),
            "n_labeled": int(len(group_frame)),
        }
        row.update(_counts(group_frame))
        row.update(_counts(group_frame, "policy_"))
        row["manual_positive_rate"] = _rate(row["n_positive"], row["n_labeled"])
        row["policy_positive_rate"] = _rate(row["policy_n_positive"], row["n_labeled"])
        row["policy_uncertain_rate"] = _rate(row["policy_n_uncertain"], row["n_labeled"])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(
    predictions_path: Path, policy_report_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    predictions = pd.read_csv(predictions_path)
    required = {
        "group",
        "review_path",
        "emhc_manual_label",
        "policy_prediction",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    predictions = predictions.rename(
        columns={
            "emhc_manual_label": "label",
            "policy_prediction": "policy_label",
        }
    )
    predictions["image_id"] = predictions["review_path"].map(lambda value: Path(value).parent.name)
    predictions["policy_label"] = predictions["policy_label"].fillna("").astype(str)
    predictions["label"] = predictions["label"].fillna("").astype(str)
    predictions = predictions.rename(columns={"policy_label": "policy_label"})
    predictions["policy_n_positive"] = predictions["policy_label"].eq("positive")
    predictions["policy_n_negative"] = predictions["policy_label"].eq("negative")
    predictions["policy_n_uncertain"] = predictions["policy_label"].eq("uncertain")

    def image_summary(image_frame: pd.DataFrame) -> dict[str, object]:
        row: dict[str, object] = {
            "group": image_frame["group"].iloc[0],
            "image_id": image_frame["image_id"].iloc[0],
            "n_labeled": int(len(image_frame)),
        }
        row.update(_counts(image_frame))
        row["policy_n_positive"] = int(image_frame["policy_label"].eq("positive").sum())
        row["policy_n_negative"] = int(image_frame["policy_label"].eq("negative").sum())
        row["policy_n_uncertain"] = int(image_frame["policy_label"].eq("uncertain").sum())
        row["manual_positive_rate"] = _rate(row["n_positive"], row["n_labeled"])
        row["policy_positive_rate"] = _rate(row["policy_n_positive"], row["n_labeled"])
        row["policy_uncertain_rate"] = _rate(row["policy_n_uncertain"], row["n_labeled"])
        return row

    image = pd.DataFrame(
        [image_summary(group) for _, group in predictions.groupby("image_id", sort=True)]
    )
    mouse = _summarize_group(predictions, "group")
    policy_report = json.loads(policy_report_path.read_text())
    provenance = {
        "summary_schema_version": "emhc_regeneration_summary.v1",
        "policy_report": str(policy_report_path),
        "predictions": str(predictions_path),
        "policy": policy_report,
        "source_review_files": sorted(predictions["review_path"].unique().tolist()),
        "n_images": int(image.shape[0]),
        "n_mice": int(mouse.shape[0]),
        "n_labeled": int(len(predictions)),
    }
    return image, mouse, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--policy-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    image, mouse, provenance = summarize(args.predictions, args.policy_report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / "emhc_image_summary.csv"
    mouse_path = args.output_dir / "emhc_mouse_summary.csv"
    provenance_path = args.output_dir / "emhc_policy_provenance.json"
    image.to_csv(image_path, index=False)
    mouse.to_csv(mouse_path, index=False)
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"saved image summary: {image_path}")
    print(f"saved mouse summary: {mouse_path}")
    print(f"saved policy provenance: {provenance_path}")


if __name__ == "__main__":
    main()
