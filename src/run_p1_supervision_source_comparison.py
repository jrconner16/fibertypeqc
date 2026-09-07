"""Run the bounded P1 existing-TA/Notch3 supervision-source comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.build_existing_ta_audit_f1_features import F1_COLUMNS

CLASSES = ["iia", "iix", "iib"]
REVIEW_CONFIDENCE = 0.70
REVIEW_MARGIN = 0.25


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=20260829,
                ),
            ),
        ]
    )


def _validate(table: pd.DataFrame, *, name: str, authority: str, fitting: bool) -> pd.DataFrame:
    required = {
        "image_id",
        "group_id",
        "pipeline_label_id",
        "target_label",
        "label_authority",
        *F1_COLUMNS,
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    if table.duplicated(["image_id", "pipeline_label_id"]).any():
        raise ValueError(f"{name} has duplicate image/pipeline-label keys.")
    if not table["label_authority"].astype(str).eq(authority).all():
        raise ValueError(f"{name} does not have uniform {authority} label authority.")
    labels = table["target_label"].astype(str).str.lower()
    if set(labels).difference(CLASSES):
        raise ValueError(f"{name} contains labels outside the locked three-class target.")
    if fitting and "eligible_for_model_fitting" in table:
        if not table["eligible_for_model_fitting"].astype(str).str.lower().eq("true").all():
            raise ValueError(f"{name} contains rows ineligible for fitting.")
    values = table.loc[:, F1_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{name} has missing or non-finite F1 values.")
    out = table.copy()
    out["target_label"] = labels
    return out.reset_index(drop=True)


def _predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    if set(train["target_label"]) != set(CLASSES):
        raise ValueError("A training split does not contain all target classes.")
    model = _model()
    model.fit(train.loc[:, F1_COLUMNS], train["target_label"])
    raw = model.predict_proba(test.loc[:, F1_COLUMNS])
    order = [list(model.named_steps["model"].classes_).index(name) for name in CLASSES]
    probabilities = raw[:, order]
    predicted = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    result = test.loc[:, ["image_id", "group_id", "pipeline_label_id", "target_label"]].copy()
    result["predicted_label"] = predicted
    for index, name in enumerate(CLASSES):
        result[f"prob_{name}"] = probabilities[:, index]
    result["model_confidence"] = probabilities.max(axis=1)
    sorted_probabilities = np.sort(probabilities, axis=1)
    result["model_margin"] = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    return result


def _metrics(predictions: pd.DataFrame, condition: str, scope: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_id, group in predictions.groupby("group_id", sort=True):
        actual = group["target_label"]
        predicted = group["predicted_label"]
        rows.append(
            {
                "condition": condition,
                "evaluation_scope": scope,
                "scope_type": "group",
                "scope_id": group_id,
                "rows": len(group),
                "macro_f1": f1_score(
                    actual, predicted, labels=CLASSES, average="macro", zero_division=0
                ),
                "balanced_accuracy": balanced_accuracy_score(actual, predicted),
                "review_rate": (
                    (group["model_confidence"] < REVIEW_CONFIDENCE)
                    | (group["model_margin"] < REVIEW_MARGIN)
                ).mean(),
            }
        )
    actual = predictions["target_label"]
    predicted = predictions["predicted_label"]
    rows.append(
        {
            "condition": condition,
            "evaluation_scope": scope,
            "scope_type": "overall",
            "scope_id": "all",
            "rows": len(predictions),
            "macro_f1": f1_score(
                actual, predicted, labels=CLASSES, average="macro", zero_division=0
            ),
            "balanced_accuracy": balanced_accuracy_score(actual, predicted),
            "review_rate": (
                (predictions["model_confidence"] < REVIEW_CONFIDENCE)
                | (predictions["model_margin"] < REVIEW_MARGIN)
            ).mean(),
        }
    )
    return pd.DataFrame(rows)


def _leave_one_group_out(
    train_source: pd.DataFrame,
    *,
    condition: str,
    scope: str,
    pooled: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for group_id in sorted(train_source["group_id"].astype(str).unique()):
        test = train_source.loc[train_source["group_id"].astype(str).eq(group_id)]
        train = train_source.loc[~train_source["group_id"].astype(str).eq(group_id)]
        if pooled is not None:
            train = pd.concat([train, pooled], ignore_index=True)
        frame = _predict(train, test)
        frame.insert(0, "condition", condition)
        frame.insert(1, "evaluation_scope", scope)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run_comparison(
    existing: pd.DataFrame, notch: pd.DataFrame, protected: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    existing_oof = _leave_one_group_out(
        existing, condition="existing_only", scope="grouped_resampling_existing"
    )
    notch_oof = _leave_one_group_out(
        notch, condition="notch3_only", scope="grouped_resampling_notch3"
    )
    pooled_existing = _leave_one_group_out(
        existing,
        condition="pooled",
        scope="grouped_resampling_existing",
        pooled=notch,
    )
    pooled_notch = _leave_one_group_out(
        notch,
        condition="pooled",
        scope="grouped_resampling_notch3",
        pooled=existing,
    )
    for frame in [existing_oof, notch_oof, pooled_existing, pooled_notch]:
        prediction_frames.append(frame)
    for condition, train, test, scope in [
        ("existing_to_notch3", existing, notch, "cross_cohort_notch3_development"),
        ("notch3_to_existing", notch, existing, "cross_cohort_existing_development"),
        ("existing_only", existing, protected, "protected_reporting_only"),
        ("notch3_only", notch, protected, "protected_reporting_only"),
        (
            "pooled",
            pd.concat([existing, notch], ignore_index=True),
            protected,
            "protected_reporting_only",
        ),
    ]:
        frame = _predict(train, test)
        frame.insert(0, "condition", condition)
        frame.insert(1, "evaluation_scope", scope)
        prediction_frames.append(frame)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.concat(
        [
            _metrics(group, str(condition), str(scope))
            for (condition, scope), group in predictions.groupby(
                ["condition", "evaluation_scope"], sort=True
            )
        ],
        ignore_index=True,
    )
    return predictions, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-features", type=Path, required=True)
    parser.add_argument("--notch3-development-features", type=Path, required=True)
    parser.add_argument("--notch3-protected-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    existing = _validate(
        pd.read_csv(args.existing_features, low_memory=False),
        name="existing",
        authority="manual_audit",
        fitting=True,
    )
    notch = _validate(
        pd.read_csv(args.notch3_development_features, low_memory=False),
        name="Notch3 development",
        authority="reviewed_myosight",
        fitting=True,
    )
    protected = _validate(
        pd.read_csv(args.notch3_protected_features, low_memory=False),
        name="Notch3 protected",
        authority="reviewed_myosight",
        fitting=False,
    )
    predictions, metrics = run_comparison(existing, notch, protected)
    args.output_dir.mkdir(parents=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    overall_metrics = metrics.loc[metrics["scope_type"].eq("overall")]
    overall_metrics.to_csv(args.output_dir / "overall_metrics.csv", index=False)
    provenance = {
        "existing_features_sha256": _sha256(args.existing_features),
        "notch3_development_features_sha256": _sha256(args.notch3_development_features),
        "notch3_protected_features_sha256": _sha256(args.notch3_protected_features),
        "feature_columns": F1_COLUMNS,
        "model": (
            "StandardScaler + LogisticRegression(C=1.0, lbfgs, class_weight=balanced, "
            "max_iter=5000, random_state=20260829)"
        ),
        "protected_use": "reporting_only; not candidate, threshold, or model selection",
    }
    provenance_path = args.output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"prediction rows: {len(predictions)}")
    print(f"overall metric rows: {(metrics['scope_type'] == 'overall').sum()}")


if __name__ == "__main__":
    main()
