"""Versioned, machine-readable QC reports with stable codes and next actions."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

QC_SCHEMA_VERSION = "fibertypeqc.qc.v1"
QC_STATUSES = frozenset(("pass", "warn", "fail"))


def qc_check(
    code: str,
    status: str,
    message: str,
    next_action: str,
    *,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in QC_STATUSES:
        raise ValueError(f"Unsupported QC status: {status}")
    check: dict[str, Any] = {
        "code": code,
        "status": status,
        "message": message,
        "next_action": next_action,
    }
    if metrics is not None:
        check["metrics"] = dict(metrics)
    return check


def build_qc_report(
    *,
    stage: str,
    checks: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    statuses = [str(check.get("status")) for check in checks]
    invalid = sorted(set(statuses) - QC_STATUSES)
    if invalid:
        raise ValueError(f"Unsupported QC statuses: {', '.join(invalid)}")
    if "fail" in statuses:
        overall_status = "fail"
    elif "warn" in statuses:
        overall_status = "warn"
    else:
        overall_status = "pass"

    actionable = next(
        (str(check["next_action"]) for check in checks if check.get("status") == "fail"),
        None,
    )
    if actionable is None:
        actionable = next(
            (str(check["next_action"]) for check in checks if check.get("status") == "warn"),
            str(checks[-1]["next_action"]) if checks else "proceed_to_next_stage",
        )
    return {
        "schema_version": QC_SCHEMA_VERSION,
        "stage": stage,
        "overall_status": overall_status,
        "recommended_next_action": actionable,
        "checks": [dict(check) for check in checks],
        "context": dict(context or {}),
    }


def write_qc_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(report),
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n"
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def postrun_checks(
    qc_stats: Mapping[str, Any],
    *,
    min_labels: int,
    max_unknown_rate: float,
    median_area_min: float,
    median_area_max: float,
    max_type_corr: float,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    checks.append(
        qc_check(
            "postrun.fiber_count",
            "warn" if bool(qc_stats["flag_low_labels"]) else "pass",
            "Fiber count is below the configured minimum."
            if bool(qc_stats["flag_low_labels"])
            else "Fiber count meets the configured minimum.",
            "inspect_segmentation"
            if bool(qc_stats["flag_low_labels"])
            else "proceed_to_next_check",
            metrics={"observed": int(qc_stats.get("n_labels", 0)), "minimum": min_labels},
        )
    )
    checks.append(
        qc_check(
            "postrun.unknown_rate",
            "warn" if bool(qc_stats["flag_high_unknown_rate"]) else "pass",
            "Unknown-call rate exceeds the configured maximum."
            if bool(qc_stats["flag_high_unknown_rate"])
            else "Unknown-call rate is within the configured maximum.",
            "confirm_channels_then_review_uncertain_fibers"
            if bool(qc_stats["flag_high_unknown_rate"])
            else "proceed_to_next_check",
            metrics={
                "observed": qc_stats.get("unknown_rate"),
                "maximum": max_unknown_rate,
            },
        )
    )
    checks.append(
        qc_check(
            "postrun.median_area",
            "warn" if bool(qc_stats["flag_median_area_outlier"]) else "pass",
            "Median fiber area is outside the configured range."
            if bool(qc_stats["flag_median_area_outlier"])
            else "Median fiber area is within the configured range.",
            "inspect_segmentation_and_pixel_scale"
            if bool(qc_stats["flag_median_area_outlier"])
            else "proceed_to_next_check",
            metrics={
                "observed": qc_stats.get("median_area"),
                "minimum": median_area_min,
                "maximum": median_area_max,
            },
        )
    )
    checks.append(
        qc_check(
            "postrun.marker_correlation",
            "warn" if bool(qc_stats["flag_high_type_corr"]) else "pass",
            "Marker-channel correlation exceeds the configured maximum."
            if bool(qc_stats["flag_high_type_corr"])
            else "Marker-channel correlation is within the configured maximum.",
            "confirm_channel_mapping_and_inspect_crosstalk"
            if bool(qc_stats["flag_high_type_corr"])
            else "proceed_to_review",
            metrics={
                "observed": qc_stats.get("type_corr"),
                "maximum": max_type_corr,
            },
        )
    )
    return checks
