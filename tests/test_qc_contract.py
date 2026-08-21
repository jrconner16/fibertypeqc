from __future__ import annotations

import json

from fibertypeqc.qc_contract import (
    build_qc_report,
    postrun_checks,
    qc_check,
    write_qc_report,
)


def test_qc_report_uses_stable_status_precedence_and_next_action():
    report = build_qc_report(
        stage="preflight",
        checks=[
            qc_check(
                "preflight.arguments_valid",
                "pass",
                "Arguments are valid.",
                "proceed_to_channel_config",
            ),
            qc_check(
                "preflight.pixel_size_available",
                "warn",
                "Pixel size is unavailable.",
                "confirm_pixel_size_before_area_interpretation",
            ),
            qc_check(
                "preflight.panel_compatible",
                "fail",
                "Panel is incompatible.",
                "correct_channel_mapping",
            ),
        ],
    )

    assert report["schema_version"] == "fibertypeqc.qc.v1"
    assert report["overall_status"] == "fail"
    assert report["recommended_next_action"] == "correct_channel_mapping"


def test_passing_qc_report_uses_last_next_action():
    report = build_qc_report(
        stage="postrun",
        checks=[
            qc_check(
                "postrun.fiber_count",
                "pass",
                "Fiber count passes.",
                "proceed_to_review",
            )
        ],
    )

    assert report["overall_status"] == "pass"
    assert report["recommended_next_action"] == "proceed_to_review"


def test_postrun_checks_preserve_existing_threshold_policy():
    checks = postrun_checks(
        {
            "n_labels": 12,
            "unknown_rate": 0.5,
            "median_area": 500.0,
            "type_corr": 0.95,
            "flag_low_labels": False,
            "flag_high_unknown_rate": True,
            "flag_median_area_outlier": False,
            "flag_high_type_corr": True,
        },
        min_labels=10,
        max_unknown_rate=0.35,
        median_area_min=200.0,
        median_area_max=15000.0,
        max_type_corr=0.92,
    )

    by_code = {check["code"]: check for check in checks}
    assert by_code["postrun.fiber_count"]["status"] == "pass"
    assert by_code["postrun.unknown_rate"]["status"] == "warn"
    assert by_code["postrun.median_area"]["status"] == "pass"
    assert by_code["postrun.marker_correlation"]["status"] == "warn"
    assert by_code["postrun.unknown_rate"]["metrics"]["maximum"] == 0.35


def test_qc_report_serializes_nonfinite_measurements_as_null(tmp_path):
    path = tmp_path / "qc.json"
    report = build_qc_report(
        stage="postrun",
        checks=[
            qc_check(
                "postrun.unknown_rate",
                "warn",
                "No fibers were available.",
                "inspect_segmentation",
                metrics={"observed": float("nan")},
            )
        ],
    )

    write_qc_report(path, report)

    assert json.loads(path.read_text())["checks"][0]["metrics"]["observed"] is None
