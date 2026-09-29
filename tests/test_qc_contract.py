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

    assert report["schema_version"] == "fibertypeqc.qc.v2"
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
            "uncertainty_rate": 0.5,
            "residual_rate": float("nan"),
            "residual_target_class": "",
            "median_area": 500.0,
            "type_corr": 0.95,
            "flag_low_labels": False,
            "flag_high_uncertainty_rate": True,
            "flag_high_residual_rate": False,
            "flag_median_area_outlier": False,
            "flag_high_type_corr": True,
        },
        min_labels=10,
        max_uncertainty_rate=0.35,
        median_area_min=200.0,
        median_area_max=15000.0,
        max_type_corr=0.92,
    )

    by_code = {check["code"]: check for check in checks}
    assert list(by_code) == [
        "postrun.fiber_count",
        "postrun.uncertainty_rate",
        "postrun.median_area",
        "postrun.marker_correlation",
    ]
    assert by_code["postrun.fiber_count"]["status"] == "pass"
    assert by_code["postrun.uncertainty_rate"]["status"] == "warn"
    assert by_code["postrun.median_area"]["status"] == "pass"
    assert by_code["postrun.marker_correlation"]["status"] == "warn"
    assert by_code["postrun.uncertainty_rate"]["metrics"]["maximum"] == 0.35


def _residual_stats(rate, flagged):
    return {
        "n_labels": 100,
        "uncertainty_rate": 0.1,
        "residual_rate": rate,
        "residual_target_class": "iix",
        "median_area": 500.0,
        "type_corr": 0.1,
        "flag_low_labels": False,
        "flag_high_uncertainty_rate": False,
        "flag_high_residual_rate": flagged,
        "flag_median_area_outlier": False,
        "flag_high_type_corr": False,
    }


def test_postrun_residual_check_is_informational_until_threshold_configured():
    checks = postrun_checks(
        _residual_stats(0.7, False),
        min_labels=10,
        max_uncertainty_rate=0.35,
        median_area_min=200.0,
        median_area_max=15000.0,
        max_type_corr=0.92,
    )

    residual = {check["code"]: check for check in checks}["postrun.residual_rate"]
    assert residual["status"] == "pass"
    assert "no threshold" in residual["message"]
    assert residual["metrics"] == {"observed": 0.7, "maximum": None, "residual_class": "iix"}


def test_postrun_residual_check_warns_when_calibrated_threshold_exceeded():
    checks = postrun_checks(
        _residual_stats(0.7, True),
        min_labels=10,
        max_uncertainty_rate=0.35,
        median_area_min=200.0,
        median_area_max=15000.0,
        max_type_corr=0.92,
        max_residual_rate=0.4,
    )

    residual = {check["code"]: check for check in checks}["postrun.residual_rate"]
    assert residual["status"] == "warn"
    assert residual["next_action"] == "inspect_marker_staining_then_review_residual_calls"


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
