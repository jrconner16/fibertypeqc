import json

import pytest

from fibertypeqc.html_report import generate_result_report, render_result_report
from fibertypeqc.qc_contract import build_qc_report, qc_check, write_qc_report
from fibertypeqc.result_bundle import build_result_bundle, write_result_bundle


def _write_report_fixture(tmp_path, *, image_id="image"):
    private_source = "/" + "Users/researcher/private/image.czi"
    fibers = tmp_path / "image_fibers.csv"
    fibers.write_text("label,fiber_type\n1,iib\n")
    summary = tmp_path / "image_summary.csv"
    summary.write_text("n_labels,unknown_rate\n1,0.25\n")
    preflight = tmp_path / "image_preflight_qc.json"
    write_qc_report(
        preflight,
        build_qc_report(
            stage="preflight",
            checks=[
                qc_check("preflight.ready", "pass", "Inputs are ready.", "proceed_to_processing")
            ],
        ),
    )
    postrun = tmp_path / "image_postrun_qc.json"
    write_qc_report(
        postrun,
        build_qc_report(
            stage="postrun",
            checks=[
                qc_check(
                    "postrun.fiber_count",
                    "warn",
                    "Fiber count is below the configured minimum.",
                    "inspect_segmentation",
                )
            ],
        ),
    )
    provenance = tmp_path / "image_run.json"
    provenance.write_text(
        json.dumps(
            {
                "application_version": "test",
                "git_commit": "abc123",
                "source_image": private_source,
                "panel": {"channels": {"laminin": 2, "type_iia": 1}},
            }
        )
    )
    bundle_path = tmp_path / "image_result_bundle.json"
    bundle = build_result_bundle(
        output_dir=tmp_path,
        image_id=image_id,
        retain_mode="tables",
        artifact_paths={
            "fiber_table": fibers,
            "image_summary": summary,
            "preflight_qc": preflight,
            "postrun_qc": postrun,
            "run_provenance": provenance,
        },
    )
    write_result_bundle(bundle_path, bundle)
    return bundle_path


def test_html_report_is_self_contained_actionable_and_omits_source_path(tmp_path):
    bundle_path = _write_report_fixture(tmp_path)

    output = generate_result_report(bundle_path)
    rendered = output.read_text()

    assert output.name == "image_result_report.html"
    assert "Inspect fiber segmentation" in rendered
    assert "postrun.fiber_count" in rendered
    assert 'href="image_fibers.csv"' in rendered
    assert "Segmented fibers" in rendered
    assert "/" + "Users/researcher" not in rendered
    assert "<script" not in rendered
    assert "https://" not in rendered


def test_html_report_escapes_bundle_content(tmp_path):
    bundle_path = _write_report_fixture(tmp_path, image_id="<img src=x onerror=alert(1)>")

    rendered = render_result_report(bundle_path)

    assert "&lt;img src=x onerror=alert(1)&gt;" in rendered
    assert "<img src=x" not in rendered


def test_html_report_rejects_artifact_path_escape(tmp_path):
    bundle_path = tmp_path / "image_result_bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "schema_version": "fibertypeqc.result_bundle.v1",
                "image_id": "image",
                "retain_mode": "full",
                "artifacts": {
                    "fiber_table": {
                        "path": "../private.csv",
                        "domains": ["fiber_identity"],
                    }
                },
            }
        )
    )

    with pytest.raises(ValueError, match="escapes the bundle directory"):
        render_result_report(bundle_path)


def test_html_report_directs_user_to_restore_missing_declared_artifact(tmp_path):
    bundle_path = _write_report_fixture(tmp_path)
    bundle = json.loads(bundle_path.read_text())
    bundle["artifacts"]["fiber_table"]["path"] = "missing_fibers.csv"
    bundle_path.write_text(json.dumps(bundle))

    rendered = render_result_report(bundle_path)

    assert "Restore declared artifacts" in rendered
    assert "restore_declared_artifacts" in rendered
