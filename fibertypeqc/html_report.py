"""Self-contained HTML reporting for a versioned result bundle."""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fibertypeqc.result_bundle import RESULT_BUNDLE_SCHEMA_VERSION

STATUS_SEVERITY = {"pass": 0, "warn": 1, "fail": 2}

ACTION_GUIDANCE = {
    "proceed_to_review": (
        "Proceed to manual review",
        "Open the fiber labels and fiber table with scripts.review_labels_napari before merging "
        "reviewed labels.",
    ),
    "confirm_pixel_size_before_area_interpretation": (
        "Confirm pixel size",
        "Pixel-size metadata is unavailable. Confirm the acquisition scale before interpreting "
        "areas or lengths in physical units.",
    ),
    "inspect_segmentation": (
        "Inspect fiber segmentation",
        "Review the fiber-label mask for missed, merged, or spurious fibers before using the "
        "per-fiber results.",
    ),
    "inspect_segmentation_and_pixel_scale": (
        "Inspect segmentation and scale",
        "Check the fiber-label mask and confirm pixel-size metadata before interpreting "
        "fiber size.",
    ),
    "confirm_channels_then_review_uncertain_fibers": (
        "Confirm channels, then review uncertain fibers",
        "Verify the panel mapping and inspect fibers flagged for review before downstream "
        "analysis.",
    ),
    "confirm_channel_mapping_and_inspect_crosstalk": (
        "Inspect channel mapping and crosstalk",
        "Confirm marker-channel assignments and inspect possible bleed-through before accepting "
        "fiber-identity calls.",
    ),
    "correct_command_arguments": (
        "Correct command arguments",
        "Review incompatible command options and rerun the pipeline.",
    ),
    "correct_channel_config": (
        "Correct channel configuration",
        "Fix the panel/channel configuration and rerun the pipeline.",
    ),
    "correct_channel_mapping": (
        "Correct channel mapping",
        "Verify channel indices against the input image and rerun the pipeline.",
    ),
    "remove_or_correct_requested_domain": (
        "Correct the requested output domain",
        "Remove an unsupported requested domain or select a panel that supplies its required "
        "observations.",
    ),
    "select_compatible_panel_or_model": (
        "Select a compatible panel or model",
        "Use a model whose declared marker and feature requirements match the configured panel.",
    ),
    "select_verified_model_artifact": (
        "Select a verified model artifact",
        "Check the model path, manifest, and digest before rerunning the pipeline.",
    ),
    "select_readable_input_image": (
        "Select a readable input image",
        "Confirm the input path and supported image format before rerunning the pipeline.",
    ),
    "restore_declared_artifacts": (
        "Restore declared artifacts",
        "One or more files declared by the result bundle are missing. Restore the complete "
        "per-image output folder or rerun the pipeline before using these results.",
    ),
}

SUMMARY_METRICS = (
    ("n_labels", "Segmented fibers"),
    ("n_fibers", "Typed fibers"),
    ("unknown_rate", "Unknown rate"),
    ("area_median", "Median area (px²)"),
    ("area_um2_median", "Median area (µm²)"),
    ("type_corr", "Marker correlation"),
    ("prop_iib", "IIb proportion"),
    ("prop_iia", "IIa proportion"),
    ("prop_iix", "IIx proportion"),
    ("prop_unknown", "Unknown proportion"),
)


def _load_bundle(bundle_path: Path) -> dict[str, Any]:
    raw = json.loads(bundle_path.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != RESULT_BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"Result bundle must use schema_version {RESULT_BUNDLE_SCHEMA_VERSION}.")
    if not isinstance(raw.get("artifacts"), dict):
        raise ValueError("Result bundle artifacts must be an object.")
    return raw


def _artifact_path(bundle_path: Path, entry: dict[str, Any]) -> Path:
    raw_path = Path(str(entry.get("path", "")))
    if raw_path.is_absolute():
        raise ValueError(f"Result artifact path must be relative: {raw_path}")
    root = bundle_path.parent.resolve()
    resolved = (root / raw_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Result artifact path escapes the bundle directory: {raw_path}") from exc
    return resolved


def _read_json_artifact(bundle_path: Path, bundle: dict[str, Any], name: str) -> dict[str, Any]:
    entry = bundle["artifacts"].get(name)
    if not isinstance(entry, dict):
        return {}
    path = _artifact_path(bundle_path, entry)
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    return raw if isinstance(raw, dict) else {}


def _read_summary(bundle_path: Path, bundle: dict[str, Any]) -> dict[str, str]:
    entry = bundle["artifacts"].get("image_summary")
    if not isinstance(entry, dict):
        return {}
    path = _artifact_path(bundle_path, entry)
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        return next(csv.DictReader(handle), {})


def _report_action(qc_reports: list[dict[str, Any]]) -> tuple[str, str, str, str]:
    actionable = [
        report for report in qc_reports if str(report.get("overall_status")) in STATUS_SEVERITY
    ]
    if not actionable:
        return (
            "warn",
            "inspect_declared_artifacts",
            "Inspect declared artifacts",
            "No readable QC report was available. Inspect the declared artifacts before use.",
        )
    selected = max(actionable, key=lambda item: STATUS_SEVERITY[str(item["overall_status"])])
    status = str(selected["overall_status"])
    action = str(selected.get("recommended_next_action") or "inspect_declared_artifacts")
    title, detail = ACTION_GUIDANCE.get(
        action,
        (
            action.replace("_", " ").capitalize(),
            "Follow the recorded QC next action before downstream analysis.",
        ),
    )
    return status, action, title, detail


def _render_metric_cards(summary: dict[str, str]) -> str:
    cards = []
    for key, label in SUMMARY_METRICS:
        value = summary.get(key, "")
        if value == "":
            continue
        cards.append(
            f'<div class="metric"><span>{escape(label)}</span>'
            f"<strong>{escape(value)}</strong></div>"
        )
    return "".join(cards) or '<p class="muted">No summary metrics were available.</p>'


def _render_qc_checks(qc_reports: list[dict[str, Any]]) -> str:
    rows = []
    for report in qc_reports:
        stage = str(report.get("stage", "unknown"))
        for check in report.get("checks", []):
            if not isinstance(check, dict):
                continue
            status = str(check.get("status", "unknown"))
            rows.append(
                "<tr>"
                f"<td>{escape(stage)}</td>"
                f'<td><span class="badge {escape(status)}">{escape(status)}</span></td>'
                f"<td><code>{escape(str(check.get('code', '')))}</code></td>"
                f"<td>{escape(str(check.get('message', '')))}</td>"
                f"<td><code>{escape(str(check.get('next_action', '')))}</code></td>"
                "</tr>"
            )
    if not rows:
        return '<tr><td colspan="5" class="muted">No readable QC checks were available.</td></tr>'
    return "".join(rows)


def _render_artifacts(bundle_path: Path, bundle: dict[str, Any]) -> tuple[str, set[str], list[str]]:
    rows = []
    domains: set[str] = set()
    missing_artifacts: list[str] = []
    for name, entry in sorted(bundle["artifacts"].items()):
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path", ""))
        resolved = _artifact_path(bundle_path, entry)
        present = resolved.is_file()
        if not present:
            missing_artifacts.append(str(name))
        entry_domains = [str(value) for value in entry.get("domains", [])]
        domains.update(entry_domains)
        status = "available" if present else "missing"
        link = (
            f'<a href="{escape(quote(path, safe="/._-"), quote=True)}">{escape(path)}</a>'
            if present
            else escape(path)
        )
        rows.append(
            "<tr>"
            f"<td><code>{escape(str(name))}</code></td>"
            f"<td>{link}</td>"
            f"<td>{escape(str(entry.get('cardinality', '')))}</td>"
            f"<td>{escape(', '.join(entry_domains))}</td>"
            f'<td><span class="artifact-status {status}">{status}</span></td>'
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="5" class="muted">No artifacts were declared.</td></tr>')
    return "".join(rows), domains, missing_artifacts


def _render_provenance(provenance: dict[str, Any]) -> str:
    panel = provenance.get("panel", {})
    channels = panel.get("channels", {}) if isinstance(panel, dict) else {}
    active_channels = ", ".join(
        f"{name}={index}" for name, index in channels.items() if index is not None
    )
    items = (
        ("Application version", provenance.get("application_version")),
        ("Git commit", provenance.get("git_commit")),
        ("Output schema", provenance.get("output_schema_version")),
        ("Active channels", active_channels),
    )
    rendered = [
        f"<dt>{escape(label)}</dt><dd>{escape(str(value))}</dd>"
        for label, value in items
        if value not in (None, "")
    ]
    return "".join(rendered) or '<p class="muted">No provenance summary was available.</p>'


def render_result_report(bundle_path: Path) -> str:
    """Render a standalone HTML document from a result bundle and its declared artifacts."""
    bundle = _load_bundle(bundle_path)
    preflight = _read_json_artifact(bundle_path, bundle, "preflight_qc")
    postrun = _read_json_artifact(bundle_path, bundle, "postrun_qc")
    qc_reports = [report for report in (postrun, preflight) if report]
    status, action, action_title, action_detail = _report_action(qc_reports)
    summary = _read_summary(bundle_path, bundle)
    artifact_rows, domains, missing_artifacts = _render_artifacts(bundle_path, bundle)
    if missing_artifacts:
        status = "fail"
        action = "restore_declared_artifacts"
        action_title, action_detail = ACTION_GUIDANCE[action]
    domain_badges = (
        "".join(f'<span class="domain">{escape(domain)}</span>' for domain in sorted(domains))
        or '<span class="muted">No domains declared</span>'
    )
    provenance = _read_json_artifact(bundle_path, bundle, "run_provenance")
    image_id = escape(str(bundle.get("image_id", "")))
    retain_mode = escape(str(bundle.get("retain_mode", "")))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FiberTypeQC result report — {image_id}</title>
<style>
:root {{ color-scheme: light; --ink:#17212b; --muted:#5f6b76; --line:#d8dee4;
  --bg:#f5f7f9; --card:#fff; --pass:#217a4a; --warn:#986500; --fail:#b42318; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:15px/1.5 system-ui,-apple-system,sans-serif;
  color:var(--ink); background:var(--bg); }}
main {{ max-width:1100px; margin:0 auto; padding:32px 20px 56px; }}
h1,h2 {{ line-height:1.2; }} h1 {{ margin-bottom:4px; }} h2 {{ margin-top:32px; }}
.subtitle,.muted {{ color:var(--muted); }}
.action {{ background:var(--card); border:2px solid var(--{status}); border-radius:10px;
  padding:18px; margin:24px 0; }}
.action h2 {{ margin:0 0 6px; }} code {{ font-size:.9em; }}
.badge,.artifact-status,.domain {{ display:inline-block; border-radius:999px;
  padding:2px 9px; font-weight:650; }}
.badge.pass,.available {{ color:var(--pass); background:#e8f5ed; }}
.badge.warn {{ color:var(--warn); background:#fff3cd; }}
.badge.fail,.missing {{ color:var(--fail); background:#fdecea; }}
.domain {{ margin:0 6px 6px 0; background:#e8eef8; color:#294b7a; }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }}
.metric {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; }}
.metric span {{ display:block; color:var(--muted); font-size:.85em; }}
.metric strong {{ font-size:1.25em; }}
.table-wrap {{ overflow-x:auto; background:var(--card); border:1px solid var(--line);
  border-radius:8px; }}
table {{ border-collapse:collapse; width:100%; }}
th,td {{ padding:10px; text-align:left; border-bottom:1px solid var(--line);
  vertical-align:top; }}
th {{ background:#eef2f5; }} tr:last-child td {{ border-bottom:0; }} a {{ color:#175cd3; }}
dl {{ display:grid; grid-template-columns:max-content 1fr; gap:6px 16px; }}
dt {{ font-weight:650; }} dd {{ margin:0; }}
footer {{ margin-top:36px; color:var(--muted); font-size:.85em; }}
</style>
</head>
<body><main>
<header><h1>FiberTypeQC result report</h1>
<p class="subtitle">Image <strong>{image_id}</strong> · retain mode {retain_mode}</p></header>
<section class="action"><span class="badge {escape(status)}">{escape(status)}</span>
<h2>{escape(action_title)}</h2><p>{escape(action_detail)}</p>
<p class="muted">Recorded action: <code>{escape(action)}</code></p></section>
<section><h2>Summary</h2>
<div class="metrics">{_render_metric_cards(summary)}</div></section>
<section><h2>Available domains</h2><p>{domain_badges}</p></section>
<section><h2>QC checks</h2><div class="table-wrap"><table><thead><tr>
<th>Stage</th><th>Status</th><th>Code</th><th>Message</th><th>Next action</th>
</tr></thead><tbody>{_render_qc_checks(qc_reports)}</tbody></table></div></section>
<section><h2>Artifacts</h2><div class="table-wrap"><table><thead><tr>
<th>Name</th><th>Relative path</th><th>Cardinality</th><th>Domains</th><th>Status</th>
</tr></thead><tbody>{artifact_rows}</tbody></table></div></section>
<section><h2>Provenance</h2><dl>{_render_provenance(provenance)}</dl></section>
<footer>Generated from <code>{RESULT_BUNDLE_SCHEMA_VERSION}</code>.
This report summarizes declared outputs and QC guidance; it does not add biological calls.</footer>
</main></body></html>"""


def generate_result_report(bundle_path: Path, output_path: Path | None = None) -> Path:
    """Render and write a self-contained HTML report for one result bundle."""
    output = output_path or bundle_path.with_name(
        bundle_path.name.removesuffix("_result_bundle.json") + "_result_report.html"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_result_report(bundle_path))
    return output
