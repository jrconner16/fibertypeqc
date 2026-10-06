"""Self-contained cohort HTML report built only from exported result tables.

Every number and chart is read from the CSVs and ``results_manifest.json`` written by
``src.summarize_results``; the report holds no state of its own, so it can be regenerated and
audited from those files.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

# Reference categorical slots (light, dark), validated in fixed order for adjacent stacked fills.
CLASS_COLORS = {
    "i": ("#2a78d6", "#3987e5"),
    "iia": ("#eb6834", "#d95926"),
    "iix": ("#1baf7a", "#199e70"),
    "iib": ("#eda100", "#c98500"),
    "hybrid": ("#e87ba4", "#d55181"),
}
OTHER_COLORS = ("#8a8984", "#a3a29c")
CLASS_LABELS = {"i": "Type I", "iia": "IIa", "iix": "IIx", "iib": "IIb", "hybrid": "Hybrid"}
LABEL_MIN_SHARE = 0.08


def _label(name: str) -> str:
    return CLASS_LABELS.get(name, name)


def _pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "–"
    return f"{100 * float(value):.1f}%"


def _css(classes: list[str]) -> str:
    def slots(index: int) -> str:
        lines = []
        for position, name in enumerate(classes):
            colors = CLASS_COLORS.get(name, OTHER_COLORS)
            lines.append(f"--c-{position}: {colors[index]};")
        return " ".join(lines)

    light = (
        "--surface: #fcfcfb; --surface-2: #f2f1ee; --ink: #0b0b0b; --ink-2: #52514e; "
        "--rule: #dcdbd6; " + slots(0)
    )
    dark = (
        "--surface: #1a1a19; --surface-2: #262624; --ink: #ffffff; --ink-2: #c3c2b7; "
        "--rule: #3a3936; " + slots(1)
    )
    return f"""
:root {{ color-scheme: light; {light} }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{ color-scheme: dark; {dark} }}
}}
:root[data-theme="dark"] {{ color-scheme: dark; {dark} }}
body {{ background: var(--surface); color: var(--ink); margin: 0;
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px 48px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
h2 {{ font-size: 17px; margin: 32px 0 8px; }}
p.meta, .note {{ color: var(--ink-2); }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; margin: 8px 0; }}
th, td {{ border-bottom: 1px solid var(--rule); padding: 4px 10px; text-align: right;
  white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--ink-2); font-weight: 600; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin: 6px 0; color: var(--ink-2); }}
.legend span::before {{ content: ""; display: inline-block; width: 10px; height: 10px;
  border-radius: 2px; margin-right: 6px; background: var(--swatch); vertical-align: -1px; }}
svg text {{ fill: var(--ink); font-size: 11px; }}
svg text.muted {{ fill: var(--ink-2); }}
svg rect.seg:hover {{ opacity: 0.85; }}
dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 2px 14px; }}
dt {{ color: var(--ink-2); }}
"""


def _composition_chart(mouse: pd.DataFrame, classes: list[str]) -> str:
    """Paired 100% bars per mouse: predicted (model) above finalized (after review)."""
    label_w, bar_w, bar_h, gap, group_gap = 150, 640, 16, 4, 14
    rows = []
    y = 4
    for _, row in mouse.iterrows():
        for view, title in (("predicted", "predicted"), ("final", "finalized")):
            x = label_w
            shares = [row.get(f"prop_{view}_{name}") for name in classes]
            text = f"{html.escape(str(row['mouse_id']))} · {title}"
            rows.append(
                f'<text x="{label_w - 8}" y="{y + bar_h - 4}" text-anchor="end" '
                f'class="{"" if view == "final" else "muted"}">{text}</text>'
            )
            for position, (name, share) in enumerate(zip(classes, shares, strict=True)):
                if share is None or pd.isna(share) or share <= 0:
                    continue
                width = bar_w * float(share)
                drawn = max(width - 2, 0.5)  # 2px surface gap between segments
                tip = f"{row['mouse_id']} {title}: {_label(name)} {_pct(share)}"
                rows.append(
                    f'<rect class="seg" x="{x:.1f}" y="{y}" width="{drawn:.1f}" '
                    f'height="{bar_h}" rx="2" fill="var(--c-{position})">'
                    f"<title>{html.escape(tip)}</title></rect>"
                )
                if share >= LABEL_MIN_SHARE:
                    rows.append(f'<text x="{x + 4:.1f}" y="{y + bar_h - 4}">{_pct(share)}</text>')
                x += width
            y += bar_h + gap
        y += group_gap
    height = y
    return (
        f'<svg role="img" aria-label="Fiber-type composition per mouse, predicted and finalized" '
        f'viewBox="0 0 {label_w + bar_w + 10} {height}" width="100%" '
        f'style="max-width:{label_w + bar_w + 10}px">' + "".join(rows) + "</svg>"
    )


def _legend(classes: list[str]) -> str:
    items = "".join(
        f'<span style="--swatch: var(--c-{position})">{_label(name)}</span>'
        for position, name in enumerate(classes)
    )
    return f'<div class="legend">{items}</div>'


def _table(frame: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    head = "".join(f"<th>{html.escape(title)}</th>" for _, title, _ in columns)
    body = []
    for _, row in frame.iterrows():
        cells = []
        for column, _, kind in columns:
            value = row.get(column)
            if kind == "pct":
                text = _pct(value)
            elif kind == "int":
                text = "–" if value is None or pd.isna(value) else f"{int(value)}"
            else:
                text = html.escape(str(value))
            cells.append(f"<td>{text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _cohort_table(cohort: pd.DataFrame, classes: list[str]) -> str:
    group_keys = [
        column
        for column in cohort.columns
        if column not in {"fiber_class", "view", "n_mice", "mean_proportion", "sd_proportion"}
    ]
    rows = []
    for group_values, group in cohort.groupby(group_keys, sort=True):
        group_values = group_values if isinstance(group_values, tuple) else (group_values,)
        record: dict[str, Any] = dict(zip(group_keys, group_values, strict=True))
        final = group[group["view"] == "final"].set_index("fiber_class")
        record["n_mice"] = int(final["n_mice"].max()) if not final.empty else 0
        for name in classes:
            if name in final.index:
                mean, sd = final.loc[name, "mean_proportion"], final.loc[name, "sd_proportion"]
                record[name] = f"{_pct(mean)} ± {_pct(sd)}" if not pd.isna(sd) else _pct(mean)
            else:
                record[name] = "–"
        rows.append(record)
    frame = pd.DataFrame(rows)
    columns = [(key, key, "text") for key in group_keys] + [("n_mice", "Mice", "int")]
    columns += [(name, f"{_label(name)} mean ± SD", "text") for name in classes]
    return _table(frame, columns)


def build_cohort_report(results_dir: Path) -> str:
    manifest = json.loads((results_dir / "results_manifest.json").read_text())
    classes: list[str] = manifest["fiber_classes"]
    mouse = pd.read_csv(results_dir / "mouse_summary.csv")
    image = pd.read_csv(results_dir / "image_summary.csv")
    finalization = manifest["inputs"].get("finalization_manifest") or {}
    meta = {
        "Project": finalization.get("project_id", "–"),
        "Model": finalization.get("model_version", "–"),
        "QC version": finalization.get("qc_version") or "–",
        "Code revision": (manifest.get("git_commit") or "–")[:12],
        "Final fiber table SHA-256": manifest["inputs"]["final_fiber_table_sha256"][:16] + "…",
    }
    unverified = [
        item["image_id"]
        for item in finalization.get("images", [])
        if item.get("verification") == "unverified"
    ]
    not_reviewed = sum(
        item.get("verification") == "not_reviewed" for item in finalization.get("images", [])
    )
    sections = [
        "<h1>FiberTypeQC cohort results</h1>",
        f'<p class="meta">Generated from exported tables in this folder · '
        f"{html.escape(manifest['created_at_utc'][:19])} UTC</p>",
        "<dl>"
        + "".join(f"<dt>{k}</dt><dd>{html.escape(str(v))}</dd>" for k, v in meta.items())
        + "</dl>",
    ]
    if not_reviewed:
        sections.append(
            f'<p class="note">{not_reviewed} section(s) were not reviewed; their values are '
            "model predictions.</p>"
        )
    if unverified:
        sections.append(
            f'<p class="note">Review inputs were not fingerprint-verified for '
            f"{len(unverified)} image(s): {html.escape(', '.join(unverified))}.</p>"
        )
    sections += [
        "<h2>Composition before and after review</h2>",
        '<p class="note">Predicted: model calls over all analysis fibers. Finalized: final '
        "types over resolved fibers; unresolved fibers are excluded from the denominator and "
        "counted below. Fibers are pooled across each mouse's sections.</p>",
        _legend(classes),
        _composition_chart(mouse, classes),
        "<h2>Mice</h2>",
        _table(
            mouse,
            [
                ("mouse_id", "Mouse", "text"),
                ("n_images", "Images", "int"),
                ("n_analysis", "Analysis fibers", "int"),
                ("n_resolved", "Resolved", "int"),
                ("n_unresolved", "Unresolved", "int"),
                ("n_excluded", "Excluded", "int"),
            ]
            + [(f"prop_final_{n}", f"{_label(n)} final", "pct") for n in classes]
            + [(f"prop_predicted_{n}", f"{_label(n)} predicted", "pct") for n in classes],
        ),
    ]
    cohort_path = results_dir / "cohort_summary.csv"
    if cohort_path.is_file():
        cohort = pd.read_csv(cohort_path, keep_default_na=False, na_values=[""])
        sections += [
            "<h2>Cohort (mouse as the unit)</h2>",
            '<p class="note">Mean ± SD across mice of mouse-level finalized proportions.</p>',
            _cohort_table(cohort, classes),
        ]
    roi_path = results_dir / "roi_summary.csv"
    if roi_path.is_file():
        roi = pd.read_csv(roi_path)
        sections += [
            "<h2>Analysis ROIs</h2>",
            _table(
                roi,
                [
                    ("image_id", "Image", "text"),
                    ("roi_name", "ROI", "text"),
                    ("n_resolved", "Resolved", "int"),
                ]
                + [(f"prop_final_{n}", f"{_label(n)} final", "pct") for n in classes],
            ),
        ]
    sections += [
        "<h2>Review and QC burden per image</h2>",
        _table(
            image,
            [
                ("image_id", "Image", "text"),
                ("mouse_id", "Mouse", "text"),
                ("n_fibers_total", "Fibers", "int"),
                ("n_reviewed", "Reviewed", "int"),
                ("n_corrected", "Corrected", "int"),
                ("n_flagged_unreviewed", "Flagged, unreviewed", "int"),
                ("n_unresolved", "Unresolved", "int"),
                ("n_excluded_image", "Excl. image", "int"),
                ("n_excluded_section", "Excl. section", "int"),
                ("n_excluded_region", "Excl. region", "int"),
                ("n_excluded_outside_analysis_roi", "Excl. outside ROI", "int"),
                ("n_excluded_fiber_review", "Excl. by reviewer", "int"),
                ("n_excluded_edge_of_image", "Excl. image edge", "int"),
            ],
        ),
        "<h2>Source tables</h2>",
        _table(
            pd.DataFrame(
                [
                    {"table": info["path"], "rows": info["rows"], "sha": info["sha256"][:16]}
                    for info in manifest["tables"].values()
                ]
            ),
            [("table", "Table", "text"), ("rows", "Rows", "int"), ("sha", "SHA-256", "text")],
        ),
    ]
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>FiberTypeQC cohort results</title><style>{_css(classes)}</style></head>"
        f"<body><main>{''.join(sections)}</main></body></html>\n"
    )


def write_cohort_report(results_dir: Path) -> Path:
    path = results_dir / "cohort_report.html"
    path.write_text(build_cohort_report(results_dir), encoding="utf-8")
    return path
