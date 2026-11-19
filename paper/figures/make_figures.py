"""Build the three JOSS paper figures.

Figure 1 (workflow.png) shows the workflow with thumbnails made from the public demo's own data:
the raw stain image, the Cellpose outlines, per-fiber measurements, the model's calls, the QC
flags, and the composition before and after the example review (demo_section_a). Only the
layout, labels, and arrows are drawn here; every image and data point comes from the demo files.

Figure 2 (review_window.png) and Figure 3 (gui_panels.png) combine unedited screenshots from
docs/images/ (crops only): the review window; the project launcher, the cohort report, and the
model builder's Build results.

Run from the repository root, with the v0.5.2 demo download unpacked somewhere:
    uv run python paper/figures/make_figures.py --demo path/to/fibertypeqc_demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tifffile  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SECTION = "demo_section_a"
MOUSE = "demo_mouse_1"

# Colors of the review window and the report, so the figures read as one system.
IIA = "#22c55e"
IIB = "#d946ef"
IIX = "#f59e0b"
CALL_COLORS = {"iia": IIA, "iib": IIB, "iix": IIX}
INK = "#1f2937"
MUTED = "#6b7280"
LINE = "#cbd5e1"
CARD = "#f8fafc"
ACCENT = "#2563eb"
BUILD = "#0f766e"
IMPROVE = "#7c3aed"
FLAG = "#facc15"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})


def hex_rgb(color: str) -> np.ndarray:
    return np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)], dtype=float) / 255


# ---------------------------------------------------------------- demo data and thumbnails


def load_demo(demo: Path) -> dict:
    project = demo / "demo_project_completed"
    batch = project / "batch" / SECTION
    image = tifffile.imread(demo / "images" / f"{SECTION}.tif").astype(float)
    labels = tifffile.imread(batch / f"{SECTION}_cellpose_labels.tif")
    fibers = pd.read_csv(batch / f"{SECTION}_fibers.csv")
    summary = pd.read_csv(project / "results" / "mouse_summary.csv")
    return {
        "image": image,
        "labels": labels,
        "fibers": fibers,
        "summary": summary.set_index("mouse_id").loc[MOUSE],
    }


def choose_window(labels: np.ndarray, fibers: pd.DataFrame, size: int = 320) -> tuple[int, int]:
    """The window with the most IIa, IIx, and flagged fibers (a rule, not a choice by eye)."""
    calls = dict(zip(fibers["label"], fibers["fiber_type"], strict=True))
    flagged = set(fibers.loc[fibers["needs_review"].astype(bool), "label"])
    best, best_score = (0, 0), -1.0
    for y in range(64, labels.shape[0] - size - 64 + 1, 32):
        for x in range(64, labels.shape[1] - size - 64 + 1, 32):
            ids = set(np.unique(labels[y : y + size, x : x + size])) - {0}
            score = sum(
                2 * (calls.get(i) == "iia") + (calls.get(i) == "iix") + (i in flagged) for i in ids
            )
            if score > best_score:
                best, best_score = (y, x), score
    return best


def stain_rgb(image: np.ndarray) -> np.ndarray:
    """Channels 0 IIb (magenta), 1 IIa (green), 2 laminin (white), as in the review window."""
    norm = [np.clip(ch / np.percentile(ch, 99.7), 0, 1) for ch in image]
    iib, iia, lam = norm
    rgb = iib[..., None] * hex_rgb(IIB) + iia[..., None] * hex_rgb(IIA) + lam[..., None] * 0.55
    return np.clip(rgb, 0, 1) ** 1.2  # display gamma only


def thumbnails(data: dict) -> dict:
    labels, fibers = data["labels"], data["fibers"]
    y, x = choose_window(labels, fibers)
    win = (slice(y, y + 320), slice(x, x + 320))
    lab = labels[win]
    raw = stain_rgb(data["image"])[win]
    lam = np.clip(data["image"][2] / np.percentile(data["image"][2], 99.7), 0, 1)[win]
    edges = find_boundaries(lab, mode="inner")

    segment = np.repeat(lam[..., None] * 0.55, 3, axis=2)
    segment[edges] = hex_rgb("#38bdf8")

    calls = dict(zip(fibers["label"], fibers["fiber_type"], strict=True))
    classified = np.ones((*lab.shape, 3)) * hex_rgb("#111827")
    for fid in np.unique(lab):
        if fid and calls.get(fid) in CALL_COLORS:
            classified[lab == fid] = hex_rgb(CALL_COLORS[calls[fid]]) * 0.85
    classified[edges] = 0.08

    flagged_ids = set(fibers.loc[fibers["needs_review"].astype(bool), "label"])
    flags = classified * 0.35 + 0.02
    ring = find_boundaries(np.where(np.isin(lab, list(flagged_ids)), lab, 0), mode="thick")
    flags[ring] = hex_rgb(FLAG)
    return {"raw": raw, "segment": segment, "classified": classified, "flags": flags}


# ---------------------------------------------------------------- drawing helpers


def card(fig, x, y, w, h, *, color=CARD, edge=LINE, lw=0.8, radius=0.06):
    ax = fig.add_axes((0, 0, 1, 1), zorder=0)
    ax.set_xlim(0, fig.get_figwidth())
    ax.set_ylim(0, fig.get_figheight())
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            fc=color,
            ec=edge,
            lw=lw,
        )
    )
    return ax


def inch_axes(fig, x, y, w, h, zorder=2):
    fw, fh = fig.get_figwidth(), fig.get_figheight()
    return fig.add_axes((x / fw, y / fh, w / fw, h / fh), zorder=zorder)


def overlay(fig):
    ax = fig.add_axes((0, 0, 1, 1), zorder=5)
    ax.set_xlim(0, fig.get_figwidth())
    ax.set_ylim(0, fig.get_figheight())
    ax.axis("off")
    ax.patch.set_alpha(0)
    return ax


def arrow(ax, start, end, color=MUTED, style="-|>", lw=1.1, ls="-", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=9,
            lw=lw,
            ls=ls,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=0,
        )
    )


def badge(ax, x, y, text, color=INK):
    ax.add_patch(plt.Circle((x, y), 0.07, fc=color, ec="white", lw=0.8, zorder=6))
    ax.text(
        x, y, text, color="white", ha="center", va="center", fontsize=5.6, weight="bold", zorder=7
    )


def pill(ax, x, y, w, h, text, color, *, filled=False, size=6.6):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={h / 2}",
            fc=color if filled else "white",
            ec=color,
            lw=1.0,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color="white" if filled else INK,
        linespacing=1.1,
    )


# ---------------------------------------------------------------- Figure 1


def workflow_figure(path: Path, data: dict) -> None:
    thumbs = thumbnails(data)
    fig = plt.figure(figsize=(7.2, 4.3))
    W = 7.2
    top = 2.35  # bottom of the thumbnail row
    size = 0.86
    gap = (W - 0.1 - 7 * size) / 6.5
    xs = [0.1 + i * (size + gap) for i in range(7)]

    card(fig, 0.02, 2.22, W - 0.04, 2.04, color="#ffffff", edge=LINE)
    ov = overlay(fig)
    ov.text(0.14, 4.08, "Analysis", fontsize=8.5, weight="bold", color=INK)
    ov.text(
        0.82,
        4.08,
        "every fiber, every section; predictions are never overwritten",
        fontsize=6.6,
        color=MUTED,
    )

    titles = [
        ("Image", "multichannel\nCZI / TIFF"),
        ("Segment", "Cellpose on\nlaminin"),
        ("Measure", "per-fiber\nfluorescence"),
        ("Classify", "panel-specific\nmodel"),
        ("QC flags", "defined criteria,\nreason shown"),
        ("Review", "staged, in\nnapari"),
        ("Finalize", "tables + report,\nsource of each call"),
    ]
    for i, x in enumerate(xs):
        ax = inch_axes(fig, x, top + 0.42, size, size)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(LINE)
        if i == 0:
            ax.imshow(thumbs["raw"], interpolation="lanczos")
        elif i == 1:
            ax.imshow(thumbs["segment"], interpolation="lanczos")
        elif i == 2:
            f = data["fibers"]
            ax.scatter(
                np.log1p(f["type2_p75"]),
                np.log1p(f["type1_p75"]),
                s=2.5,
                c=[CALL_COLORS.get(c, MUTED) for c in f["fiber_type"]],
                alpha=0.55,
                lw=0,
            )
            ax.set_facecolor("#ffffff")
            ax.text(
                0.97, 0.03, "IIa →", transform=ax.transAxes, ha="right", fontsize=5, color=MUTED
            )
            ax.text(0.04, 0.96, "IIb ↑", transform=ax.transAxes, va="top", fontsize=5, color=MUTED)
        elif i == 3:
            ax.imshow(thumbs["classified"], interpolation="nearest")
        elif i == 4:
            ax.imshow(thumbs["flags"], interpolation="nearest")
        elif i == 5:
            shot = Image.open(ROOT / "docs/images/demo_review_workspace.png").convert("RGB")
            # The panel's stage tabs and the current fiber, with why it is shown (a crop).
            ax.imshow(np.asarray(shot.crop((1402, 205, 1790, 640))), interpolation="lanczos")
        else:
            s = data["summary"]
            classes = ["iib", "iia", "iix"]
            for row, (key, label) in enumerate((("predicted", "model"), ("final", "final"))):
                left = 0.0
                for cls in classes:
                    value = float(s[f"prop_{key}_{cls}"])
                    ax.barh(1 - row, value, left=left, color=CALL_COLORS[cls], height=0.62)
                    left += value
                ax.text(0.0, 1.38 - row, label, ha="left", va="bottom", fontsize=5, color=MUTED)
            ax.set_xlim(-0.04, 1.04)
            ax.set_ylim(-0.5, 2.05)
            ax.text(0.5, 1.82, MOUSE, ha="center", fontsize=4.8, color=MUTED)
            ax.set_facecolor("#ffffff")
        ov.text(
            x + size / 2 + 0.09, top + 0.27, titles[i][0], ha="center", fontsize=7.4, weight="bold"
        )
        badge(ov, x + size / 2 - 0.035 * len(titles[i][0]) - 0.04, top + 0.31, str(i + 1))
        ov.text(
            x + size / 2,
            top + 0.02,
            titles[i][1],
            ha="center",
            va="bottom",
            fontsize=5.6,
            color=MUTED,
            linespacing=1.05,
        )
    for a, b in zip(xs[:-1], xs[1:], strict=True):
        arrow(ov, (a + size + 0.02, top + 0.42 + size / 2), (b - 0.02, top + 0.42 + size / 2))
    # Legend for the call colors.
    lx = xs[3]
    for k, (name, color) in enumerate((("IIa", IIA), ("IIb", IIB), ("IIx (inferred)", IIX))):
        ov.add_patch(plt.Rectangle((lx + k * 0.42, 3.86), 0.07, 0.07, color=color))
        ov.text(lx + k * 0.42 + 0.1, 3.895, name, va="center", fontsize=5.6, color=INK)
    ov.add_patch(plt.Rectangle((xs[4] + 0.58, 3.86), 0.07, 0.07, fc="none", ec=FLAG, lw=1.4))
    ov.text(xs[4] + 0.68, 3.895, "flagged", va="center", fontsize=5.6, color=INK)

    # The two model lanes and promotion.
    card(fig, 0.02, 0.36, W - 0.04, 1.78, color="#ffffff", edge=LINE)
    ov.text(0.14, 1.93, "Models", fontsize=8.5, weight="bold", color=INK)
    ov.text(
        0.75,
        1.93,
        "criteria fixed in a versioned recipe; the project's model changes only by promotion",
        fontsize=6.6,
        color=MUTED,
    )
    h = 0.34
    build = [
        "No model for\nthe panel",
        "Test animals set\naside first",
        "Blind labels on\na random sample",
        "Candidates, by-animal\nCV on training animals",
        "Evaluate on\ntest animals",
    ]
    improve = [
        "Model does poorly\non your images",
        "Reviewed corrections\n(from step 6)",
        "Candidates vs current,\nby-animal CV on review labels",
    ]
    widths = [0.9, 1.0, 1.0, 1.25, 0.9]
    x = 0.14
    yb = 1.3
    ov.text(0.14, yb + h + 0.07, "Build a first model", fontsize=6.6, color=BUILD, weight="bold")
    ends = []
    for k, (label, w) in enumerate(zip(build, widths, strict=True)):
        pill(ov, x, yb, w, h, label, BUILD, filled=(k == 0), size=5.8)
        if k:
            arrow(ov, (ends[-1] + 0.01, yb + h / 2), (x - 0.01, yb + h / 2), color=BUILD)
        ends.append(x + w)
        x += w + 0.14
    yi = 0.54
    x = 0.14
    ov.text(0.14, yi + h + 0.07, "Improve a model", fontsize=6.6, color=IMPROVE, weight="bold")
    iends = []
    for k, (label, w) in enumerate(zip(improve, [1.0, 1.1, 1.75], strict=True)):
        pill(ov, x, yi, w, h, label, IMPROVE, filled=(k == 0), size=5.8)
        if k:
            arrow(ov, (iends[-1] + 0.01, yi + h / 2), (x - 0.01, yi + h / 2), color=IMPROVE)
        iends.append(x + w)
        x += w + 0.14
    # Promotion node, then up to the classifier.
    px, py, pw, ph = 6.05, 0.54, 1.0, 1.1
    ov.add_patch(
        FancyBboxPatch(
            (px, py),
            pw,
            ph,
            boxstyle="round,pad=0,rounding_size=0.1",
            fc=ACCENT,
            ec=ACCENT,
        )
    )
    ov.text(px + pw / 2, py + ph / 2 + 0.16, "Promote", ha="center", color="white", weight="bold")
    ov.text(
        px + pw / 2,
        py + ph / 2 - 0.12,
        "explicit, recorded,\nexact rollback;\nbecomes the model\nin step 4",
        ha="center",
        va="center",
        color="white",
        fontsize=5.6,
        linespacing=1.05,
    )
    arrow(ov, (ends[-1] + 0.02, yb + h / 2), (px - 0.02, py + ph - 0.12), color=BUILD)
    arrow(ov, (iends[-1] + 0.02, yi + h / 2), (px - 0.02, py + 0.12), color=IMPROVE)

    # Provenance strip.
    ov.add_patch(
        FancyBboxPatch(
            (0.02, 0.03),
            W - 0.04,
            0.26,
            boxstyle="round,pad=0,rounding_size=0.08",
            fc="#f1f5f9",
            ec="none",
        )
    )
    ov.text(
        W / 2,
        0.16,
        "Provenance:  panel and feature checks  ·  model SHA-256  ·  input digests and an "
        "append-only log for review decisions  ·  value_source for every finalized fiber type",
        ha="center",
        va="center",
        fontsize=6.2,
        color=INK,
    )
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- Figures 2 and 3


def labelled_canvas(panels, layout, width_px, gap=36, margin=56):
    """Place images (already scaled) on a white canvas; layout gives (x, y) for each panel."""
    height = max(y + im.height for (x, y), im in zip(layout, panels, strict=True))
    canvas = Image.new("RGB", (margin + width_px, height), "white")
    for (x, y), im in zip(layout, panels, strict=True):
        canvas.paste(im, (margin + x, y))
    return canvas


def save_canvas(canvas, path, letters):
    fig = plt.figure(figsize=(9.0, 9.0 * canvas.height / canvas.width))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(canvas)
    ax.axis("off")
    for letter, (x, y) in letters:
        ax.text(x, y, letter, fontsize=15, weight="bold", va="top", ha="left", color=INK)
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)


def review_figure(path: Path) -> None:
    shot = Image.open(ROOT / "docs/images/demo_review_workspace.png").convert("RGB")
    # Crop only: napari's layer list and controls (left) and the status bar (bottom).
    shot = shot.crop((298, 0, shot.width, shot.height - 40))
    canvas = Image.new("RGB", (shot.width, shot.height), "white")
    canvas.paste(shot, (0, 0))
    fig = plt.figure(figsize=(9.0, 9.0 * canvas.height / canvas.width))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(canvas)
    ax.axis("off")
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)


def gui_figure(path: Path) -> None:
    launcher = Image.open(ROOT / "docs/images/launcher_steps.png").convert("RGB")
    report = Image.open(ROOT / "docs/images/report_composition_light.png").convert("RGB")
    builder = Image.open(ROOT / "docs/images/builder_build_results.png").convert("RGB")
    width, gap, margin = 1800, 70, 56
    # Top row: launcher (a) and Build results (b) at one height; below: the report (c).
    ratio_l = launcher.width / launcher.height
    ratio_b = builder.width / builder.height
    row_h = round((width - gap) / (ratio_l + ratio_b))
    launcher = launcher.resize((round(row_h * ratio_l), row_h), Image.LANCZOS)
    builder = builder.resize((width - gap - launcher.width, row_h), Image.LANCZOS)
    report_w = round(width * 0.64)
    report = report.resize(
        (report_w, round(report.height * report_w / report.width)), Image.LANCZOS
    )
    report_x = (width - report_w) // 2
    layout = [(0, 0), (launcher.width + gap, 0), (report_x, row_h + gap)]
    canvas = labelled_canvas([launcher, builder, report], layout, width, margin=margin)
    letters = [
        ("a", (4, 4)),
        ("b", (margin + launcher.width + 12, 4)),
        ("c", (margin + report_x - 52, row_h + gap + 4)),
    ]
    save_canvas(canvas, path, letters)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--demo", type=Path, required=True, help="unpacked fibertypeqc_demo folder")
    args = parser.parse_args(argv)
    workflow_figure(OUT / "workflow.png", load_demo(args.demo))
    review_figure(OUT / "review_window.png")
    gui_figure(OUT / "gui_panels.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
