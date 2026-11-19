"""Build the two JOSS paper figures.

Figure 1 is a schematic of the workflow, drawn here. Figure 2 combines two existing, unedited
screenshots from the public demo (docs/images/): the review window and the top of the cohort report.

Run from the repository root:
    uv run python paper/figures/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

MAIN = "#dbe7f3"
REVIEW = "#fbe3c8"
MODEL = "#e3efd9"
EDGE = "#3a3a3a"


def box(ax, x, y, w, h, text, color, size=8.5, bold=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            fc=color,
            ec=EDGE,
            lw=0.9,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        fontweight="bold" if bold else "normal",
        linespacing=1.15,
    )


def arrow(ax, start, end, style="-|>", ls="-", color=EDGE):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=10,
            lw=1.0,
            ls=ls,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def workflow_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.set_xlim(0, 18.1)
    ax.set_ylim(0, 8.4)
    ax.axis("off")

    w, h, step = 2.1, 1.3, 2.25
    x = [0.1 + i * step for i in range(8)]  # one column per main step
    row1, row2, row3 = 6.0, 3.7, 1.3

    # Main path, left to right.
    main = [
        ("Multichannel\nimages\n(CZI, TIFF)", MAIN),
        ("Segmentation\n(Cellpose,\nlaminin)", MAIN),
        ("Per-fiber\nfluorescence\nfeatures", MAIN),
        ("Panel-specific\nclassifier", MAIN),
        ("QC flags\nwith reasons", REVIEW),
        ("Staged review\n(napari)", REVIEW),
        ("Finalize\n(predictions\nkept)", REVIEW),
        ("Tables +\nHTML report", MAIN),
    ]
    for xi, (label, color) in zip(x, main, strict=True):
        box(ax, xi, row1, w, h, label, color, size=7.5)
    for x0, x1 in zip(x[:-1], x[1:], strict=True):
        arrow(ax, (x0 + w, row1 + h / 2), (x1, row1 + h / 2))
    ax.text(
        x[0],
        row1 + h + 0.45,
        "Analysis with a released or promoted model",
        fontsize=8.5,
        fontweight="bold",
    )
    ax.text(
        x[5] + w / 2,
        row1 + h + 0.15,
        "section verdicts · flagged fibers · clicked fibers",
        ha="center",
        fontsize=6.5,
        style="italic",
    )

    # Explicit promotion, directly below the classifier it replaces.
    bh = 1.3
    box(ax, x[3], row2, w, bh, "Explicit\npromotion\n(recorded,\nundoable)", MODEL, size=7)
    arrow(ax, (x[3] + w / 2, row2 + bh), (x[3] + w / 2, row1), ls="--")

    # Build a first model: left to right, ending below the promotion box.
    build = [
        "Segment and\nmeasure only\n(model: none)",
        "Blind labels\n(test mice set\naside first)",
        "Build\ncandidates\n(by-mouse CV,\nfixed recipe)",
        "Evaluate on\nheld-out test\nmice (blind\nlabels)",
    ]
    for i, label in enumerate(build):
        box(ax, x[i], row3, w, bh, label, MODEL, size=7)
    for i in range(3):
        arrow(ax, (x[i] + w, row3 + bh / 2), (x[i + 1], row3 + bh / 2))
    arrow(ax, (x[3] + w / 2, row3 + bh), (x[3] + w / 2, row2))
    ax.text(
        x[0], row3 + bh + 0.25, "No model for the panel: build one", fontsize=8, fontweight="bold"
    )

    # Improve a model: corrections from review, right to left, into the promotion box.
    improve = {
        6: "Reviewed\ncorrections",
        5: "Train\ncandidates\n(fixed recipe)",
        4: "Compare with\ncurrent model\n(by-mouse CV,\nreview labels)",
    }
    for i, label in improve.items():
        box(ax, x[i], row3, w, bh, label, MODEL, size=7)
    for i in (6, 5):
        arrow(ax, (x[i], row3 + bh / 2), (x[i - 1] + w, row3 + bh / 2))
    arrow(ax, (x[4] + 0.3, row3 + bh), (x[3] + w, row2 + bh / 2))
    arrow(ax, (x[5] + w - 0.3, row1), (x[6] + w - 0.3, row3 + bh), ls=":", color="#7a7a7a")
    ax.text(
        x[4] + 0.5,
        row3 + bh + 0.25,
        "Improve from review corrections",
        fontsize=8,
        fontweight="bold",
    )

    # Provenance band.
    ax.add_patch(
        FancyBboxPatch(
            (0.1, 0.05),
            17.85,
            0.8,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            fc="#f1f1f1",
            ec="#9a9a9a",
            lw=0.8,
        )
    )
    ax.text(
        9.0,
        0.45,
        "Provenance throughout: panel and feature checks · model SHA-256 · versioned recipes "
        "and model history\ninput digests and an append-only log for review decisions · the "
        "source of every finalized fiber type (value_source)",
        ha="center",
        va="center",
        fontsize=6.8,
        linespacing=1.3,
    )
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def review_and_report_figure(path: Path) -> None:
    review = Image.open(ROOT / "docs/images/demo_review_workspace.png").convert("RGB")
    report = Image.open(ROOT / "docs/images/report_composition_light.png").convert("RGB")
    # Crops only, no edits: drop napari's layer list and layer controls (the left 298 px) so the
    # image and the review panel are larger in print; the report image is used whole
    # (provenance header and every mouse's composition before and after review).
    review = review.crop((298, 0, review.width, review.height))

    # Stack the two images on one white canvas: the review window full width, the report scaled
    # to 80% of that width and centred, with a left margin for the panel letters.
    margin, gap = 70, 40
    scale = 0.80 * review.width / report.width
    report = report.resize((round(report.width * scale), round(report.height * scale)))
    width = margin + review.width
    height = review.height + gap + report.height
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(review, (margin, 0))
    report_x = margin + (review.width - report.width) // 2
    canvas.paste(report, (report_x, review.height + gap))

    fig = plt.figure(figsize=(9.0, 9.0 * height / width))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(canvas)
    ax.axis("off")
    for label, x, y in (("a", 8, 10), ("b", report_x - 50, review.height + gap + 10)):
        ax.text(x, y, label, fontsize=14, fontweight="bold", va="top", ha="left")
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    workflow_figure(OUT / "workflow.png")
    review_and_report_figure(OUT / "review_and_report.png")
