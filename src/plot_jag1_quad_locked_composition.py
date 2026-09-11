"""Plot mouse-level compositions from the locked Jag1 Quad candidate score."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

TYPES = ("i", "iia", "iib", "iix")
TYPE_LABELS = {"i": "Type I", "iia": "Type IIa", "iib": "Type IIb", "iix": "Type IIx"}
GROUPS = ("cre_negative_mdx", "cre_positive_mdxJAG")
GROUP_LABELS = {
    "cre_negative_mdx": "mdx\nCre-negative",
    "cre_positive_mdxJAG": "mdxJAG\nCre-positive",
}
GROUP_COLORS = {"cre_negative_mdx": "#4C78A8", "cre_positive_mdxJAG": "#E45756"}


def eligible_mouse_composition(path: Path) -> pd.DataFrame:
    """Load exactly one eligible, mouse-level row per biological mouse."""
    data = pd.read_csv(path)
    required = {"mouse_id", "cre_status", "analysis_eligible", *(f"pct_{name}" for name in TYPES)}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Mouse composition CSV missing columns: {missing}")
    data["analysis_eligible"] = data["analysis_eligible"].astype(str).str.lower().eq("true")
    data = data.loc[data["analysis_eligible"]].copy()
    if data.empty:
        raise ValueError("No analysis-eligible mice in composition CSV")
    if data.duplicated("mouse_id").any():
        raise ValueError("Expected one mouse-level row per mouse")
    if set(data["cre_status"]) - set(GROUPS):
        raise ValueError("Unexpected Cre status in composition CSV")
    return data.sort_values(["cre_status", "mouse_id"], kind="stable")


def plot_composition(data: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    for fiber_type, axis in zip(TYPES, axes.flat, strict=True):
        column = f"pct_{fiber_type}"
        for x, group in enumerate(GROUPS):
            values = data.loc[data["cre_status"].eq(group), column] * 100
            axis.scatter(
                [x] * len(values), values, color=GROUP_COLORS[group], s=52, alpha=0.9, zorder=3
            )
            if not values.empty:
                axis.hlines(values.mean(), x - 0.25, x + 0.25, color="black", linewidth=2, zorder=4)
        axis.set_title(TYPE_LABELS[fiber_type])
        axis.set_ylabel("Predicted fiber composition (%)")
        axis.set_xticks(range(len(GROUPS)), [GROUP_LABELS[group] for group in GROUPS])
        axis.set_xlim(-0.5, 1.5)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Jag1-regeneration Quad: locked-model exploratory composition", fontsize=14)
    fig.text(
        0.5,
        0.01,
        "One dot = biological mouse; black line = group mean. No held-out label validation used.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot locked Jag1 Quad mouse compositions")
    parser.add_argument("--mouse-composition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_composition(eligible_mouse_composition(args.mouse_composition), args.output)
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
