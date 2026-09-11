"""A deliberately small, queue-only blinded fiber-type review workspace.

Unlike the project dashboard, this module never opens a fiber table or model
output.  The queue supplies only identity, image/label paths, and sampling
stratum; decisions are stored under a reviewer-specific private directory.
"""

from __future__ import annotations

import argparse
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.segmentation import find_boundaries

from src.io_utils import load_multichannel_image
from src.review.storage import atomic_write_dataframe

REQUIRED = {
    "mouse_id",
    "image_id",
    "section_id",
    "fiber_id",
    "sampling_stratum",
    "raw_image_path",
    "fiber_labels_path",
}
LABELS = {"1": "i", "2": "iia", "3": "iib", "4": "iix", "u": "uncertain", "x": "exclude"}
OBSERVED_CHANNEL_NAMES = ("Type I", "Type IIa", "laminin", "Type IIb")
CHANNEL_TOGGLES = dict(zip(("q", "w", "e", "r"), OBSERVED_CHANNEL_NAMES, strict=True))


def _review_path(queue: Path, reviewer: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", reviewer.strip())
    if not safe:
        raise ValueError("--reviewer must contain at least one letter or number")
    return queue.parent / "reviews" / safe / f"{queue.stem}_decisions.csv"


def _load_queue(path: Path) -> pd.DataFrame:
    queue = pd.read_csv(path, dtype={"fiber_id": "Int64"}).fillna("")
    missing = sorted(REQUIRED - set(queue.columns))
    if missing:
        raise ValueError(f"Blinded queue is missing columns: {missing}")
    queue["fiber_id"] = pd.to_numeric(queue["fiber_id"], errors="raise").astype(int)
    if queue.duplicated(["image_id", "fiber_id"]).any():
        raise ValueError("A blinded queue may not contain a fiber more than once")
    return queue


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["reviewer", "image_id", "fiber_id", "label"])
    return pd.read_csv(path, dtype={"fiber_id": "Int64"}).fillna("")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Blinded, reviewer-isolated fiber review")
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--display-downsample", type=int, default=2)
    args = parser.parse_args(argv)
    if args.display_downsample < 1:
        raise ValueError("--display-downsample must be at least 1")
    queue = _load_queue(args.queue.resolve())
    output = _review_path(args.queue.resolve(), args.reviewer)
    existing = _load_existing(output)
    completed = set(
        zip(
            existing.image_id.astype(str),
            pd.to_numeric(existing.fiber_id, errors="coerce").fillna(-1).astype(int),
            strict=False,
        )
    )
    queue = queue[
        ~queue.apply(lambda row: (str(row.image_id), int(row.fiber_id)) in completed, axis=1)
    ].reset_index(drop=True)
    total_queue_size = len(queue) + len(completed)

    import napari
    from qtpy.QtWidgets import (
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )

    viewer = napari.Viewer(title="FiberTypeQC blinded fiber review")
    status = QLabel()
    status.setWordWrap(True)
    context = QLabel()
    context.setWordWrap(True)
    widget = QWidget()
    widget.setMinimumWidth(280)
    widget.setMaximumWidth(440)
    widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(6, 6, 6, 6)
    layout.setSpacing(4)
    layout.addWidget(context)
    layout.addWidget(
        QLabel(
            "Observed assay channels are named in the Layers list. "
            "Blinding hides candidate/model outputs only.\n"
            "Hotkeys: 1 I · 2 IIa · 3 IIb · 4 IIx · U uncertain · X exclude. "
            "Each label key autosaves and advances.\n"
            "Display: click a button below or use its key in brackets. "
            "B shows segmentation boundaries; O shows the target outline; "
            "0 restores the standard display; Z removes the last saved label."
        )
    )
    reopen_hint = QLabel("Reopen these controls: Window → Blinded review controls")
    reopen_hint.setWordWrap(True)
    layout.addWidget(reopen_hint)
    layout.addWidget(status)
    buttons = QHBoxLayout()
    for key, label in (
        ("1", "I"),
        ("2", "IIa"),
        ("3", "IIb"),
        ("4", "IIx"),
        ("u", "uncertain"),
        ("x", "exclude"),
    ):
        button = QPushButton(f"{label} ({key.upper()})")
        button.clicked.connect(lambda _=False, value=key: decide(value))
        buttons.addWidget(button)
    layout.addLayout(buttons)
    display_controls = QVBoxLayout()
    channel_controls = QHBoxLayout()
    display_buttons: dict[str, QPushButton] = {}
    for name, key in (("Type I", "Q"), ("Type IIa", "W"), ("laminin", "E"), ("Type IIb", "R")):
        short_name = {"Type I": "I", "Type IIa": "IIa", "laminin": "lam", "Type IIb": "IIb"}[name]
        button = QPushButton(f"{short_name} [{key}]")
        button.setCheckable(True)
        button.setChecked(True)
        button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        display_buttons[name] = button
        channel_controls.addWidget(button)
    display_controls.addLayout(channel_controls)
    overlay_controls = QHBoxLayout()
    for name, key, visible in (
        ("fiber boundaries", "B", False),
        ("selected fiber outline", "O", True),
    ):
        short_name = "bounds" if name == "fiber boundaries" else "outline"
        button = QPushButton(f"{short_name} [{key}]")
        button.setCheckable(True)
        button.setChecked(visible)
        button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        display_buttons[name] = button
        overlay_controls.addWidget(button)
    reset_button = QPushButton("reset [0]")
    undo_button = QPushButton("undo [Z]")
    reset_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
    undo_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
    overlay_controls.addWidget(reset_button)
    overlay_controls.addWidget(undo_button)
    display_controls.addLayout(overlay_controls)
    layout.addLayout(display_controls)
    controls_dock = viewer.window.add_dock_widget(
        widget, area="right", name="Blinded review controls"
    )
    controls_dock.setMinimumWidth(280)
    controls_dock.setMaximumWidth(440)
    position = 0
    loaded_image = ""
    labels: np.ndarray | None = None
    last_decision_index: int | None = None
    last_position: int | None = None

    def show_current() -> None:
        nonlocal loaded_image, labels
        if position >= len(queue):
            context.setText("Queue complete. All decisions are autosaved.")
            status.setText(str(output))
            return
        row = queue.iloc[position]
        # Identity and stratum are intentional provenance; candidate/model fields are absent.
        context.setText(
            f"{len(completed) + position + 1}/{total_queue_size} "
            f"({total_queue_size - len(completed) - position} remaining) · "
            f"mouse {row.mouse_id} · {row.section_id} "
            f"· fiber {int(row.fiber_id)} · {row.sampling_stratum}"
        )
        if loaded_image != str(row.image_id):
            raw = load_multichannel_image(Path(str(row.raw_image_path)))[
                :, :: args.display_downsample, :: args.display_downsample
            ]
            labels = np.asarray(tifffile.imread(str(row.fiber_labels_path)), dtype=np.int32)[
                :: args.display_downsample, :: args.display_downsample
            ]
            if raw.shape[1:] != labels.shape:
                raise ValueError(f"Raw/label shape mismatch for {row.image_id}")
            for layer in list(viewer.layers):
                viewer.layers.remove(layer)
            if raw.shape[0] != len(OBSERVED_CHANNEL_NAMES):
                raise ValueError(
                    f"Expected {len(OBSERVED_CHANNEL_NAMES)} observed channels for "
                    f"{row.image_id}; found {raw.shape[0]}"
                )
            viewer.add_image(
                raw,
                channel_axis=0,
                name=list(OBSERVED_CHANNEL_NAMES),
                colormap=["magenta", "cyan", "gray", "yellow"],
                blending="additive",
            )
            boundary_layer = viewer.add_labels(labels, name="fiber boundaries", opacity=0.20)
            outline_layer = viewer.add_labels(
                np.zeros_like(labels, dtype=np.uint8), name="selected fiber outline", opacity=0.85
            )
            # Segmentation is review context only in this workspace; do not permit edits.
            boundary_layer.editable = False
            outline_layer.editable = False
            boundary_layer.visible = False
            # Keep label layers unselected; choosing an image layer avoids edit-mode UI.
            viewer.layers.selection.active = viewer.layers[OBSERVED_CHANNEL_NAMES[0]]
            loaded_image = str(row.image_id)
        assert labels is not None
        mask = labels == int(row.fiber_id)
        if not mask.any():
            raise ValueError(f"Fiber {row.fiber_id} is absent from {row.image_id} labels")
        viewer.layers["selected fiber outline"].data = find_boundaries(mask, mode="thick").astype(
            np.uint8
        )
        y, x = np.argwhere(mask).mean(axis=0)
        viewer.camera.center = (float(y), float(x))
        viewer.camera.zoom = max(viewer.camera.zoom, 3)
        status.setText("Ready")

    def decide(key: str) -> None:
        nonlocal existing, last_decision_index, last_position, position
        if position >= len(queue):
            return
        row = queue.iloc[position]
        record = {
            "reviewer": args.reviewer,
            "mouse_id": row.mouse_id,
            "image_id": row.image_id,
            "section_id": row.section_id,
            "fiber_id": int(row.fiber_id),
            "sampling_stratum": row.sampling_stratum,
            "label": LABELS[key],
            "timestamp": datetime.now(UTC).isoformat(),
            "provenance": "blinded_queue_only; candidate/model outputs not loaded",
        }
        existing = pd.concat([existing, pd.DataFrame([record])], ignore_index=True)
        atomic_write_dataframe(output, existing)
        last_decision_index = len(existing) - 1
        last_position = position
        position += 1
        show_current()

    def toggle_layer(name: str) -> None:
        try:
            layer = viewer.layers[name]
        except KeyError:
            return
        layer.visible = not layer.visible
        display_buttons[name].setChecked(layer.visible)
        status.setText(f"{name}: {'shown' if layer.visible else 'hidden'}")

    def reset_display() -> None:
        for name in OBSERVED_CHANNEL_NAMES:
            layer = viewer.layers[name]
            layer.visible = True
            if hasattr(layer, "reset_contrast_limits"):
                layer.reset_contrast_limits()
        viewer.layers["fiber boundaries"].visible = False
        viewer.layers["selected fiber outline"].visible = True
        for name in OBSERVED_CHANNEL_NAMES:
            display_buttons[name].setChecked(True)
        display_buttons["fiber boundaries"].setChecked(False)
        display_buttons["selected fiber outline"].setChecked(True)
        viewer.layers.selection.active = viewer.layers[OBSERVED_CHANNEL_NAMES[0]]
        status.setText("Display reset: channels shown, boundaries hidden, target outline shown")

    def undo_last_decision() -> None:
        nonlocal existing, last_decision_index, last_position, position
        if last_decision_index is None or last_position is None:
            status.setText("No decision from this session is available to undo")
            return
        existing = existing.drop(index=last_decision_index).reset_index(drop=True)
        atomic_write_dataframe(output, existing)
        position = last_position
        last_decision_index = None
        last_position = None
        show_current()
        status.setText("Last decision removed; current fiber restored")

    for key in LABELS:
        viewer.bind_key(key, lambda event=None, value=key: decide(value), overwrite=True)
    for key, layer_name in CHANNEL_TOGGLES.items():
        viewer.bind_key(
            key,
            lambda event=None, value=layer_name: toggle_layer(value),
            overwrite=True,
        )
    viewer.bind_key("b", lambda event=None: toggle_layer("fiber boundaries"), overwrite=True)
    viewer.bind_key("o", lambda event=None: toggle_layer("selected fiber outline"), overwrite=True)
    viewer.bind_key("0", lambda event=None: reset_display(), overwrite=True)
    viewer.bind_key("z", lambda event=None: undo_last_decision(), overwrite=True)
    for name, button in display_buttons.items():
        button.clicked.connect(lambda _checked=False, value=name: toggle_layer(value))
    reset_button.clicked.connect(reset_display)
    undo_button.clicked.connect(undo_last_decision)
    show_current()
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
