"""Blinded, image-first development review for visually selected Jag1 Quad fibers.

The reviewer chooses a fiber directly in the image before classifying it.  This
mode deliberately has no candidate list, fiber table, prediction, probability,
or model-margin input.
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

from src.blinded_fiber_review_napari import (
    CHANNEL_TOGGLES,
    LABELS,
    OBSERVED_CHANNEL_NAMES,
)
from src.io_utils import load_multichannel_image
from src.review.storage import atomic_write_dataframe

REQUIRED = {
    "mouse_id",
    "image_id",
    "section_id",
    "split",
    "raw_image_path",
    "fiber_labels_path",
}
DEFAULT_REVIEW_NAME = "development_visual_hunt"
DEFAULT_STRATUM = "development_visual_hunt"


def _safe_name(value: str, argument: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not safe:
        raise ValueError(f"{argument} must contain at least one letter or number")
    return safe


def _review_path(manifest: Path, reviewer: str, review_name: str = DEFAULT_REVIEW_NAME) -> Path:
    return (
        manifest.parent
        / "reviews"
        / _safe_name(reviewer, "--reviewer")
        / f"{_safe_name(review_name, '--review-name')}_decisions.csv"
    )


def _load_development_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype=str).fillna("")
    missing = sorted(REQUIRED - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    development = manifest[manifest.split.eq("development")].copy()
    if development.empty:
        raise ValueError("Manifest contains no development sections")
    return development.sort_values(["mouse_id", "image_id"], kind="stable").reset_index(drop=True)


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["image_id", "fiber_id"])
    return pd.read_csv(path, dtype={"fiber_id": "Int64"}).fillna("")


def _fiber_keys(path: Path) -> set[tuple[str, int]]:
    table = _load_existing(path)
    required = {"image_id", "fiber_id"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Excluded-decision file is missing columns: {missing}")
    return {
        (str(row.image_id), int(row.fiber_id))
        for row in table.itertuples(index=False)
        if pd.notna(row.fiber_id)
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Blinded visual fiber hunt for development mice")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--display-downsample", type=int, default=2)
    parser.add_argument("--review-name", default=DEFAULT_REVIEW_NAME)
    parser.add_argument("--sampling-stratum", default=DEFAULT_STRATUM)
    parser.add_argument(
        "--exclude-decisions",
        type=Path,
        action="append",
        default=[],
        help="Private prior-review CSV to keep out of this visual follow-up; may be repeated.",
    )
    args = parser.parse_args(argv)
    if args.display_downsample < 1:
        raise ValueError("--display-downsample must be at least 1")
    manifest_path = args.manifest.resolve()
    images = _load_development_manifest(manifest_path)
    output = _review_path(manifest_path, args.reviewer, args.review_name)
    existing = _load_existing(output)
    completed = _fiber_keys(output) if output.is_file() else set()
    unavailable = set(completed)
    for path in args.exclude_decisions:
        if not path.is_file():
            raise FileNotFoundError(f"Excluded-decision file does not exist: {path}")
        unavailable.update(_fiber_keys(path.resolve()))

    import napari
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QKeySequence
    from qtpy.QtWidgets import (
        QHBoxLayout,
        QLabel,
        QPushButton,
        QShortcut,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )

    viewer = napari.Viewer(title="FiberTypeQC blinded visual fiber hunt")
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
        QLabel("Choose a fiber by clicking it. Candidate/model outputs are not loaded.")
    )
    layout.addWidget(QLabel(f"Review stratum: {args.sampling_stratum}"))
    navigation = QHBoxLayout()
    previous_button = QPushButton("previous image [,]")
    next_button = QPushButton("next image [.]")
    navigation.addWidget(previous_button)
    navigation.addWidget(next_button)
    layout.addLayout(navigation)
    classify_header = QLabel("Classify selected fiber — saves, then clears selection")
    classify_header.setStyleSheet("font-weight: bold; margin-top: 6px;")
    layout.addWidget(classify_header)
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
    display_header = QLabel("Display controls — no label is saved")
    display_header.setStyleSheet("font-weight: bold; margin-top: 6px;")
    layout.addWidget(display_header)
    display_buttons: dict[str, QPushButton] = {}
    channel_controls = QHBoxLayout()
    for name, key in (("Type I", "Q"), ("Type IIa", "W"), ("Type IIb", "E"), ("laminin", "R")):
        short_name = {"Type I": "I", "Type IIa": "IIa", "laminin": "lam", "Type IIb": "IIb"}[name]
        button = QPushButton(f"{short_name} [{key}]")
        button.setCheckable(True)
        button.setChecked(True)
        display_buttons[name] = button
        channel_controls.addWidget(button)
    layout.addLayout(channel_controls)
    overlay_controls = QHBoxLayout()
    for name, key, visible in (
        ("fiber boundaries", "B", False),
        ("selected fiber outline", "O", True),
    ):
        button = QPushButton(f"{'bounds' if name == 'fiber boundaries' else 'outline'} [{key}]")
        button.setCheckable(True)
        button.setChecked(visible)
        display_buttons[name] = button
        overlay_controls.addWidget(button)
    reset_button = QPushButton("reset [0]")
    center_button = QPushButton("center target [F]")
    overlay_controls.addWidget(reset_button)
    overlay_controls.addWidget(center_button)
    layout.addLayout(overlay_controls)
    recovery_header = QLabel("Recovery")
    recovery_header.setStyleSheet("font-weight: bold; margin-top: 6px;")
    layout.addWidget(recovery_header)
    undo_button = QPushButton("Undo last saved label [Z]")
    layout.addWidget(undo_button)
    layout.addWidget(QLabel("Reopen controls: Window → Blinded visual hunt controls"))
    layout.addWidget(status)
    controls_dock = viewer.window.add_dock_widget(
        widget, area="right", name="Blinded visual hunt controls"
    )
    controls_dock.setMinimumWidth(280)
    controls_dock.setMaximumWidth(440)

    image_position = 0
    loaded_image = ""
    labels: np.ndarray | None = None
    selected_fiber_id: int | None = None
    undo_history: list[tuple[str, int]] = []

    def center_selected() -> None:
        if labels is None or selected_fiber_id is None:
            status.setText("Click a fiber before centering it")
            return
        mask = labels == selected_fiber_id
        if not mask.any():
            return
        y, x = np.argwhere(mask).mean(axis=0)
        viewer.camera.center = (float(y), float(x))
        viewer.camera.zoom = max(viewer.camera.zoom, 3)

    def update_context() -> None:
        row = images.iloc[image_position]
        context.setText(
            f"image {image_position + 1}/{len(images)} · mouse {row.mouse_id} · {row.section_id} "
            f"· {len(completed)} visual-hunt decisions saved"
        )

    def clear_selection(message: str = "Click a fiber to select it") -> None:
        nonlocal selected_fiber_id
        selected_fiber_id = None
        viewer.layers["selected fiber outline"].data = np.zeros_like(labels, dtype=np.uint8)
        status.setText(message)

    def load_image() -> None:
        nonlocal loaded_image, labels
        row = images.iloc[image_position]
        if loaded_image == str(row.image_id):
            update_context()
            clear_selection()
            return
        raw = load_multichannel_image(Path(str(row.raw_image_path)))[
            :, :: args.display_downsample, :: args.display_downsample
        ]
        labels = np.asarray(tifffile.imread(str(row.fiber_labels_path)), dtype=np.int32)[
            :: args.display_downsample, :: args.display_downsample
        ]
        if raw.shape[0] != len(OBSERVED_CHANNEL_NAMES) or raw.shape[1:] != labels.shape:
            raise ValueError(f"Unexpected raw/label layout for {row.image_id}")
        for layer in list(viewer.layers):
            viewer.layers.remove(layer)
        viewer.add_image(
            raw,
            channel_axis=0,
            name=list(OBSERVED_CHANNEL_NAMES),
            colormap=["magenta", "cyan", "gray", "yellow"],
            blending="additive",
        )
        boundaries = viewer.add_labels(labels, name="fiber boundaries", opacity=0.20)
        outline = viewer.add_labels(
            np.zeros_like(labels, dtype=np.uint8), name="selected fiber outline", opacity=0.85
        )
        boundaries.editable = False
        outline.editable = False
        boundaries.visible = False
        viewer.layers.selection.active = viewer.layers[OBSERVED_CHANNEL_NAMES[0]]
        loaded_image = str(row.image_id)
        update_context()
        clear_selection()

    def set_selected(fiber_id: int) -> None:
        nonlocal selected_fiber_id
        selected_fiber_id = fiber_id
        mask = labels == fiber_id
        viewer.layers["selected fiber outline"].data = find_boundaries(mask, mode="thick").astype(
            np.uint8
        )
        center_selected()
        status.setText(f"Fiber {fiber_id} selected — press a label key or click a label button")

    def select_at_position(_viewer: object, event: object) -> None:
        if labels is None or getattr(event, "type", "") != "mouse_press":
            return
        y, x = (int(round(value)) for value in event.position[-2:])
        if y < 0 or x < 0 or y >= labels.shape[0] or x >= labels.shape[1]:
            return
        fiber_id = int(labels[y, x])
        if fiber_id <= 0:
            status.setText("That pixel is outside a segmented fiber")
            return
        row = images.iloc[image_position]
        if (str(row.image_id), fiber_id) in unavailable:
            status.setText("That fiber was already reviewed and is not in this follow-up")
            return
        set_selected(fiber_id)

    def decide(key: str) -> None:
        nonlocal existing
        if selected_fiber_id is None:
            status.setText("Click a fiber before assigning a label")
            return
        row = images.iloc[image_position]
        record = {
            "reviewer": args.reviewer,
            "mouse_id": row.mouse_id,
            "image_id": row.image_id,
            "section_id": row.section_id,
            "fiber_id": selected_fiber_id,
            "sampling_stratum": args.sampling_stratum,
            "label": LABELS[key],
            "timestamp": datetime.now(UTC).isoformat(),
            "provenance": "blinded_visual_manual_selection; candidate/model outputs not loaded",
        }
        existing = pd.concat([existing, pd.DataFrame([record])], ignore_index=True)
        atomic_write_dataframe(output, existing)
        completed.add((str(row.image_id), selected_fiber_id))
        unavailable.add((str(row.image_id), selected_fiber_id))
        undo_history.append((str(row.image_id), selected_fiber_id))
        update_context()
        clear_selection("Saved. Click the next fiber to select it")

    def undo_last_decision() -> None:
        nonlocal existing, image_position
        if not undo_history:
            status.setText("No decision from this session is available to undo")
            return
        image_id, fiber_id = undo_history.pop()
        existing = existing.iloc[:-1].reset_index(drop=True)
        atomic_write_dataframe(output, existing)
        completed.discard((image_id, fiber_id))
        unavailable.discard((image_id, fiber_id))
        image_position = int(images.index[images["image_id"].eq(image_id)][0])
        load_image()
        set_selected(fiber_id)
        status.setText("Last decision removed; fiber restored")

    def change_image(delta: int) -> None:
        nonlocal image_position
        image_position = (image_position + delta) % len(images)
        load_image()

    def toggle_layer(name: str) -> None:
        layer = viewer.layers[name]
        layer.visible = not layer.visible
        display_buttons[name].setChecked(layer.visible)

    def reset_display() -> None:
        for name in OBSERVED_CHANNEL_NAMES:
            layer = viewer.layers[name]
            layer.visible = True
            layer.reset_contrast_limits()
            display_buttons[name].setChecked(True)
        viewer.layers["fiber boundaries"].visible = False
        viewer.layers["selected fiber outline"].visible = True
        display_buttons["fiber boundaries"].setChecked(False)
        display_buttons["selected fiber outline"].setChecked(True)

    shortcuts: list[QShortcut] = []

    def add_shortcut(sequence: str, callback: object) -> None:
        shortcut = QShortcut(QKeySequence(sequence), widget)
        shortcut.setContext(Qt.WindowShortcut)
        shortcut.activated.connect(callback)
        shortcuts.append(shortcut)

    for key in LABELS:
        add_shortcut(key, lambda value=key: decide(value))
    for key, layer_name in CHANNEL_TOGGLES.items():
        add_shortcut(key, lambda value=layer_name: toggle_layer(value))
    add_shortcut("B", lambda: toggle_layer("fiber boundaries"))
    add_shortcut("O", lambda: toggle_layer("selected fiber outline"))
    add_shortcut("0", reset_display)
    add_shortcut("F", center_selected)
    add_shortcut("Z", undo_last_decision)
    add_shortcut(",", lambda: change_image(-1))
    add_shortcut(".", lambda: change_image(1))
    for name, button in display_buttons.items():
        button.clicked.connect(lambda _checked=False, value=name: toggle_layer(value))
    previous_button.clicked.connect(lambda: change_image(-1))
    next_button.clicked.connect(lambda: change_image(1))
    reset_button.clicked.connect(reset_display)
    center_button.clicked.connect(center_selected)
    undo_button.clicked.connect(undo_last_decision)
    viewer.mouse_drag_callbacks.append(select_at_position)
    load_image()
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
