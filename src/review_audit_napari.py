from __future__ import annotations

import argparse
from pathlib import Path

import napari
import numpy as np
import pandas as pd
import tifffile
from magicgui import magicgui
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

from src.io_utils import load_multichannel_image
from src.label_masks import eroded_label_mask
from src.typing_display import (
    normalize_for_display,
    optional_channel,
    threshold_from_table,
    typing_composite,
    typing_signal_for_display,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review a sampled audit CSV in Napari by highlighting target fiber IDs."
    )
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-id", type=str, required=True)
    parser.add_argument(
        "--display-channel",
        type=int,
        default=2,
        help="Channel index to show as grayscale background for multichannel images.",
    )
    parser.add_argument("--iib-channel", type=int, default=0)
    parser.add_argument("--iia-channel", type=int, default=1)
    parser.add_argument("--membrane-channel", type=int, default=2)
    parser.add_argument("--signal-scale", type=float, default=5.0)
    parser.add_argument("--threshold-floor", type=float, default=1.0)
    parser.add_argument("--typing-preprocess", type=str, default="global_subtract")
    parser.add_argument("--typing-bg-quantile", type=float, default=0.02)
    parser.add_argument("--typing-tile-size", type=int, default=512)
    parser.add_argument("--typing-bg-sigma", type=float, default=24.0)
    parser.add_argument("--typing-smooth-sigma", type=float, default=0.8)
    parser.add_argument("--typing-erode-px", type=int, default=2)
    parser.add_argument(
        "--display-downsample",
        type=int,
        default=1,
        help="Downsample factor for display layers. Use 2 or 4 for large section exports.",
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=None,
        help=(
            "Optional output CSV for audit adjudications. Defaults to "
            "<audit_csv_stem>_reviewed_<image_id>.csv next to the audit CSV."
        ),
    )
    return parser


def _load_audit_subset(path: Path, image_id: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    subset = df.loc[df["image_id"].astype(str) == image_id].copy()
    if subset.empty:
        raise ValueError(f"No audit rows found for image_id={image_id}")
    subset["label"] = subset["label"].astype(int)
    sort_cols: list[str] = []
    ascending: list[bool] = []
    if "audit_bucket_count" in subset.columns:
        sort_cols.append("audit_bucket_count")
        ascending.append(False)
    elif "iia_evidence_score" in subset.columns:
        sort_cols.append("iia_evidence_score")
        ascending.append(False)
    sort_cols.append("label")
    ascending.append(True)
    return subset.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def _load_manifest_row(path: Path, image_id: str) -> pd.Series:
    df = pd.read_csv(path)
    row = df.loc[df["image_id"].astype(str) == image_id]
    if row.empty:
        raise ValueError(f"No manifest row found for image_id={image_id}")
    return row.iloc[0]


def _load_background_image(source_image: Path, display_channel: int) -> np.ndarray:
    image = load_multichannel_image(source_image)
    if image.ndim == 2:
        return image
    if display_channel < 0 or display_channel >= image.shape[0]:
        display_channel = min(max(display_channel, 0), image.shape[0] - 1)
    return image[display_channel]


def _find_output_files(output_root: Path, image_id: str) -> tuple[Path, Path]:
    image_dir = output_root / image_id
    labels_path = image_dir / f"{image_id}_cellpose_labels.tif"
    fibers_path = image_dir / f"{image_id}_fibers.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")
    if not fibers_path.exists():
        raise FileNotFoundError(f"Missing fibers file: {fibers_path}")
    return labels_path, fibers_path


def _display_channel(image: np.ndarray, channel: int | None) -> np.ndarray:
    if image.ndim == 2:
        return image
    if channel is None:
        return image[0]
    if channel < 0 or channel >= image.shape[0]:
        raise ValueError(f"Display channel {channel} out of range for image shape {image.shape}")
    return image[channel]


def _downsample_2d(data: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return data
    return data[::factor, ::factor]


def _downsample_image_chw(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return image
    if image.ndim == 2:
        return _downsample_2d(image, factor)
    return image[:, ::factor, ::factor]


def _default_review_output(audit_csv: Path, image_id: str) -> Path:
    stem = audit_csv.stem
    return audit_csv.with_name(f"{stem}_reviewed_{image_id}.csv")


def _label_centroid_map(
    labels: np.ndarray, target_labels: list[int]
) -> dict[int, tuple[float, float]]:
    centroid_map: dict[int, tuple[float, float]] = {}
    target_set = set(target_labels)
    ys, xs = np.nonzero(labels)
    for label_id in sorted(target_set):
        mask = labels == label_id
        if not mask.any():
            continue
        coords = np.column_stack(np.nonzero(mask))
        centroid = coords.mean(axis=0)
        centroid_map[label_id] = (float(centroid[0]), float(centroid[1]))
    return centroid_map


def _target_mask(labels: np.ndarray, label_id: int) -> np.ndarray:
    out = np.zeros_like(labels, dtype=np.int32)
    out[labels == label_id] = 1
    return out


def _scaled_display_data(data: np.ndarray, gain: float) -> np.ndarray:
    gain = max(0.0, float(gain))
    scaled = np.asarray(data, dtype=np.float32) * gain
    return np.clip(scaled, 0.0, 1.0)


def _focus_viewer_canvas(viewer: napari.Viewer) -> None:
    for attr_chain in (
        ("window", "_qt_viewer", "canvas", "native"),
        ("window", "qt_viewer", "canvas", "native"),
    ):
        obj = viewer
        try:
            for attr in attr_chain:
                obj = getattr(obj, attr)
            obj.setFocus()
            return
        except Exception:
            continue


def _build_points(
    audit: pd.DataFrame, centroid_map: dict[int, tuple[float, float]]
) -> tuple[np.ndarray, dict]:
    rows = []
    labels = []
    buckets = []
    for row in audit.itertuples(index=False):
        centroid = centroid_map.get(int(row.label))
        if centroid is None:
            continue
        rows.append(centroid)
        labels.append(str(int(row.label)))
        buckets.append(getattr(row, "audit_bucket_list", ""))
    points = (
        np.asarray(rows, dtype=np.float32)
        if rows
        else np.empty((0, 2), dtype=np.float32)
    )
    properties = {
        "label": np.asarray(labels, dtype=object),
        "bucket": np.asarray(buckets, dtype=object),
    }
    return points, properties


def _row_text(row: pd.Series) -> str:
    fields = [
        f"image_id={row.get('image_id', '')}",
        f"label={int(row.get('label', -1))}",
        f"fiber_type={row.get('fiber_type', '')}",
        f"buckets={row.get('audit_bucket_list', '')}",
        f"needs_review={row.get('needs_review', '')}",
        f"margin={row.get('model_margin', '')}",
        f"confidence={row.get('model_confidence', '')}",
        f"myosight_label={row.get('myosight_label', '')}",
        f"disparity={row.get('disparity_bucket', '')}",
        f"audit_corrected_type={row.get('audit_corrected_type', '')}",
        f"audit_is_uncertain={row.get('audit_is_uncertain', '')}",
        f"audit_is_excluded={row.get('audit_is_excluded', '')}",
        f"audit_notes={row.get('audit_notes', '')}",
    ]
    candidate_cols = [c for c in row.index if c.startswith('candidate_pred_')]
    for col in sorted(candidate_cols):
        fields.append(f"{col}={row.get(col, '')}")
    return "\n".join(fields)


def _toggle_layer_visibility(viewer: napari.Viewer, layer_name: str) -> None:
    layer = viewer.layers.get(layer_name)
    if layer is None:
        return
    layer.visible = not bool(layer.visible)


def _show_only_layers(viewer: napari.Viewer, visible_names: set[str]) -> None:
    persistent = {"cellpose_labels", "audit_target", "audit_points"}
    for layer in viewer.layers:
        if layer.name in persistent:
            layer.visible = True
        else:
            layer.visible = layer.name in visible_names


def _load_or_create_review_table(audit: pd.DataFrame, review_output: Path) -> pd.DataFrame:
    table = audit.copy()
    for col, default in (
        ("audit_corrected_type", ""),
        ("audit_is_uncertain", False),
        ("audit_is_excluded", False),
        ("audit_notes", ""),
    ):
        if col not in table.columns:
            table[col] = default
    if review_output.exists():
        saved = pd.read_csv(review_output, low_memory=False)
        keep = [
            c
            for c in (
                "image_id",
                "label",
                "audit_corrected_type",
                "audit_is_uncertain",
                "audit_is_excluded",
                "audit_notes",
            )
            if c in saved.columns
        ]
        if {"image_id", "label"}.issubset(keep):
            saved = saved.loc[:, keep].copy()
            saved["label"] = saved["label"].astype(int)
            table = table.drop(
                columns=[
                    "audit_corrected_type",
                    "audit_is_uncertain",
                    "audit_is_excluded",
                    "audit_notes",
                ],
                errors="ignore",
            ).merge(saved, on=["image_id", "label"], how="left")
            table["audit_corrected_type"] = table["audit_corrected_type"].fillna("")
            table["audit_is_uncertain"] = table["audit_is_uncertain"].fillna(False).astype(bool)
            table["audit_is_excluded"] = table["audit_is_excluded"].fillna(False).astype(bool)
            table["audit_notes"] = table["audit_notes"].fillna("")
    return table


def _save_review_table(review_table: pd.DataFrame, review_output: Path) -> None:
    review_output.parent.mkdir(parents=True, exist_ok=True)
    review_table.to_csv(review_output, index=False)


def main() -> None:
    args = build_parser().parse_args()
    display_downsample = max(1, int(args.display_downsample))
    audit = _load_audit_subset(args.audit_csv, args.image_id)
    review_output = args.review_output or _default_review_output(args.audit_csv, args.image_id)
    audit = _load_or_create_review_table(audit, review_output)
    manifest_row = _load_manifest_row(args.manifest, args.image_id)
    source_image = Path(str(manifest_row["source_image"]))
    labels_path, fibers_path = _find_output_files(args.output_root, args.image_id)

    full_image = load_multichannel_image(source_image)
    labels = np.asarray(tifffile.imread(labels_path))
    if labels.ndim != 2:
        labels = np.squeeze(labels)
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D labels image, got shape {labels.shape}")
    fibers = pd.read_csv(fibers_path)
    display_image = _downsample_image_chw(full_image, display_downsample)
    display_labels = _downsample_2d(labels, display_downsample).astype(np.int32)
    display_background = _display_channel(display_image, args.display_channel)

    target_labels = audit["label"].astype(int).tolist()
    centroid_map = _label_centroid_map(display_labels, target_labels)
    points, properties = _build_points(audit, centroid_map)
    type1_threshold = threshold_from_table(fibers, "type1_threshold")
    type2_threshold = threshold_from_table(fibers, "type2_threshold")
    display_erode_px = max(1, int(round(args.typing_erode_px / display_downsample)))
    signal_mask = eroded_label_mask(
        display_labels.astype(np.int32), display_erode_px
    ).astype(np.float32)

    viewer = napari.Viewer(title=f"Audit review: {args.image_id}")
    viewer.add_image(
        typing_composite(
            display_image,
            args.iib_channel,
            args.iia_channel,
            args.membrane_channel,
            type1_threshold=type1_threshold,
            type2_threshold=type2_threshold,
            preprocess=args.typing_preprocess,
            bg_quantile=args.typing_bg_quantile,
            tile_size=args.typing_tile_size,
            bg_sigma=args.typing_bg_sigma,
            smooth_sigma=args.typing_smooth_sigma,
            signal_scale=args.signal_scale,
            threshold_floor=args.threshold_floor,
            signal_mask=signal_mask,
        ),
        name="typing_composite_threshold_scaled",
        rgb=True,
        visible=False,
    )
    viewer.add_image(display_background, name="single_channel_gray", colormap="gray", visible=False)

    raw_layer_specs = (
        ("raw_iib", args.iib_channel, "magenta", True),
        ("raw_iia", args.iia_channel, "green", True),
        ("raw_membrane", args.membrane_channel, "gray", True),
    )
    for raw_name, raw_channel, raw_colormap, raw_visible in raw_layer_specs:
        raw = optional_channel(display_image, raw_channel)
        if raw is None:
            continue
        viewer.add_image(
            raw if raw_colormap != "gray" else normalize_for_display(raw),
            name=f"{raw_name}_ch{raw_channel}",
            colormap=raw_colormap,
            blending="additive" if raw_colormap != "gray" else "translucent",
            opacity=0.75 if raw_colormap != "gray" else 0.45,
            visible=raw_visible,
        )

    raw_iia_layer = None
    for layer in viewer.layers:
        if layer.name == f"raw_iia_ch{args.iia_channel}":
            raw_iia_layer = layer
            break

    iib_signal = optional_channel(display_image, args.iib_channel)
    if iib_signal is not None:
        viewer.add_image(
            typing_signal_for_display(
                iib_signal,
                threshold=type1_threshold,
                preprocess=args.typing_preprocess,
                bg_quantile=args.typing_bg_quantile,
                tile_size=args.typing_tile_size,
                bg_sigma=args.typing_bg_sigma,
                smooth_sigma=args.typing_smooth_sigma,
                signal_scale=args.signal_scale,
                threshold_floor=args.threshold_floor,
                mask=signal_mask,
            ),
            name=f"iib_signal_ch{args.iib_channel}",
            colormap="magenta",
            blending="additive",
            opacity=0.8,
            visible=False,
        )

    iia_signal = optional_channel(display_image, args.iia_channel)
    iia_signal_display = None
    iia_signal_layer = None
    if iia_signal is not None:
        iia_signal_display = typing_signal_for_display(
            iia_signal,
            threshold=type2_threshold,
            preprocess=args.typing_preprocess,
            bg_quantile=args.typing_bg_quantile,
            tile_size=args.typing_tile_size,
            bg_sigma=args.typing_bg_sigma,
            smooth_sigma=args.typing_smooth_sigma,
            signal_scale=args.signal_scale,
            threshold_floor=args.threshold_floor,
            mask=signal_mask,
        )
        iia_signal_layer = viewer.add_image(
            iia_signal_display,
            name=f"iia_signal_ch{args.iia_channel}",
            colormap="green",
            blending="additive",
            opacity=0.8,
            visible=False,
        )

    display_settings = {
        "iia_gain": 1.0,
        "iia_opacity": 0.8,
        "raw_iia_opacity": 0.75,
    }

    def apply_display_settings() -> None:
        if iia_signal_layer is not None and iia_signal_display is not None:
            iia_signal_layer.data = _scaled_display_data(
                iia_signal_display, display_settings["iia_gain"]
            )
            iia_signal_layer.opacity = float(display_settings["iia_opacity"])
        if raw_iia_layer is not None:
            raw_iia_layer.opacity = float(display_settings["raw_iia_opacity"])

    apply_display_settings()

    labels_layer = viewer.add_labels(display_labels, name="cellpose_labels", opacity=0.25)
    if hasattr(labels_layer, "contour"):
        labels_layer.contour = 1
    target_layer = viewer.add_labels(
        np.zeros_like(display_labels, dtype=np.int32), name="audit_target", opacity=0.8
    )
    if hasattr(target_layer, "color"):
        try:
            target_layer.color = {1: "white"}
        except Exception:
            pass
    if hasattr(target_layer, "contour"):
        target_layer.contour = 2
    viewer.add_points(
        points,
        name="audit_points",
        properties=properties,
        text={"string": "{label}", "size": 10, "color": "yellow", "anchor": "upper_left"},
        face_color="yellow",
        border_color="yellow",
        border_width=0.2,
        opacity=0.8,
        size=10,
    )

    state = {"index": 0}
    info_label = QLabel()
    info_label.setWordWrap(True)
    decision_label = QLabel("Current audit decision: <none>")
    decision_label.setWordWrap(True)
    save_label = QLabel("Review not saved yet.")
    save_label.setWordWrap(True)

    def save_current_state() -> None:
        _save_review_table(audit, review_output)
        message = f"Saved review CSV: {review_output}"
        save_label.setText(message)
        print(message)

    def set_current_label(label_name: str) -> None:
        idx = state["index"]
        audit.loc[idx, "audit_corrected_type"] = label_name
        audit.loc[idx, "audit_is_uncertain"] = label_name == "uncertain"
        audit.loc[idx, "audit_is_excluded"] = label_name == "exclude"
        save_current_state()
        update_view(idx)
        _focus_viewer_canvas(viewer)

    def clear_current_label() -> None:
        idx = state["index"]
        audit.loc[idx, "audit_corrected_type"] = ""
        audit.loc[idx, "audit_is_uncertain"] = False
        audit.loc[idx, "audit_is_excluded"] = False
        save_current_state()
        update_view(idx)
        _focus_viewer_canvas(viewer)

    def update_view(index: int) -> None:
        if audit.empty:
            return
        state["index"] = int(max(0, min(index, len(audit) - 1)))
        row = audit.iloc[state["index"]]
        label_id = int(row["label"])
        target_layer.data = _target_mask(display_labels, label_id)
        centroid = centroid_map.get(label_id)
        if centroid is not None:
            viewer.camera.center = (centroid[0], centroid[1])
            viewer.camera.zoom = max(viewer.camera.zoom, 3.0)
        info_label.setText(_row_text(row))
        if bool(row.get("audit_is_excluded", False)):
            decision = "exclude"
        elif bool(row.get("audit_is_uncertain", False)):
            decision = "uncertain"
        else:
            decision = str(row.get("audit_corrected_type", "")).strip() or "<none>"
        decision_label.setText(f"Current audit decision: {decision}")

    @magicgui(call_button="Next")
    def next_fiber() -> None:
        update_view(state["index"] + 1)

    @magicgui(call_button="Previous")
    def previous_fiber() -> None:
        update_view(state["index"] - 1)

    @magicgui(call_button="Mark IIb")
    def mark_iib() -> None:
        set_current_label("iib")

    @magicgui(call_button="Mark IIa")
    def mark_iia() -> None:
        set_current_label("iia")

    @magicgui(call_button="Mark IIx")
    def mark_iix() -> None:
        set_current_label("iix")

    @magicgui(call_button="Mark Hybrid")
    def mark_hybrid() -> None:
        set_current_label("hybrid")

    @magicgui(call_button="Mark Uncertain")
    def mark_uncertain() -> None:
        set_current_label("uncertain")

    @magicgui(call_button="Mark Exclude")
    def mark_exclude() -> None:
        set_current_label("exclude")

    @magicgui(call_button="Clear Audit Label")
    def clear_label() -> None:
        clear_current_label()

    @magicgui(call_button="Save Review CSV")
    def save_review() -> None:
        save_current_state()

    @magicgui(
        auto_call=True,
        iia_gain={
            "widget_type": "FloatSlider",
            "label": "IIa Enhanced Gain",
            "min": 0.1,
            "max": 1.5,
            "step": 0.05,
        },
        iia_opacity={
            "widget_type": "FloatSlider",
            "label": "IIa Enhanced Opacity",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
        },
        raw_iia_opacity={
            "widget_type": "FloatSlider",
            "label": "IIa Raw Opacity",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
        },
    )
    def adjust_iia_display(
        iia_gain: float = 1.0,
        iia_opacity: float = 0.8,
        raw_iia_opacity: float = 0.75,
    ) -> None:
        display_settings["iia_gain"] = float(iia_gain)
        display_settings["iia_opacity"] = float(iia_opacity)
        display_settings["raw_iia_opacity"] = float(raw_iia_opacity)
        apply_display_settings()

    panel = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(info_label)
    layout.addWidget(decision_label)
    layout.addWidget(save_label)
    layout.addWidget(adjust_iia_display.native)
    layout.addWidget(mark_iib.native)
    layout.addWidget(mark_iia.native)
    layout.addWidget(mark_iix.native)
    layout.addWidget(mark_hybrid.native)
    layout.addWidget(mark_uncertain.native)
    layout.addWidget(mark_exclude.native)
    layout.addWidget(clear_label.native)
    layout.addWidget(save_review.native)
    layout.addWidget(previous_fiber.native)
    layout.addWidget(next_fiber.native)
    panel.setLayout(layout)
    viewer.window.add_dock_widget(panel, area="right", name="Audit Review")
    _focus_viewer_canvas(viewer)

    @viewer.bind_key("n")
    def _next(event=None) -> None:
        next_fiber()

    @viewer.bind_key("p")
    def _previous(event=None) -> None:
        previous_fiber()

    @viewer.bind_key("b")
    def _mark_iib(event=None) -> None:
        mark_iib()

    @viewer.bind_key("a")
    def _mark_iia(event=None) -> None:
        mark_iia()

    @viewer.bind_key("x")
    def _mark_iix(event=None) -> None:
        mark_iix()

    @viewer.bind_key("h")
    def _mark_hybrid(event=None) -> None:
        mark_hybrid()

    @viewer.bind_key("u")
    def _mark_uncertain(event=None) -> None:
        mark_uncertain()

    @viewer.bind_key("e")
    def _mark_exclude(event=None) -> None:
        mark_exclude()

    @viewer.bind_key("c")
    def _clear(event=None) -> None:
        clear_label()

    @viewer.bind_key("1")
    def _toggle_raw_iib(event=None) -> None:
        _toggle_layer_visibility(viewer, f"raw_iib_ch{args.iib_channel}")

    @viewer.bind_key("2")
    def _toggle_raw_iia(event=None) -> None:
        _toggle_layer_visibility(viewer, f"raw_iia_ch{args.iia_channel}")

    @viewer.bind_key("3")
    def _toggle_iib_signal(event=None) -> None:
        _toggle_layer_visibility(viewer, f"iib_signal_ch{args.iib_channel}")

    @viewer.bind_key("4")
    def _toggle_iia_signal(event=None) -> None:
        _toggle_layer_visibility(viewer, f"iia_signal_ch{args.iia_channel}")

    @viewer.bind_key("5")
    def _toggle_membrane(event=None) -> None:
        _toggle_layer_visibility(viewer, f"raw_membrane_ch{args.membrane_channel}")

    @viewer.bind_key("6")
    def _toggle_composite(event=None) -> None:
        _toggle_layer_visibility(viewer, "typing_composite_threshold_scaled")

    @viewer.bind_key("0")
    def _show_context_only(event=None) -> None:
        _show_only_layers(viewer, set())

    @viewer.bind_key("7")
    def _show_raw_context(event=None) -> None:
        _show_only_layers(
            viewer,
            {
                f"raw_iib_ch{args.iib_channel}",
                f"raw_iia_ch{args.iia_channel}",
                f"raw_membrane_ch{args.membrane_channel}",
            },
        )

    @viewer.bind_key("8")
    def _show_enhanced_context(event=None) -> None:
        _show_only_layers(
            viewer,
            {
                f"iib_signal_ch{args.iib_channel}",
                f"iia_signal_ch{args.iia_channel}",
                f"raw_membrane_ch{args.membrane_channel}",
            },
        )

    @viewer.bind_key("9")
    def _show_composite_context(event=None) -> None:
        _show_only_layers(viewer, {"typing_composite_threshold_scaled"})

    update_view(0)
    print(f"source image: {source_image}")
    print(f"labels: {labels_path}")
    print(f"fibers: {fibers_path}")
    print(f"audit rows: {len(audit)}")
    print(f"review output: {review_output}")
    print(f"display downsample: {display_downsample}")
    print("Hotkeys: n/p nav | b/a/x/h/u/e classify | c clear")
    print("Layer hotkeys: 1 raw IIb | 2 raw IIa | 3 enhanced IIb | 4 enhanced IIa")
    print(
        "Layer hotkeys: 5 membrane | 6 composite | 0 labels-only | "
        "7 raw set | 8 enhanced set | 9 composite set"
    )
    napari.run()


if __name__ == "__main__":
    main()
