from __future__ import annotations

import argparse
import sys
from pathlib import Path

import napari
import numpy as np
import pandas as pd
import tifffile
from magicgui import magicgui
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget
from skimage.draw import polygon

from fibertypeqc.config import resolve_channel_config
from src.fiber_type_labels import REVIEW_TYPES, normalize_review_label, to_biological_label
from src.io_utils import load_multichannel_image
from src.label_masks import eroded_label_mask
from src.typing_display import (
    normalize_for_display,
    optional_channel,
    threshold_from_table,
    typing_composite,
    typing_signal_for_display,
)

TYPE_TO_CODE = {
    "iix_candidate": 1,
    "iix": 1,
    "iib": 2,
    "iia": 3,
    "hybrid_candidate": 4,
    "hybrid": 5,
    "uncertain": 6,
    "exclude": 7,
    "i": 8,
}
CODE_COLORS = {
    1: "gray",
    2: "magenta",
    3: "green",
    4: "yellow",
    5: "cyan",
    6: "orange",
    7: "red",
    8: "blue",
}
CODE_RGBA = {
    1: (0.55, 0.55, 0.55, 0.35),
    2: (1.00, 0.00, 1.00, 0.45),
    3: (0.00, 1.00, 0.00, 0.45),
    4: (1.00, 1.00, 0.00, 0.45),
    5: (0.00, 1.00, 1.00, 0.45),
    6: (1.00, 0.55, 0.00, 0.45),
    7: (1.00, 0.00, 0.00, 0.55),
    8: (0.10, 0.35, 1.00, 0.45),
}


class ReviewState:
    def __init__(
        self,
        image_path: Path,
        labels_path: Path,
        fibers_path: Path,
        output_path: Path,
    ) -> None:
        self.image_path = image_path
        self.labels_path = labels_path
        self.fibers_path = fibers_path
        self.output_path = output_path
        self.selected_fiber_id: int | None = None
        self.image: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.review: pd.DataFrame | None = None
        self.pending_polygon_label: int | None = None


def _fiber_id_column(df: pd.DataFrame) -> str:
    if "fiber_id" in df.columns:
        return "fiber_id"
    if "label" in df.columns:
        return "label"
    raise ValueError("Fiber table must contain either 'fiber_id' or 'label'.")


def _prediction_column(df: pd.DataFrame) -> str | None:
    for col in ("predicted_type", "fiber_type"):
        if col in df.columns:
            return col
    return None


def _load_or_create_review_table(fibers_path: Path, output_path: Path) -> pd.DataFrame:
    fibers = pd.read_csv(fibers_path)
    id_col = _fiber_id_column(fibers)
    pred_col = _prediction_column(fibers)

    review = pd.DataFrame()
    review["fiber_id"] = fibers[id_col].astype(int)
    if pred_col:
        review["predicted_type"] = fibers[pred_col].map(to_biological_label)
    else:
        review["predicted_type"] = "iix_candidate"
    review["corrected_type"] = ""
    review["emhc_manual_label"] = ""
    review["is_uncertain"] = False
    review["is_hybrid"] = False
    review["is_excluded"] = False
    review["label_source"] = "unreviewed"

    keep_cols = [
        c
        for c in (
            "confidence",
            "model_confidence",
            "model_margin",
            "prob_iib",
            "prob_iia",
            "prob_iix",
            "needs_review",
            "typing_signal_qc_flags",
            "area",
            "area_um2",
            "score_type1",
            "score_type2",
            "type1_mean",
            "type2_mean",
            "type1_p75",
            "type2_p75",
            "type1_p90",
            "type2_p90",
            "type1_pctl",
            "type2_pctl",
            "type1_coverage",
            "type2_coverage",
            "type1_signal_evidence",
            "type2_signal_evidence",
            "type1_threshold",
            "type2_threshold",
            "type1_p75_threshold",
            "type2_p75_threshold",
            "type1_p90_threshold",
            "type2_p90_threshold",
            "type1_pctl_threshold",
            "type2_pctl_threshold",
            "type1_cov_threshold",
            "type2_cov_threshold",
        )
        if c in fibers.columns
    ]
    if keep_cols:
        review = review.merge(
            fibers[[id_col, *keep_cols]].rename(columns={id_col: "fiber_id"}),
            on="fiber_id",
            how="left",
        )

    if output_path.exists():
        saved = pd.read_csv(output_path)
        if "fiber_id" not in saved.columns:
            raise ValueError(f"Existing review file has no fiber_id column: {output_path}")
        update_cols = [
            c
            for c in (
                "corrected_type",
                "emhc_manual_label",
                "is_uncertain",
                "is_hybrid",
                "is_excluded",
                "label_source",
            )
            if c in saved.columns
        ]
        saved_update = saved[["fiber_id", *update_cols]].copy()
        saved_update["fiber_id"] = saved_update["fiber_id"].astype(int)
        review = review.drop(columns=update_cols, errors="ignore").merge(
            saved_update,
            on="fiber_id",
            how="left",
        )
        for col in update_cols:
            if col == "corrected_type":
                review[col] = review[col].fillna("").map(normalize_review_label)
            elif col == "emhc_manual_label":
                review[col] = review[col].fillna("").astype(str).str.strip().str.lower()
            elif col == "label_source":
                review[col] = review[col].fillna("unreviewed")
            else:
                review[col] = review[col].fillna(False).astype(bool)

    return review.sort_values("fiber_id").reset_index(drop=True)


def _save_review(review: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(output_path, index=False)


def _display_channel(image: np.ndarray, channel: int | None) -> np.ndarray:
    if image.ndim == 2:
        return image
    if channel is None:
        return image[0]
    if channel < 0 or channel >= image.shape[0]:
        raise ValueError(
            f"Display channel {channel} out of range for image with {image.shape[0]} channels."
        )
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


def _build_review_overlay(labels: np.ndarray, review: pd.DataFrame) -> np.ndarray:
    max_label = int(labels.max())
    lut = np.zeros(max_label + 1, dtype=np.uint8)
    for row in review.itertuples(index=False):
        fiber_id = int(row.fiber_id)
        if fiber_id > max_label:
            continue
        corrected_type = normalize_review_label(row.corrected_type)
        predicted_type = to_biological_label(row.predicted_type)
        type_name = corrected_type if corrected_type else predicted_type
        lut[fiber_id] = TYPE_TO_CODE.get(type_name, TYPE_TO_CODE["iix_candidate"])
    return lut[labels]


def _build_review_rgba(labels: np.ndarray, review: pd.DataFrame) -> np.ndarray:
    type_map = _build_review_overlay(labels, review)
    out = np.zeros((*type_map.shape, 4), dtype=np.float32)
    for code, rgba in CODE_RGBA.items():
        out[type_map == code] = rgba
    return out


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin(("true", "1", "yes"))


def _flagged_fiber_ids(review: pd.DataFrame, max_label: int) -> np.ndarray:
    if "needs_review" not in review.columns:
        return np.empty(0, dtype=np.int32)
    needs_review = _as_bool_series(review["needs_review"])
    if "label_source" in review.columns:
        manually_reviewed = review["label_source"].fillna("").astype(str).eq("manual_gold")
    else:
        manually_reviewed = pd.Series(False, index=review.index)

    flagged = review.loc[needs_review & ~manually_reviewed, "fiber_id"].astype(int)
    flagged = flagged[(flagged > 0) & (flagged <= max_label)]
    return flagged.to_numpy(dtype=np.int32)


def _build_needs_review_points(labels: np.ndarray, review: pd.DataFrame) -> np.ndarray:
    max_label = int(labels.max())
    flagged = _flagged_fiber_ids(review, max_label)
    if flagged.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    ys, xs = np.nonzero(labels)
    lids = labels[ys, xs].astype(np.int32)
    valid = np.isin(lids, flagged)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float32)

    lids = lids[valid]
    ys = ys[valid].astype(np.float64)
    xs = xs[valid].astype(np.float64)
    counts = np.bincount(lids, minlength=max_label + 1).astype(np.float64)
    y_sum = np.bincount(lids, weights=ys, minlength=max_label + 1)
    x_sum = np.bincount(lids, weights=xs, minlength=max_label + 1)

    present = flagged[counts[flagged] > 0]
    points = np.column_stack((y_sum[present] / counts[present], x_sum[present] / counts[present]))
    return points.astype(np.float32)


def _row_for_fiber(review: pd.DataFrame, fiber_id: int) -> pd.Series | None:
    rows = review.index[review["fiber_id"] == int(fiber_id)].tolist()
    if not rows:
        return None
    return review.loc[rows[0]]


def launch_review(
    image_path: Path,
    labels_path: Path,
    fibers_path: Path,
    output_path: Path | None = None,
    nuclei_labels_path: Path | None = None,
    display_channel: int | None = None,
    i_channel: int | None = None,
    emhc_channel: int | None = None,
    dapi_channel: int | None = None,
    iib_channel: int | None = 0,
    iia_channel: int | None = 1,
    membrane_channel: int | None = 2,
    typing_preprocess: str = "global_subtract",
    typing_bg_quantile: float = 0.02,
    typing_tile_size: int = 512,
    typing_bg_sigma: float = 24.0,
    typing_smooth_sigma: float = 0.8,
    typing_erode_px: int = 2,
    signal_scale: float = 5.0,
    threshold_floor: float = 1.0,
    mask_signal_to_fibers: bool = True,
    display_downsample: int = 1,
    minimal_layers: bool = False,
    show_needs_review_flags: bool = True,
) -> None:
    display_downsample = max(1, int(display_downsample))
    output_path = output_path or fibers_path.with_name(f"{fibers_path.stem}_manual_review.csv")
    state = ReviewState(image_path, labels_path, fibers_path, output_path)
    state.image = load_multichannel_image(image_path)
    state.labels = np.asarray(tifffile.imread(labels_path)).astype(np.int32)
    state.review = _load_or_create_review_table(fibers_path, output_path)
    display_image = _downsample_image_chw(state.image, display_downsample)
    display_labels = _downsample_2d(state.labels, display_downsample).astype(np.int32)
    type1_threshold = threshold_from_table(state.review, "type1_threshold")
    type2_threshold = threshold_from_table(state.review, "type2_threshold")
    display_erode_px = max(1, int(round(typing_erode_px / display_downsample)))
    signal_mask = (
        eroded_label_mask(display_labels, display_erode_px).astype(np.float32)
        if mask_signal_to_fibers
        else None
    )

    viewer = napari.Viewer(title=f"Fiber type review: {image_path.name}")
    pickable_layers = []
    composite_layer = viewer.add_image(
        typing_composite(
            display_image,
            iib_channel,
            iia_channel,
            membrane_channel,
            type1_threshold=type1_threshold,
            type2_threshold=type2_threshold,
            preprocess=typing_preprocess,
            bg_quantile=typing_bg_quantile,
            tile_size=typing_tile_size,
            bg_sigma=typing_bg_sigma,
            smooth_sigma=typing_smooth_sigma,
            signal_scale=signal_scale,
            threshold_floor=threshold_floor,
            signal_mask=signal_mask,
        ),
        name="typing_composite_threshold_scaled",
        rgb=True,
    )
    pickable_layers.append(composite_layer)
    if not minimal_layers:
        raw_layers = (
            ("raw_i", i_channel, "blue"),
            ("raw_emhc", emhc_channel, "yellow"),
            ("raw_iib", iib_channel, "magenta"),
            ("raw_iia", iia_channel, "green"),
            ("raw_membrane", membrane_channel, "gray"),
            ("raw_dapi", dapi_channel, "cyan"),
        )
        for raw_name, raw_channel, raw_colormap in raw_layers:
            raw = optional_channel(display_image, raw_channel)
            if raw is None:
                continue
            raw_layer = viewer.add_image(
                raw,
                name=f"{raw_name}_ch{raw_channel}",
                colormap=raw_colormap,
                blending="additive" if raw_colormap != "gray" else "translucent",
                opacity=0.7 if raw_colormap != "gray" else 0.45,
                visible=False,
            )
            pickable_layers.append(raw_layer)
    if (not minimal_layers) and i_channel is not None:
        type_i = optional_channel(display_image, i_channel)
        if type_i is not None:
            type_i_layer = viewer.add_image(
                typing_signal_for_display(
                    type_i,
                    threshold=None,
                    preprocess=typing_preprocess,
                    bg_quantile=typing_bg_quantile,
                    tile_size=typing_tile_size,
                    bg_sigma=typing_bg_sigma,
                    smooth_sigma=typing_smooth_sigma,
                    signal_scale=signal_scale,
                    threshold_floor=threshold_floor,
                    mask=signal_mask,
                ),
                name=f"i_signal_ch{i_channel}",
                colormap="blue",
                blending="additive",
                opacity=0.7,
                visible=True,
            )
            pickable_layers.append(type_i_layer)
    if (not minimal_layers) and iib_channel is not None:
        type1 = optional_channel(display_image, iib_channel)
        if type1 is not None:
            type1_layer = viewer.add_image(
                typing_signal_for_display(
                    type1,
                    threshold=type1_threshold,
                    preprocess=typing_preprocess,
                    bg_quantile=typing_bg_quantile,
                    tile_size=typing_tile_size,
                    bg_sigma=typing_bg_sigma,
                    smooth_sigma=typing_smooth_sigma,
                    signal_scale=signal_scale,
                    threshold_floor=threshold_floor,
                    mask=signal_mask,
                ),
                name=f"iib_signal_ch{iib_channel}",
                colormap="magenta",
                blending="additive",
                opacity=0.7,
                visible=False,
            )
            pickable_layers.append(type1_layer)
    if (not minimal_layers) and iia_channel is not None:
        type2 = optional_channel(display_image, iia_channel)
        if type2 is not None:
            type2_layer = viewer.add_image(
                typing_signal_for_display(
                    type2,
                    threshold=type2_threshold,
                    preprocess=typing_preprocess,
                    bg_quantile=typing_bg_quantile,
                    tile_size=typing_tile_size,
                    bg_sigma=typing_bg_sigma,
                    smooth_sigma=typing_smooth_sigma,
                    signal_scale=signal_scale,
                    threshold_floor=threshold_floor,
                    mask=signal_mask,
                ),
                name=f"iia_signal_ch{iia_channel}",
                colormap="green",
                blending="additive",
                opacity=0.7,
                visible=False,
            )
            pickable_layers.append(type2_layer)
    if (not minimal_layers) and membrane_channel is not None:
        membrane = optional_channel(display_image, membrane_channel)
        if membrane is not None:
            membrane_layer = viewer.add_image(
                normalize_for_display(membrane),
                name=f"membrane_ch{membrane_channel}",
                colormap="gray",
                opacity=0.45,
                visible=False,
            )
            pickable_layers.append(membrane_layer)
    if not minimal_layers:
        gray_layer = viewer.add_image(
            _display_channel(display_image, display_channel),
            name="single_channel_gray",
            colormap="gray",
            visible=False,
        )
        pickable_layers.append(gray_layer)
    labels_layer = viewer.add_labels(display_labels, name="cellpose_labels", opacity=0.25)
    pickable_layers.append(labels_layer)
    if hasattr(labels_layer, "contour"):
        labels_layer.contour = 1
    if nuclei_labels_path is not None:
        nuclei_full = np.asarray(tifffile.imread(nuclei_labels_path)).astype(np.int32)
        if nuclei_full.shape != state.labels.shape:
            raise ValueError(
                f"Nuclear labels have shape {nuclei_full.shape}, expected {state.labels.shape}."
            )
        viewer.add_labels(
            _downsample_2d(nuclei_full, display_downsample),
            name="nuclei_labels",
            opacity=0.45,
            visible=True,
        )
    polygon_layer = viewer.add_shapes(
        name="new_fiber_polygon",
        ndim=2,
        edge_color="cyan",
        face_color="transparent",
        edge_width=2,
        visible=True,
    )
    needs_review_layer = None
    if show_needs_review_flags:
        needs_review_points = _build_needs_review_points(display_labels, state.review)
        try:
            needs_review_layer = viewer.add_points(
                needs_review_points,
                name="needs_review_flags",
                size=16,
                face_color="yellow",
                border_color="black",
                opacity=0.85,
                visible=True,
            )
        except TypeError:
            needs_review_layer = viewer.add_points(
                needs_review_points,
                name="needs_review_flags",
                size=16,
                face_color="yellow",
                edge_color="black",
                opacity=0.85,
                visible=True,
            )
        pickable_layers.append(needs_review_layer)
    selected_fiber_layer = viewer.add_labels(
        np.zeros_like(display_labels, dtype=np.uint8),
        name="selected_fiber",
        opacity=0.8,
        visible=True,
    )
    pickable_layers.append(selected_fiber_layer)
    if hasattr(selected_fiber_layer, "color"):
        try:
            selected_fiber_layer.color = {1: "white"}
        except Exception:
            pass
    if hasattr(selected_fiber_layer, "contour"):
        selected_fiber_layer.contour = 2

    corrected_labels_path = labels_path.with_name(f"{labels_path.stem}_corrected.tif")
    overlay_layer = None
    overlay_rgb_layer = None
    if not minimal_layers:
        overlay_layer = viewer.add_labels(
            _build_review_overlay(display_labels, state.review),
            name="fiber_type_review_overlay",
            opacity=0.25,
            visible=False,
        )
        pickable_layers.append(overlay_layer)
        if hasattr(overlay_layer, "color"):
            try:
                overlay_layer.color = CODE_COLORS
            except Exception:
                pass
        overlay_rgb_layer = viewer.add_image(
            _build_review_rgba(display_labels, state.review),
            name="fiber_type_fixed_color_overlay",
            rgb=True,
            blending="translucent",
            visible=False,
        )
        pickable_layers.append(overlay_rgb_layer)

    status = QLabel("")
    status.setWordWrap(True)
    hotkey_label = QLabel(
        "Fiber type: i Type I | b IIb | a IIa | x IIx | h hybrid | u uncertain | e exclude\n"
        "eMHC: p positive | n negative"
    )
    hotkey_label.setWordWrap(True)

    def set_status(message: str) -> None:
        status.setText(message)
        print(message)

    def selected_summary() -> str:
        if state.selected_fiber_id is None:
            return "No fiber selected. Click a fiber in the cellpose_labels layer."
        row = _row_for_fiber(state.review, state.selected_fiber_id)
        if row is None:
            return f"Selected fiber_id={state.selected_fiber_id}, not found in fiber table."
        corrected = (
            row.corrected_type if isinstance(row.corrected_type, str) and row.corrected_type else ""
        )
        corrected_txt = corrected if corrected else "(not corrected)"
        bits = [
            f"fiber_id={state.selected_fiber_id}",
            f"predicted={row.predicted_type}",
            f"corrected={corrected_txt}",
        ]
        if "confidence" in row.index and not pd.isna(row.confidence):
            bits.append(f"confidence={float(row.confidence):.3f}")
        if "model_margin" in row.index and not pd.isna(row.model_margin):
            bits.append(f"margin={float(row.model_margin):.3f}")
        prob_bits = []
        for label, col in (("IIb", "prob_iib"), ("IIa", "prob_iia"), ("IIx", "prob_iix")):
            if col in row.index and not pd.isna(row[col]):
                prob_bits.append(f"{label}={float(row[col]):.2f}")
        if prob_bits:
            bits.append("probs " + "/".join(prob_bits))
        if "needs_review" in row.index and not pd.isna(row.needs_review):
            bits.append(f"needs_review={bool(row.needs_review)}")
        if "typing_signal_qc_flags" in row.index:
            flags = str(row.typing_signal_qc_flags)
            if flags and flags.lower() != "nan":
                bits.append(f"flags={flags}")
        return " | ".join(bits)

    def refresh_overlay() -> None:
        if overlay_layer is not None:
            overlay_layer.data = _build_review_overlay(display_labels, state.review)
        if overlay_rgb_layer is not None:
            overlay_rgb_layer.data = _build_review_rgba(display_labels, state.review)
        if needs_review_layer is not None:
            needs_review_layer.data = _build_needs_review_points(display_labels, state.review)

    def save_and_report(prefix: str = "Saved") -> None:
        _save_review(state.review, state.output_path)
        n_reviewed = int((state.review["label_source"] == "manual_gold").sum())
        set_status(f"{prefix}: {state.output_path} | reviewed={n_reviewed} | {selected_summary()}")

    def assign_type(type_name: str) -> None:
        type_name = normalize_review_label(type_name)
        if state.selected_fiber_id is None:
            set_status("Click a fiber before assigning a type.")
            return
        idx = state.review.index[state.review["fiber_id"] == int(state.selected_fiber_id)].tolist()
        if not idx:
            set_status(
                f"fiber_id={state.selected_fiber_id} is not present in {state.fibers_path.name}"
            )
            return
        i = idx[0]
        state.review.loc[i, "corrected_type"] = type_name
        state.review.loc[i, "is_uncertain"] = type_name == "uncertain"
        state.review.loc[i, "is_hybrid"] = type_name == "hybrid"
        state.review.loc[i, "is_excluded"] = type_name == "exclude"
        state.review.loc[i, "label_source"] = "manual_gold"
        refresh_overlay()
        save_and_report(f"Assigned {type_name}")

    def assign_emhc(label: str) -> None:
        if state.selected_fiber_id is None:
            set_status("Click a fiber before assigning eMHC.")
            return
        idx = state.review.index[state.review["fiber_id"] == int(state.selected_fiber_id)].tolist()
        if idx:
            state.review.loc[idx[0], "emhc_manual_label"] = label
            save_and_report(f"Assigned eMHC={label}")

    def pick_fiber(layer, event) -> None:
        if event.type != "mouse_press":
            return
        value = labels_layer.get_value(event.position, world=True)
        if value is None:
            return
        if isinstance(value, tuple):
            if not value:
                return
            value = value[0]
        fiber_id = int(value)
        if fiber_id <= 0:
            state.selected_fiber_id = None
            selected_fiber_layer.data = np.zeros_like(display_labels, dtype=np.uint8)
            set_status("Background selected. Click inside a labeled fiber.")
            return
        state.selected_fiber_id = fiber_id
        labels_layer.selected_label = fiber_id
        selected_fiber_layer.data = (display_labels == fiber_id).astype(np.uint8)
        set_status(selected_summary())

    for layer in pickable_layers:
        layer.mouse_drag_callbacks.append(pick_fiber)

    @magicgui(
        call_button="Assign Selected Fiber",
        corrected_type={"choices": REVIEW_TYPES},
    )
    def assign_widget(corrected_type: str = "iib") -> None:
        assign_type(corrected_type)

    @magicgui(
        call_button="Assign Selected eMHC",
        emhc_label={"choices": ["positive", "negative", "uncertain"]},
    )
    def assign_emhc_widget(emhc_label: str = "positive") -> None:
        if emhc_label not in {"positive", "negative", "uncertain"}:
            set_status("eMHC label must be positive, negative, or uncertain.")
            return
        assign_emhc(emhc_label)

    @magicgui(call_button="Clear Selected Correction")
    def clear_widget() -> None:
        if state.selected_fiber_id is None:
            set_status("Click a fiber before clearing a correction.")
            return
        idx = state.review.index[state.review["fiber_id"] == int(state.selected_fiber_id)].tolist()
        if not idx:
            set_status(
                f"fiber_id={state.selected_fiber_id} is not present in {state.fibers_path.name}"
            )
            return
        i = idx[0]
        state.review.loc[i, "corrected_type"] = ""
        state.review.loc[i, "is_uncertain"] = False
        state.review.loc[i, "is_hybrid"] = False
        state.review.loc[i, "is_excluded"] = False
        state.review.loc[i, "label_source"] = "unreviewed"
        refresh_overlay()
        save_and_report("Cleared correction")

    @magicgui(call_button="Save Review CSV")
    def save_widget() -> None:
        save_and_report()

    def require_full_resolution() -> bool:
        if display_downsample == 1:
            return True
        set_status(
            "Segmentation repair requires --display-downsample 1 to preserve label geometry."
        )
        return False

    @magicgui(call_button="Add Missing Fiber (polygon)")
    def add_fiber_polygon_widget() -> None:
        if not require_full_resolution():
            return
        new_label = int(np.max(labels_layer.data)) + 1
        polygon_layer.data = []
        state.pending_polygon_label = new_label
        polygon_layer.mode = "add_polygon"
        viewer.layers.selection.active = polygon_layer
        set_status(
            f"Click around missing fiber {new_label}; double-click the final point, then "
            "Commit New Fiber Polygon."
        )

    @magicgui(call_button="Commit New Fiber Polygon")
    def commit_fiber_polygon_widget() -> None:
        if not require_full_resolution():
            return
        new_label = state.pending_polygon_label
        if new_label is None:
            set_status("Click Add Missing Fiber (polygon) before committing a polygon.")
            return
        if not polygon_layer.data:
            set_status("Draw a polygon first; double-click its final point to finish it.")
            return
        coords = np.asarray(polygon_layer.data[-1], dtype=np.float64)
        if coords.ndim != 2 or coords.shape[0] < 3:
            set_status("A new fiber polygon needs at least three points.")
            return
        row_coords, col_coords = polygon(coords[:, 0], coords[:, 1], shape=labels_layer.data.shape)
        empty = labels_layer.data[row_coords, col_coords] == 0
        if not np.any(empty):
            set_status("The polygon overlaps existing labels; redraw it inside the missing fiber.")
            return
        labels_layer.data[row_coords[empty], col_coords[empty]] = new_label
        labels_layer.refresh()
        state.selected_fiber_id = new_label
        selected_fiber_layer.data = (labels_layer.data == new_label).astype(np.uint8)
        polygon_layer.data = []
        state.pending_polygon_label = None
        skipped = int((~empty).sum())
        suffix = f" Kept {skipped} overlapping pixels unchanged." if skipped else ""
        set_status(
            f"Added fiber label {new_label} from polygon. "
            "Save Corrected Segmentation to persist it."
            f"{suffix}"
        )

    @magicgui(call_button="Add Missing Fiber (brush)")
    def add_fiber_brush_widget() -> None:
        if not require_full_resolution():
            return
        new_label = int(np.max(labels_layer.data)) + 1
        labels_layer.selected_label = new_label
        labels_layer.mode = "paint"
        viewer.layers.selection.active = labels_layer
        set_status(f"Paint new fiber label {new_label}, then Save Corrected Segmentation.")

    @magicgui(call_button="Edit Selected Boundary (paint)")
    def edit_boundary_widget() -> None:
        if not require_full_resolution() or state.selected_fiber_id is None:
            return
        labels_layer.selected_label = int(state.selected_fiber_id)
        labels_layer.mode = "paint"
        set_status("Paint the selected boundary; use Napari erase mode to remove pixels.")

    @magicgui(call_button="Delete Selected Artifact")
    def delete_fiber_widget() -> None:
        if not require_full_resolution() or state.selected_fiber_id is None:
            return
        labels_layer.data[labels_layer.data == int(state.selected_fiber_id)] = 0
        labels_layer.refresh()
        set_status("Deleted selected label. Save Corrected Segmentation to persist it.")

    @magicgui(call_button="Save Corrected Segmentation")
    def save_corrected_labels_widget() -> None:
        if not require_full_resolution():
            return
        tifffile.imwrite(corrected_labels_path, np.asarray(labels_layer.data, dtype=np.int32))
        set_status(
            f"Saved corrected labels: {corrected_labels_path}. Re-quantify this mask before review."
        )

    def _iib_key(layer_or_viewer) -> None:
        assign_type("iib")

    def _i_key(layer_or_viewer) -> None:
        assign_type("i")

    def _iia_key(layer_or_viewer) -> None:
        assign_type("iia")

    def _iix_key(layer_or_viewer) -> None:
        assign_type("iix")

    def _hybrid_key(layer_or_viewer) -> None:
        assign_type("hybrid")

    def _uncertain_key(layer_or_viewer) -> None:
        assign_type("uncertain")

    def _exclude_key(layer_or_viewer) -> None:
        assign_type("exclude")

    def _emhc_positive_key(layer_or_viewer) -> None:
        assign_emhc("positive")

    def _emhc_negative_key(layer_or_viewer) -> None:
        assign_emhc("negative")

    for target in (viewer, *pickable_layers):
        target.bind_key("i", _i_key, overwrite=True)
        target.bind_key("b", _iib_key, overwrite=True)
        target.bind_key("a", _iia_key, overwrite=True)
        target.bind_key("x", _iix_key, overwrite=True)
        target.bind_key("h", _hybrid_key, overwrite=True)
        target.bind_key("u", _uncertain_key, overwrite=True)
        target.bind_key("e", _exclude_key, overwrite=True)
        target.bind_key("p", _emhc_positive_key, overwrite=True)
        target.bind_key("n", _emhc_negative_key, overwrite=True)

    panel = QWidget()
    panel.setMinimumWidth(320)
    panel.setMaximumWidth(520)
    layout = QVBoxLayout(panel)
    layout.addWidget(hotkey_label)
    layout.addWidget(assign_widget.native)
    layout.addWidget(assign_emhc_widget.native)
    layout.addWidget(clear_widget.native)
    layout.addWidget(save_widget.native)
    layout.addWidget(add_fiber_polygon_widget.native)
    layout.addWidget(commit_fiber_polygon_widget.native)
    layout.addWidget(add_fiber_brush_widget.native)
    layout.addWidget(edit_boundary_widget.native)
    layout.addWidget(delete_fiber_widget.native)
    layout.addWidget(save_corrected_labels_widget.native)
    layout.addWidget(status)
    viewer.window.add_dock_widget(panel, area="right", name="Manual Fiber Type Review")

    set_status(
        "Ready. Composite is threshold-scaled: IIb=magenta, IIa=green, membrane=gray; "
        "Type I=blue.\n"
        f"Type signal below {threshold_floor:.1f}x threshold is hidden.\n"
        f"display_downsample={display_downsample}; minimal_layers={minimal_layers}.\n"
        "Click a fiber, then press i=Type I, b=IIb, a=IIa, x=IIx, h=hybrid, "
        "u=uncertain, e=exclude; p=eMHC positive, n=eMHC negative."
    )
    napari.run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Click-to-annotate napari fiber type review UI.")
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Multichannel TIFF/CZI image path.",
    )
    parser.add_argument("--labels", required=True, type=Path, help="Cellpose label TIFF path.")
    parser.add_argument(
        "--nuclei-labels",
        type=Path,
        default=None,
        help="Optional DAPI/nuclear label TIFF to display as a separate overlay.",
    )
    parser.add_argument(
        "--fibers",
        required=True,
        type=Path,
        help="Per-fiber CSV with label/fiber_id rows.",
    )
    parser.add_argument("--output", type=Path, help="Manual review CSV path.")
    parser.add_argument(
        "--channel-config",
        type=Path,
        default=None,
        help=(
            "YAML file with panel-aware channel mapping under 'channels' "
            "and optional 'classification'."
        ),
    )
    parser.add_argument(
        "--display-channel",
        type=int,
        help="Channel to display as grayscale background.",
    )
    parser.add_argument(
        "--i-channel",
        type=int,
        default=None,
        help="Optional type I marker channel index.",
    )
    parser.add_argument(
        "--iia-channel",
        type=int,
        default=None,
        help="Optional IIa marker channel index.",
    )
    parser.add_argument(
        "--iib-channel",
        type=int,
        default=None,
        help="Optional IIb marker channel index.",
    )
    parser.add_argument(
        "--iix-channel",
        type=int,
        default=None,
        help="Optional IIx marker channel index.",
    )
    parser.add_argument(
        "--emhc-channel",
        type=int,
        default=None,
        help="Optional eMHC marker channel index for separate regeneration review.",
    )
    parser.add_argument(
        "--dapi-channel",
        type=int,
        default=None,
        help="Optional DAPI channel index.",
    )
    parser.add_argument(
        "--membrane-channel",
        type=int,
        default=None,
        help="Membrane signal channel.",
    )
    parser.add_argument(
        "--type1-channel",
        type=int,
        default=None,
        help="Legacy alias for --iib-channel.",
    )
    parser.add_argument(
        "--type2-channel",
        type=int,
        default=None,
        help="Legacy alias for --iia-channel.",
    )
    parser.add_argument(
        "--typing-preprocess",
        type=str,
        default="global_subtract",
        choices=["raw", "global_subtract", "tile_subtract", "gaussian_subtract"],
        help="Type-channel preprocessing for review display.",
    )
    parser.add_argument(
        "--typing-bg-quantile",
        type=float,
        default=0.02,
        help="Low quantile subtracted when using global_subtract or tile_subtract.",
    )
    parser.add_argument("--typing-tile-size", type=int, default=512)
    parser.add_argument(
        "--typing-bg-sigma",
        type=float,
        default=24.0,
        help="Background subtraction sigma for review display.",
    )
    parser.add_argument(
        "--typing-smooth-sigma",
        type=float,
        default=0.8,
        help="Smoothing sigma for review display.",
    )
    parser.add_argument(
        "--typing-erode-px",
        type=int,
        default=2,
        help="Pixels to erode each fiber for type-channel review display.",
    )
    parser.add_argument(
        "--signal-scale",
        type=float,
        default=5.0,
        help="Signal multiple above threshold that maps to full brightness.",
    )
    parser.add_argument(
        "--threshold-floor",
        type=float,
        default=1.0,
        help="Signal multiple below which type-channel display is hidden.",
    )
    parser.add_argument(
        "--show-type-background",
        action="store_true",
        help="Do not mask type-channel display to segmented fibers.",
    )
    parser.add_argument(
        "--display-downsample",
        type=int,
        default=1,
        help="Downsample review display layers only; corrections still save original fiber IDs.",
    )
    parser.add_argument(
        "--minimal-layers",
        action="store_true",
        help="Open only essential review layers for very large images.",
    )
    parser.add_argument(
        "--no-needs-review-flags",
        action="store_true",
        help="Do not add yellow needs_review point markers.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    channel_cfg, channel_warnings = resolve_channel_config(
        channel_config_path=args.channel_config,
        i_channel=args.i_channel,
        iia_channel=args.iia_channel,
        iib_channel=args.iib_channel,
        iix_channel=args.iix_channel,
        emhc_channel=args.emhc_channel,
        dapi_channel=args.dapi_channel,
        type1_channel=args.type1_channel,
        type2_channel=args.type2_channel,
        membrane_channel=args.membrane_channel,
    )
    for warning in channel_warnings:
        print(f"Warning: {warning}", file=sys.stderr, flush=True)
    launch_review(
        image_path=args.image,
        labels_path=args.labels,
        fibers_path=args.fibers,
        output_path=args.output,
        nuclei_labels_path=args.nuclei_labels,
        display_channel=args.display_channel,
        i_channel=channel_cfg.i_channel,
        emhc_channel=channel_cfg.emhc_channel,
        dapi_channel=channel_cfg.dapi_channel,
        iib_channel=channel_cfg.iib_channel,
        iia_channel=channel_cfg.iia_channel,
        membrane_channel=channel_cfg.membrane_channel,
        typing_preprocess=args.typing_preprocess,
        typing_bg_quantile=args.typing_bg_quantile,
        typing_tile_size=args.typing_tile_size,
        typing_bg_sigma=args.typing_bg_sigma,
        typing_smooth_sigma=args.typing_smooth_sigma,
        typing_erode_px=args.typing_erode_px,
        signal_scale=args.signal_scale,
        threshold_floor=args.threshold_floor,
        mask_signal_to_fibers=(not args.show_type_background),
        display_downsample=args.display_downsample,
        minimal_layers=args.minimal_layers,
        show_needs_review_flags=(not args.no_needs_review_flags),
    )


if __name__ == "__main__":
    main()
