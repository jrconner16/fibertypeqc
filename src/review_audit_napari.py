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
    return subset.sort_values(
        ["audit_bucket_count", "label"], ascending=[False, True]
    ).reset_index(drop=True)


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
    audit = _load_audit_subset(args.audit_csv, args.image_id)
    review_output = args.review_output or _default_review_output(args.audit_csv, args.image_id)
    audit = _load_or_create_review_table(audit, review_output)
    manifest_row = _load_manifest_row(args.manifest, args.image_id)
    source_image = Path(str(manifest_row["source_image"]))
    labels_path, fibers_path = _find_output_files(args.output_root, args.image_id)

    background = _load_background_image(source_image, args.display_channel)
    labels = np.asarray(tifffile.imread(labels_path))
    if labels.ndim != 2:
        labels = np.squeeze(labels)
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D labels image, got shape {labels.shape}")

    target_labels = audit["label"].astype(int).tolist()
    centroid_map = _label_centroid_map(labels, target_labels)
    points, properties = _build_points(audit, centroid_map)

    viewer = napari.Viewer(title=f"Audit review: {args.image_id}")
    viewer.add_image(background, name="background", colormap="gray")
    viewer.add_labels(labels, name="cellpose_labels", opacity=0.25)
    target_layer = viewer.add_labels(
        np.zeros_like(labels, dtype=np.int32), name="audit_target", opacity=0.8
    )
    viewer.add_points(
        points,
        name="audit_points",
        properties=properties,
        text={"string": "{label}", "size": 10, "color": "yellow", "anchor": "upper_left"},
        face_color="transparent",
        edge_color="yellow",
        size=10,
    )

    state = {"index": 0}
    info_label = QLabel()
    info_label.setWordWrap(True)

    def save_current_state() -> None:
        _save_review_table(audit, review_output)

    def set_current_label(label_name: str) -> None:
        idx = state["index"]
        audit.loc[idx, "audit_corrected_type"] = label_name
        audit.loc[idx, "audit_is_uncertain"] = label_name == "uncertain"
        audit.loc[idx, "audit_is_excluded"] = label_name == "exclude"
        save_current_state()
        update_view(idx)

    def clear_current_label() -> None:
        idx = state["index"]
        audit.loc[idx, "audit_corrected_type"] = ""
        audit.loc[idx, "audit_is_uncertain"] = False
        audit.loc[idx, "audit_is_excluded"] = False
        save_current_state()
        update_view(idx)

    def update_view(index: int) -> None:
        if audit.empty:
            return
        state["index"] = int(max(0, min(index, len(audit) - 1)))
        row = audit.iloc[state["index"]]
        label_id = int(row["label"])
        target_layer.data = _target_mask(labels, label_id)
        centroid = centroid_map.get(label_id)
        if centroid is not None:
            viewer.camera.center = (centroid[0], centroid[1])
            viewer.camera.zoom = max(viewer.camera.zoom, 3.0)
        info_label.setText(_row_text(row))

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

    panel = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(info_label)
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

    update_view(0)
    print(f"source image: {source_image}")
    print(f"labels: {labels_path}")
    print(f"fibers: {fibers_path}")
    print(f"audit rows: {len(audit)}")
    print(f"review output: {review_output}")
    napari.run()


if __name__ == "__main__":
    main()
