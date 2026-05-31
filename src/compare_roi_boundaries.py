from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import roifile
import tifffile
from skimage.measure import find_contours

from src.io_utils import load_multichannel_image
from src.label_masks import erode_labels

PIXEL_SIZE_UM = 0.908


def polygon_area(coords: np.ndarray) -> float:
    x = coords[:, 0]
    y = coords[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def load_display_image(path: Path, channel: int | None = None) -> np.ndarray:
    if channel is not None:
        arr = load_multichannel_image(path)[channel]
        arr = arr.astype(float)
        lo, hi = np.percentile(arr, [1, 99.8])
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
        return np.repeat(arr[..., None], 3, axis=2)

    arr = np.asarray(tifffile.imread(path))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[-1] > 4:
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    arr = arr.astype(float)
    lo, hi = np.percentile(arr, [1, 99.8])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return arr


def label_centroids_and_areas(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.bincount(labels.ravel())[1:]
    yy, xx = np.indices(labels.shape)
    sum_x = np.bincount(labels.ravel(), weights=xx.ravel())[1:]
    sum_y = np.bincount(labels.ravel(), weights=yy.ravel())[1:]
    keep = counts > 0
    label_ids = np.nonzero(keep)[0] + 1
    centroids = np.column_stack([sum_x[keep] / counts[keep], sum_y[keep] / counts[keep]])
    areas_um2 = counts[keep] * PIXEL_SIZE_UM * PIXEL_SIZE_UM
    return label_ids, centroids, areas_um2


def labels_under_points(labels: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    rounded = np.rint(points_xy).astype(int)
    values = np.zeros(len(rounded), dtype=int)
    in_bounds = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < labels.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < labels.shape[0])
    )
    values[in_bounds] = labels[rounded[in_bounds, 1], rounded[in_bounds, 0]]
    return values


def select_quantile_matches(matches: pd.DataFrame, n: int) -> pd.DataFrame:
    quantiles = np.linspace(0.05, 0.95, n)
    selected = []
    used: set[int] = set()
    for q in quantiles:
        target = matches["area_ratio"].quantile(q)
        candidates = matches.assign(distance_to_target=(matches["area_ratio"] - target).abs())
        for idx in candidates.sort_values("distance_to_target").index:
            if int(idx) not in used:
                selected.append(matches.loc[idx])
                used.add(int(idx))
                break
    return pd.DataFrame(selected).sort_values("area_ratio")


def make_panel(
    *,
    image: np.ndarray,
    labels: np.ndarray,
    rois: list[roifile.ImagejRoi],
    matches: pd.DataFrame,
    output_path: Path,
    label_offset_xy: np.ndarray,
    crop_size: int,
) -> None:
    n = len(matches)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, (_, row) in zip(axes, matches.iterrows(), strict=False):
        roi = rois[int(row["roi_index"])]
        roi_coords = roi.coordinates().astype(float)
        cx, cy = roi_coords.mean(axis=0)
        half = crop_size // 2
        x0 = max(int(round(cx)) - half, 0)
        y0 = max(int(round(cy)) - half, 0)
        x1 = min(x0 + crop_size, image.shape[1])
        y1 = min(y0 + crop_size, image.shape[0])
        x0 = max(x1 - crop_size, 0)
        y0 = max(y1 - crop_size, 0)

        ax.imshow(image[y0:y1, x0:x1])

        roi_plot = roi_coords - np.array([x0, y0])
        ax.plot(
            np.r_[roi_plot[:, 0], roi_plot[0, 0]],
            np.r_[roi_plot[:, 1], roi_plot[0, 1]],
            color="#00e5ff",
            linewidth=2.0,
            label="MyoSight ROI",
        )

        label_id = int(row["label_id"])
        mask = labels == label_id
        for contour in find_contours(mask.astype(np.uint8), 0.5):
            contour_xy = np.column_stack([contour[:, 1], contour[:, 0]])
            contour_xy = contour_xy - label_offset_xy - np.array([x0, y0])
            in_view = (
                (contour_xy[:, 0] >= -5)
                & (contour_xy[:, 0] <= crop_size + 5)
                & (contour_xy[:, 1] >= -5)
                & (contour_xy[:, 1] <= crop_size + 5)
            )
            if in_view.any():
                ax.plot(
                    contour_xy[:, 0],
                    contour_xy[:, 1],
                    color="#ff3b30",
                    linewidth=2.0,
                    label="Pipeline label",
                )

        ax.set_title(
            f"ROI {int(row['roi_index'])} / label {label_id}\n"
            f"area ratio {row['area_ratio']:.2f}x, centroid {row['centroid_distance_px']:.1f}px",
            fontsize=10,
        )
        ax.set_axis_off()

    for ax in axes[n:]:
        ax.set_axis_off()

    handles, labels_text = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels_text[:2], loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overlay MyoSight ROI outlines and pipeline label outlines for matched fibers."
    )
    parser.add_argument("--myosight-results-dir", type=Path, required=True)
    parser.add_argument("--myosight-image", type=Path, required=True)
    parser.add_argument(
        "--background-channel",
        type=int,
        default=None,
        help="Display one raw channel as grayscale instead of an RGB/channel composite.",
    )
    parser.add_argument("--pipeline-labels", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation/roi_boundary_overlays"),
    )
    parser.add_argument("--name", type=str, default="roi_boundary_overlay")
    parser.add_argument("--n", type=int, default=9)
    parser.add_argument("--max-centroid-distance", type=float, default=25.0)
    parser.add_argument("--crop-size", type=int, default=220)
    parser.add_argument(
        "--label-offset-x",
        type=float,
        default=None,
        help="Override label-to-ROI x offset. label ROI-coordinate = label x - offset.",
    )
    parser.add_argument(
        "--label-offset-y",
        type=float,
        default=None,
        help="Override label-to-ROI y offset. label ROI-coordinate = label y - offset.",
    )
    parser.add_argument("--erosion-sweep-max", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    image = load_display_image(args.myosight_image, args.background_channel)
    labels = np.asarray(tifffile.imread(args.pipeline_labels))
    rois = roifile.roiread(args.myosight_results_dir / "ROISet.zip")
    results = pd.read_csv(args.myosight_results_dir / "Results.txt", sep="\t")
    results = results.loc[:, [c for c in results.columns if str(c).strip()]]

    roi_coords = [roi.coordinates().astype(float) for roi in rois]
    roi_centroids = np.array([coords.mean(axis=0) for coords in roi_coords])
    roi_areas_um2 = np.array(
        [polygon_area(coords) * PIXEL_SIZE_UM * PIXEL_SIZE_UM for coords in roi_coords]
    )

    label_ids, label_centroids, label_areas_um2 = label_centroids_and_areas(labels)

    all_roi_coords = np.vstack(roi_coords)
    label_nonzero_yx = np.column_stack(np.nonzero(labels))
    label_nonzero_xy = label_nonzero_yx[:, ::-1]
    label_offset_xy = label_nonzero_xy.min(axis=0) - all_roi_coords.min(axis=0)
    if args.label_offset_x is not None or args.label_offset_y is not None:
        label_offset_xy = np.array(
            [
                0.0 if args.label_offset_x is None else args.label_offset_x,
                0.0 if args.label_offset_y is None else args.label_offset_y,
            ]
        )

    centroid_label_ids = labels_under_points(labels, roi_centroids + label_offset_xy)
    label_centroids_by_id = dict(zip(label_ids, label_centroids, strict=True))
    label_areas_by_id = dict(zip(label_ids, label_areas_um2, strict=True))
    matched = centroid_label_ids > 0
    centroid_distances = np.full(len(rois), np.nan)
    pipeline_areas_um2 = np.full(len(rois), np.nan)
    for i, label_id in enumerate(centroid_label_ids):
        if label_id <= 0:
            continue
        adjusted_centroid = label_centroids_by_id[label_id] - label_offset_xy
        centroid_distances[i] = np.linalg.norm(adjusted_centroid - roi_centroids[i])
        pipeline_areas_um2[i] = label_areas_by_id[label_id]
    matches = pd.DataFrame(
        {
            "roi_index": np.arange(len(rois)),
            "label_id": centroid_label_ids,
            "centroid_distance_px": centroid_distances,
            "myosight_area_um2": roi_areas_um2,
            "myosight_table_area_um2": results["Area"].to_numpy(float),
            "pipeline_area_um2": pipeline_areas_um2,
        }
    )
    matches["area_ratio"] = matches["pipeline_area_um2"] / matches["myosight_area_um2"]
    matches = matches[
        matched & (matches["centroid_distance_px"] <= args.max_centroid_distance)
    ].copy()

    selected = select_quantile_matches(matches, args.n)
    sweep_rows = []
    for erode_px in range(max(0, int(args.erosion_sweep_max)) + 1):
        csa_labels = labels if erode_px == 0 else erode_labels(labels, erode_px)
        csa_counts = np.bincount(csa_labels.ravel(), minlength=int(label_ids.max()) + 1)
        csa_areas_um2 = csa_counts[matches["label_id"].to_numpy(dtype=int)] * (
            PIXEL_SIZE_UM * PIXEL_SIZE_UM
        )
        ratios = csa_areas_um2 / matches["myosight_area_um2"].to_numpy(dtype=float)
        sweep_rows.append(
            {
                "erode_px": erode_px,
                "n_matched": int(len(ratios)),
                "median_ratio": float(np.median(ratios)),
                "p25_ratio": float(np.quantile(ratios, 0.25)),
                "p75_ratio": float(np.quantile(ratios, 0.75)),
                "mean_ratio": float(np.mean(ratios)),
                "median_area_um2": float(np.median(csa_areas_um2)),
                "median_abs_log_ratio": float(np.median(np.abs(np.log(ratios)))),
            }
        )
    sweep = pd.DataFrame(sweep_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matches_path = args.output_dir / f"{args.name}_matched_fibers.csv"
    selected_path = args.output_dir / f"{args.name}_selected_fibers.csv"
    sweep_path = args.output_dir / f"{args.name}_erosion_sweep.csv"
    panel_path = args.output_dir / f"{args.name}_panel.png"
    matches.to_csv(matches_path, index=False)
    selected.to_csv(selected_path, index=False)
    sweep.to_csv(sweep_path, index=False)
    make_panel(
        image=image,
        labels=labels,
        rois=rois,
        matches=selected,
        output_path=panel_path,
        label_offset_xy=label_offset_xy,
        crop_size=args.crop_size,
    )

    print(f"matched fibers: {len(matches)} / {len(rois)}")
    print(f"label offset x,y: {label_offset_xy.tolist()}")
    print(
        "area ratio pipeline/myosight median p25-p75: "
        f"{matches['area_ratio'].median():.3f} "
        f"{matches['area_ratio'].quantile(0.25):.3f}-"
        f"{matches['area_ratio'].quantile(0.75):.3f}"
    )
    print(f"saved panel: {panel_path}")
    print(f"saved selected fibers: {selected_path}")
    print(f"saved all matches: {matches_path}")
    print(f"saved erosion sweep: {sweep_path}")
    print("erosion sweep:")
    print(sweep.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
