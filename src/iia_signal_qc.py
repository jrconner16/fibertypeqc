from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import distance_transform_edt, find_objects

from src.io_utils import ensure_dir, load_multichannel_image, save_dataframe
from src.label_masks import erode_labels


@dataclass
class IIaQCConfig:
    iib_channel: int = 0
    iia_channel: int = 1
    type_i_channel: int | None = None
    erosion_px: int = 2
    ring_inner_px: int = 3
    ring_outer_px: int = 12
    min_bg_pixels: int = 20
    fallback_ring_outer_px: int = 40
    fallback_local_px: int = 80
    fallback_local_quantile: float = 10.0
    weak_snr: float = 1.5
    clear_snr: float = 3.0
    clear_iib_p75_snr: float = 3.0
    clear_iib_p90_snr: float = 5.0
    weak_iib_p75_snr: float = 1.5
    weak_iib_p90_snr: float = 3.0
    clear_margin: float = 1.0
    weak_margin: float = 0.5
    min_interior_area: int = 20
    noise_floor: float = 1.0


def robust_mad_noise(values: np.ndarray, noise_floor: float = 1e-6) -> float:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")

    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    noise = 1.4826 * mad
    if not np.isfinite(noise) or noise <= noise_floor:
        noise = float(np.std(values))
    return max(float(noise), float(noise_floor))


def _safe_percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def _channel_roles(cfg: IIaQCConfig) -> dict[str, int]:
    roles = {"iib": int(cfg.iib_channel), "iia": int(cfg.iia_channel)}
    if cfg.type_i_channel is not None:
        roles["type_i"] = int(cfg.type_i_channel)
    return roles


def _expanded_slice(
    slc: tuple[slice, slice],
    shape: tuple[int, int],
    pad: int,
) -> tuple[slice, slice]:
    y0 = max(0, int(slc[0].start) - pad)
    y1 = min(shape[0], int(slc[0].stop) + pad)
    x0 = max(0, int(slc[1].start) - pad)
    x1 = min(shape[1], int(slc[1].stop) + pad)
    return slice(y0, y1), slice(x0, x1)


def _quantiles_by_label(
    values: np.ndarray,
    groups: np.ndarray,
    label_ids: np.ndarray,
    quantiles: tuple[float, ...],
) -> dict[float, np.ndarray]:
    out = {
        q: np.full(len(label_ids), np.nan, dtype=np.float32)
        for q in quantiles
    }
    values = np.asarray(values, dtype=np.float32)
    groups = np.asarray(groups, dtype=np.int32)
    valid = groups > 0
    if not np.any(valid):
        return out

    values = values[valid]
    groups = groups[valid]
    order = np.argsort(groups, kind="mergesort")
    values = values[order]
    groups = groups[order]
    unique, starts = np.unique(groups, return_index=True)
    ends = np.r_[starts[1:], len(groups)]
    label_to_pos = {int(label_id): i for i, label_id in enumerate(label_ids)}

    for label_id, start, end in zip(unique, starts, ends, strict=True):
        pos = label_to_pos.get(int(label_id))
        if pos is None:
            continue
        group_values = values[start:end]
        for q in quantiles:
            out[q][pos] = _safe_percentile(group_values, q)
    return out


def _noise_by_label(
    values: np.ndarray,
    groups: np.ndarray,
    label_ids: np.ndarray,
    noise_floor: float,
) -> np.ndarray:
    out = np.full(len(label_ids), np.nan, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    groups = np.asarray(groups, dtype=np.int32)
    valid = groups > 0
    if not np.any(valid):
        return out

    values = values[valid]
    groups = groups[valid]
    order = np.argsort(groups, kind="mergesort")
    values = values[order]
    groups = groups[order]
    unique, starts = np.unique(groups, return_index=True)
    ends = np.r_[starts[1:], len(groups)]
    label_to_pos = {int(label_id): i for i, label_id in enumerate(label_ids)}

    for label_id, start, end in zip(unique, starts, ends, strict=True):
        pos = label_to_pos.get(int(label_id))
        if pos is not None:
            out[pos] = robust_mad_noise(values[start:end], noise_floor)
    return out


def _local_low_background_by_label(
    channel_image: np.ndarray,
    labels: np.ndarray,
    label_ids: np.ndarray,
    needs_fallback: np.ndarray,
    cfg: IIaQCConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bg_median = np.full(len(label_ids), np.nan, dtype=np.float32)
    bg_noise = np.full(len(label_ids), np.nan, dtype=np.float32)
    bg_area = np.zeros(len(label_ids), dtype=np.int32)
    object_slices = find_objects(labels)

    for pos, label_id in enumerate(label_ids):
        if not bool(needs_fallback[pos]):
            continue
        lid = int(label_id)
        slc = object_slices[lid - 1] if lid <= len(object_slices) else None
        if slc is None:
            continue
        expanded = _expanded_slice(slc, labels.shape, int(cfg.fallback_local_px))
        local_labels = labels[expanded]
        candidates = np.asarray(channel_image[expanded], dtype=np.float32)[local_labels != lid]
        candidates = candidates[np.isfinite(candidates)]
        if candidates.size == 0:
            continue

        q = float(np.clip(cfg.fallback_local_quantile, 0.0, 50.0))
        cutoff = float(np.percentile(candidates, max(q * 2.0, 25.0)))
        low_pixels = candidates[candidates <= cutoff]
        if low_pixels.size == 0:
            low_pixels = candidates
        bg_median[pos] = float(np.percentile(candidates, q))
        bg_noise[pos] = robust_mad_noise(low_pixels, cfg.noise_floor)
        bg_area[pos] = int(low_pixels.size)

    return bg_median, bg_noise, bg_area


def extract_intensity_features(
    image_chw: np.ndarray,
    labels: np.ndarray,
    cfg: IIaQCConfig,
) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=np.int32)
    image_chw = np.asarray(image_chw)
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D labels, got shape {labels.shape}")
    if image_chw.ndim != 3:
        raise ValueError(f"Expected CHW image, got shape {image_chw.shape}")
    if image_chw.shape[1:] != labels.shape:
        raise ValueError(f"Image shape {image_chw.shape[1:]} does not match labels {labels.shape}")

    roles = _channel_roles(cfg)
    for role, channel in roles.items():
        if channel < 0 or channel >= image_chw.shape[0]:
            raise ValueError(
                f"{role} channel {channel} outside image channel count {image_chw.shape[0]}"
            )

    label_ids = np.unique(labels)
    label_ids = label_ids[label_ids > 0]
    eroded = erode_labels(labels, int(cfg.erosion_px))
    background = labels == 0
    distances, nearest_indices = distance_transform_edt(background, return_indices=True)
    nearest_labels = labels[tuple(nearest_indices)]
    ring_mask = (
        background
        & (distances > float(cfg.ring_inner_px))
        & (distances <= float(cfg.ring_outer_px))
        & (nearest_labels > 0)
    )
    ring_labels = nearest_labels[ring_mask].astype(np.int32)
    ring_counts = np.bincount(ring_labels, minlength=int(label_ids.max()) + 1)[label_ids]
    wide_ring_mask = (
        background
        & (distances > float(cfg.ring_inner_px))
        & (distances <= float(max(cfg.fallback_ring_outer_px, cfg.ring_outer_px)))
        & (nearest_labels > 0)
    )
    wide_ring_labels = nearest_labels[wide_ring_mask].astype(np.int32)
    wide_ring_counts = np.bincount(
        wide_ring_labels,
        minlength=int(label_ids.max()) + 1,
    )[label_ids]
    interior_counts = np.bincount(eroded.ravel(), minlength=int(label_ids.max()) + 1)[label_ids]
    use_wide = (ring_counts < int(cfg.min_bg_pixels)) & (
        wide_ring_counts >= int(cfg.min_bg_pixels)
    )
    use_local = wide_ring_counts < int(cfg.min_bg_pixels)
    bg_source = np.full(len(label_ids), "primary_ring", dtype=object)
    bg_source[use_wide] = "wide_ring"
    bg_source[use_local] = "local_low_percentile"
    selected_bg_area = ring_counts.astype(np.int32, copy=True)
    selected_bg_area[use_wide] = wide_ring_counts[use_wide]

    df = pd.DataFrame(
        {
            "label": label_ids.astype(np.int32),
            "typing_qc_interior_area": interior_counts.astype(np.int32),
            "typing_qc_bg_ring_area": ring_counts.astype(np.int32),
            "typing_qc_wide_bg_ring_area": wide_ring_counts.astype(np.int32),
            "typing_qc_bg_area": selected_bg_area.astype(np.int32),
            "typing_qc_bg_source": bg_source,
            "typing_qc_erosion_px": int(cfg.erosion_px),
            "typing_qc_ring_inner_px": int(cfg.ring_inner_px),
            "typing_qc_ring_outer_px": int(cfg.ring_outer_px),
            "typing_qc_fallback_ring_outer_px": int(cfg.fallback_ring_outer_px),
        }
    )
    if df.empty:
        return df

    interior_labels = eroded.astype(np.int32)
    for role, channel in roles.items():
        channel_image = np.asarray(image_chw[channel], dtype=np.float32)
        fiber_stats = _quantiles_by_label(
            channel_image.ravel(),
            interior_labels.ravel(),
            label_ids,
            (50, 75, 90),
        )
        bg_values = channel_image[ring_mask]
        bg_stats = _quantiles_by_label(bg_values, ring_labels, label_ids, (50,))
        bg_noise = _noise_by_label(bg_values, ring_labels, label_ids, cfg.noise_floor)
        wide_bg_values = channel_image[wide_ring_mask]
        wide_bg_stats = _quantiles_by_label(
            wide_bg_values,
            wide_ring_labels,
            label_ids,
            (50,),
        )
        wide_bg_noise = _noise_by_label(
            wide_bg_values,
            wide_ring_labels,
            label_ids,
            cfg.noise_floor,
        )
        local_bg_median, local_bg_noise, local_bg_area = _local_low_background_by_label(
            channel_image,
            labels,
            label_ids,
            use_local,
            cfg,
        )

        raw_median = fiber_stats[50]
        raw_p75 = fiber_stats[75]
        raw_p90 = fiber_stats[90]
        bg_median = bg_stats[50]
        bg_median = bg_median.copy()
        bg_noise = bg_noise.copy()
        bg_median[use_wide] = wide_bg_stats[50][use_wide]
        bg_noise[use_wide] = wide_bg_noise[use_wide]
        bg_median[use_local] = local_bg_median[use_local]
        bg_noise[use_local] = local_bg_noise[use_local]
        df.loc[use_local, "typing_qc_bg_area"] = local_bg_area[use_local]

        df[f"{role}_raw_median"] = raw_median
        df[f"{role}_raw_p75"] = raw_p75
        df[f"{role}_raw_p90"] = raw_p90
        df[f"{role}_bg_median"] = bg_median
        df[f"{role}_bg_noise"] = bg_noise
        df[f"{role}_bg_sub_median"] = raw_median - bg_median
        df[f"{role}_bg_sub_p75"] = raw_p75 - bg_median
        df[f"{role}_bg_sub_p90"] = raw_p90 - bg_median
        df[f"{role}_snr_median"] = (raw_median - bg_median) / bg_noise
        df[f"{role}_snr_p75"] = (raw_p75 - bg_median) / bg_noise
        df[f"{role}_snr_p90"] = (raw_p90 - bg_median) / bg_noise

    for role in roles:
        values = df[f"{role}_bg_sub_median"].to_numpy(dtype=np.float32)
        finite = np.isfinite(values)
        if np.any(finite):
            center = float(np.median(values[finite]))
            scale = robust_mad_noise(values[finite], cfg.noise_floor)
            ranks = pd.Series(values).rank(pct=True, na_option="keep")
            df[f"{role}_percentile_rank"] = ranks.to_numpy(dtype=np.float32)
            df[f"{role}_robust_z"] = ((values - center) / scale).astype(np.float32)
        else:
            df[f"{role}_percentile_rank"] = np.nan
            df[f"{role}_robust_z"] = np.nan

    return df


def compute_type_scores(df: pd.DataFrame, cfg: IIaQCConfig) -> pd.DataFrame:
    out = df.copy()
    roles = list(_channel_roles(cfg))

    for role in roles:
        snr_median = out[f"{role}_snr_median"].to_numpy(dtype=np.float32)
        snr_p75 = out[f"{role}_snr_p75"].to_numpy(dtype=np.float32)
        out[f"{role}_type_score"] = np.maximum(snr_median, snr_p75)

    score_matrix = out[[f"{role}_type_score" for role in roles]].to_numpy(dtype=np.float32)
    score_matrix = np.where(np.isfinite(score_matrix), score_matrix, -np.inf)
    order = np.argsort(score_matrix, axis=1)
    top_idx = order[:, -1]
    second_idx = order[:, -2] if len(roles) > 1 else top_idx
    top_scores = score_matrix[np.arange(len(out)), top_idx]
    second_scores = score_matrix[np.arange(len(out)), second_idx]
    finite_top = np.isfinite(top_scores)
    finite_second = np.isfinite(second_scores)
    finite_pair = finite_top & finite_second
    type_margin = np.full(len(out), np.nan, dtype=np.float32)
    type_margin[finite_pair] = top_scores[finite_pair] - second_scores[finite_pair]

    out["top_type_channel"] = [roles[i] for i in top_idx]
    out["second_type_channel"] = [roles[i] for i in second_idx]
    out["top_type_score"] = np.where(finite_top, top_scores, np.nan)
    out["second_type_score"] = np.where(finite_second, second_scores, np.nan)
    out["type_margin"] = type_margin
    return out


def assign_conservative_type(df: pd.DataFrame, cfg: IIaQCConfig) -> pd.DataFrame:
    out = compute_type_scores(df, cfg).copy()
    roles = list(_channel_roles(cfg))

    small_interior = out["typing_qc_interior_area"] < int(cfg.min_interior_area)
    empty_bg = out["typing_qc_bg_area"] < int(cfg.min_bg_pixels)
    qc_fail = small_interior | empty_bg

    all_low = np.ones(len(out), dtype=bool)
    for role in roles:
        all_low &= out[f"{role}_type_score"].to_numpy(dtype=np.float32) < float(cfg.weak_snr)

    iia_snr = out["iia_snr_median"].to_numpy(dtype=np.float32)
    iia_bg_sub = out["iia_bg_sub_median"].to_numpy(dtype=np.float32)
    iia_top = out["top_type_channel"].to_numpy(dtype=str) == "iia"
    margin = out["type_margin"].to_numpy(dtype=np.float32)

    iib_snr = out["iib_snr_median"].to_numpy(dtype=np.float32)
    iib_bg_sub = out["iib_bg_sub_median"].to_numpy(dtype=np.float32)
    iib_top = out["top_type_channel"].to_numpy(dtype=str) == "iib"

    clear_iia = (
        ~qc_fail
        & iia_top
        & (iia_snr >= float(cfg.clear_snr))
        & (iia_bg_sub > 0)
        & (margin >= float(cfg.clear_margin))
    )
    iib_p75 = out["iib_snr_p75"].to_numpy(dtype=np.float32)
    iib_p90 = out["iib_snr_p90"].to_numpy(dtype=np.float32)
    clear_iib_signal = (
        (iib_snr >= float(cfg.clear_snr))
        | (iib_p75 >= float(cfg.clear_iib_p75_snr))
        | (iib_p90 >= float(cfg.clear_iib_p90_snr))
    )
    weak_iib_signal = (
        (iib_snr >= float(cfg.weak_snr))
        | (iib_p75 >= float(cfg.weak_iib_p75_snr))
        | (iib_p90 >= float(cfg.weak_iib_p90_snr))
    )

    clear_iib = (
        ~qc_fail
        & iib_top
        & clear_iib_signal
        & (iib_bg_sub > 0)
        & (margin >= float(cfg.clear_margin))
    )
    weak_iia = (
        ~qc_fail
        & ~clear_iia
        & (
            ((iia_snr >= float(cfg.weak_snr)) & (iia_snr < float(cfg.clear_snr)))
            | ((iia_bg_sub > 0) & iia_top & (margin < float(cfg.clear_margin)))
            | ((out["iia_robust_z"].to_numpy(dtype=np.float32) > 1.0) & (iia_snr > 0))
        )
    )
    weak_iib = ~qc_fail & ~clear_iib & iib_top & weak_iib_signal & (iib_bg_sub > 0)
    inferred_iix = ~qc_fail & all_low & ~weak_iia & ~weak_iib
    uncertain = ~(clear_iia | clear_iib | weak_iia | weak_iib | inferred_iix) | qc_fail

    conservative = np.full(len(out), "uncertain", dtype=object)
    conservative[inferred_iix] = "inferred_IIx"
    conservative[weak_iib] = "weak_IIb"
    conservative[weak_iia] = "weak_IIa_or_possible_hybrid"
    conservative[clear_iib] = "clear_IIb"
    conservative[clear_iia] = "clear_IIa"

    liberal = conservative.copy()
    liberal[weak_iia | clear_iia] = "IIa_like"
    liberal[weak_iib | clear_iib] = "IIb_like"
    liberal[inferred_iix] = "inferred_IIx"
    liberal[uncertain] = "uncertain"

    out["predicted_type_conservative"] = conservative
    out["predicted_type_liberal"] = liberal
    out["predicted_type_final"] = conservative
    out["conservative_type_call"] = conservative
    out["liberal_type_call"] = liberal
    out["final_type_call"] = conservative
    out["is_clear_iia"] = clear_iia
    out["is_weak_iia"] = weak_iia
    out["is_clear_iib"] = clear_iib
    out["is_weak_iib"] = weak_iib
    out["is_inferred_iix"] = inferred_iix
    out["is_uncertain"] = uncertain

    noisy_iia = out["iia_bg_noise"] > out["iia_bg_noise"].quantile(0.95)
    flags = []
    for idx in out.index:
        row_flags = []
        if bool(small_interior.loc[idx]):
            row_flags.append("small_interior")
        if bool(empty_bg.loc[idx]):
            row_flags.append("insufficient_background")
        if bool(noisy_iia.loc[idx]):
            row_flags.append("high_iia_background_noise")
        if bool(weak_iia[out.index.get_loc(idx)]):
            row_flags.append("weak_iia")
        if bool(weak_iib[out.index.get_loc(idx)]):
            row_flags.append("weak_iib")
        if bool(uncertain[out.index.get_loc(idx)]):
            row_flags.append("uncertain")
        flags.append("|".join(row_flags))
    out["qc_flags"] = flags
    return out


def merge_with_fibers(fibers: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    if "label" not in fibers.columns:
        raise ValueError("Fiber CSV must contain a 'label' column")
    overlap = [c for c in qc.columns if c in fibers.columns and c != "label"]
    if overlap:
        qc = qc.rename(columns={c: f"typing_qc_{c}" for c in overlap})
    return fibers.merge(qc, on="label", how="left")


def conservative_liberal_summary(df: pd.DataFrame, image_id: str) -> pd.DataFrame:
    total = max(len(df), 1)
    categories = [
        "clear_IIa",
        "weak_IIa_or_possible_hybrid",
        "inferred_IIx",
        "clear_IIb",
        "weak_IIb",
        "uncertain",
    ]
    rows = []
    for category in categories:
        n = int((df["predicted_type_conservative"] == category).sum())
        rows.append(
            {
                "image_id": image_id,
                "category": category,
                "n_fibers": n,
                "percent_fibers": n / total,
            }
        )
    rows.append(
        {
            "image_id": image_id,
            "category": "IIa_like_liberal",
            "n_fibers": int((df["predicted_type_liberal"] == "IIa_like").sum()),
            "percent_fibers": float((df["predicted_type_liberal"] == "IIa_like").mean()),
        }
    )
    rows.append(
        {
            "image_id": image_id,
            "category": "IIb_like_liberal",
            "n_fibers": int((df["predicted_type_liberal"] == "IIb_like").sum()),
            "percent_fibers": float((df["predicted_type_liberal"] == "IIb_like").mean()),
        }
    )
    return pd.DataFrame(rows)


def image_level_metrics(df: pd.DataFrame, image_id: str) -> pd.DataFrame:
    conservative = df["predicted_type_conservative"]
    liberal = df["predicted_type_liberal"]
    return pd.DataFrame(
        [
            {
                "image_id": image_id,
                "n_fibers": int(len(df)),
                "percent_clear_iia": float((conservative == "clear_IIa").mean()),
                "percent_weak_iia": float(
                    (conservative == "weak_IIa_or_possible_hybrid").mean()
                ),
                "percent_iia_like_liberal": float((liberal == "IIa_like").mean()),
                "percent_inferred_iix": float((conservative == "inferred_IIx").mean()),
                "percent_clear_iib": float((conservative == "clear_IIb").mean()),
                "percent_weak_iib": float((conservative == "weak_IIb").mean()),
                "percent_iib_like_liberal": float((liberal == "IIb_like").mean()),
                "percent_uncertain": float((conservative == "uncertain").mean()),
                "median_iia_bg_median": float(df["iia_bg_median"].median()),
                "median_iia_bg_noise": float(df["iia_bg_noise"].median()),
                "mean_iia_bg_median": float(df["iia_bg_median"].mean()),
                "mean_iia_bg_noise": float(df["iia_bg_noise"].mean()),
            }
        ]
    )


def export_random_audit_set(
    df: pd.DataFrame,
    output_dir: Path,
    n_per_class: int = 50,
    seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    samples = []
    for category in [
        "clear_IIa",
        "weak_IIa_or_possible_hybrid",
        "inferred_IIx",
        "clear_IIb",
        "weak_IIb",
        "uncertain",
    ]:
        subset = df[df["predicted_type_conservative"] == category]
        if subset.empty:
            continue
        take = min(int(n_per_class), len(subset))
        choices = rng.choice(subset.index.to_numpy(), size=take, replace=False)
        samples.append(subset.loc[choices])

    audit = pd.concat(samples, ignore_index=True) if samples else df.head(0).copy()
    keep = [
        c
        for c in [
            "image_id",
            "label",
            "fiber_id",
            "predicted_type_conservative",
            "predicted_type_liberal",
            "iia_raw_median",
            "iia_bg_median",
            "iia_bg_noise",
            "iia_bg_sub_median",
            "iia_snr_median",
            "iia_bg_sub_p75",
            "iia_snr_p75",
            "iia_percentile_rank",
            "iia_robust_z",
            "top_type_channel",
            "second_type_channel",
            "type_margin",
            "centroid_x",
            "centroid_y",
            "qc_flags",
        ]
        if c in audit.columns
    ]
    path = output_dir / "random_audit_fibers.csv"
    audit[keep].to_csv(path, index=False)
    return path


def generate_iia_qc_plots(df: pd.DataFrame, output_dir: Path, cfg: IIaQCConfig) -> list[Path]:
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    categories = [
        "clear_IIa",
        "weak_IIa_or_possible_hybrid",
        "inferred_IIx",
        "clear_IIb",
        "weak_IIb",
        "uncertain",
    ]

    def hist_by_category(
        column: str,
        title: str,
        path: Path,
        vlines: tuple[tuple[float, str, str], ...] = (),
    ) -> None:
        finite = df[np.isfinite(df[column].to_numpy(dtype=np.float32))]
        if finite.empty:
            return
        values = finite[column].to_numpy(dtype=np.float32)
        lo, hi = np.percentile(values, [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        bins = np.linspace(lo, hi, 60)
        fig, ax = plt.subplots(figsize=(8, 5))
        for category in categories:
            subset = finite.loc[finite["predicted_type_conservative"] == category, column]
            subset = subset.to_numpy(dtype=np.float32)
            subset = subset[np.isfinite(subset)]
            subset = subset[(subset >= lo) & (subset <= hi)]
            if subset.size == 0:
                continue
            ax.hist(subset, bins=bins, density=True, histtype="step", linewidth=1.4, label=category)
        for x, color, label in vlines:
            ax.axvline(x, color=color, linestyle="--", linewidth=1, label=label)
        ax.set_title(title)
        ax.set_xlabel(column)
        ax.set_ylabel("density")
        ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    path = output_dir / "iia_bg_sub_median_distribution.png"
    hist_by_category(
        "iia_bg_sub_median",
        "IIa background-subtracted median by call",
        path,
        ((0, "black", "zero"),),
    )
    if path.exists():
        paths.append(path)

    path = output_dir / "iia_snr_median_distribution.png"
    hist_by_category(
        "iia_snr_median",
        "IIa median SNR by call",
        path,
        (
            (float(cfg.weak_snr), "orange", "weak SNR"),
            (float(cfg.clear_snr), "red", "clear SNR"),
        ),
    )
    if path.exists():
        paths.append(path)

    scatter_df = df[
        np.isfinite(df["iia_bg_noise"].to_numpy(dtype=np.float32))
        & np.isfinite(df["iia_snr_median"].to_numpy(dtype=np.float32))
    ].copy()
    if len(scatter_df) > 5000:
        scatter_df = scatter_df.sample(n=5000, random_state=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    for category in categories:
        subset = scatter_df[scatter_df["predicted_type_conservative"] == category]
        if subset.empty:
            continue
        ax.scatter(
            subset["iia_bg_noise"],
            subset["iia_snr_median"],
            s=8,
            alpha=0.45,
            label=category,
        )
    ax.axhline(cfg.weak_snr, color="orange", linestyle="--", linewidth=1)
    ax.axhline(cfg.clear_snr, color="red", linestyle="--", linewidth=1)
    ax.set_title("IIa SNR vs local background noise")
    ax.set_xlabel("iia_bg_noise")
    ax.set_ylabel("iia_snr_median")
    ax.legend(loc="upper right", fontsize=7)
    path = output_dir / "iia_snr_vs_background_noise.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    return paths


def run_iia_signal_qc(
    image_path: Path,
    labels_path: Path,
    fibers_path: Path,
    output_dir: Path,
    cfg: IIaQCConfig,
    image_id: str | None = None,
    audit_n_per_class: int = 50,
    skip_plots: bool = False,
) -> dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    image_id = image_id or image_path.stem
    image = load_multichannel_image(image_path)
    labels = np.asarray(tifffile.imread(labels_path), dtype=np.int32)
    fibers = pd.read_csv(fibers_path)

    qc = extract_intensity_features(image, labels, cfg)
    qc = assign_conservative_type(qc, cfg)
    qc["image_id"] = image_id
    merged = merge_with_fibers(fibers, qc)
    if "image_id" not in merged.columns:
        merged["image_id"] = image_id

    fibers_out = output_dir / f"{image_id}_iia_signal_qc_fibers.csv"
    summary_out = output_dir / f"{image_id}_iia_signal_qc_summary.csv"
    metrics_out = output_dir / f"{image_id}_iia_signal_qc_image_metrics.csv"
    save_dataframe(fibers_out, merged)
    save_dataframe(summary_out, conservative_liberal_summary(merged, image_id))
    save_dataframe(metrics_out, image_level_metrics(merged, image_id))
    audit_out = export_random_audit_set(merged, output_dir, n_per_class=audit_n_per_class)

    plot_paths: list[Path] = []
    if not skip_plots:
        plot_paths = generate_iia_qc_plots(merged, output_dir, cfg)

    return {
        "fibers": fibers_out,
        "summary": summary_out,
        "image_metrics": metrics_out,
        "audit": audit_out,
        **{f"plot_{i + 1}": path for i, path in enumerate(plot_paths)},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit weak IIa calls using local background and SNR features."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--fibers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-id", type=str, default=None)
    parser.add_argument("--iib-channel", type=int, default=0)
    parser.add_argument("--iia-channel", type=int, default=1)
    parser.add_argument("--type-i-channel", type=int, default=None)
    parser.add_argument("--erosion-px", type=int, default=2)
    parser.add_argument("--ring-inner-px", type=int, default=3)
    parser.add_argument("--ring-outer-px", type=int, default=12)
    parser.add_argument("--min-bg-pixels", type=int, default=20)
    parser.add_argument("--fallback-ring-outer-px", type=int, default=40)
    parser.add_argument("--fallback-local-px", type=int, default=80)
    parser.add_argument("--fallback-local-quantile", type=float, default=10.0)
    parser.add_argument("--weak-snr", type=float, default=1.5)
    parser.add_argument("--clear-snr", type=float, default=3.0)
    parser.add_argument("--clear-iib-p75-snr", type=float, default=3.0)
    parser.add_argument("--clear-iib-p90-snr", type=float, default=5.0)
    parser.add_argument("--weak-iib-p75-snr", type=float, default=1.5)
    parser.add_argument("--weak-iib-p90-snr", type=float, default=3.0)
    parser.add_argument("--clear-margin", type=float, default=1.0)
    parser.add_argument("--weak-margin", type=float, default=0.5)
    parser.add_argument("--min-interior-area", type=int, default=20)
    parser.add_argument(
        "--noise-floor",
        type=float,
        default=1.0,
        help="Minimum local background noise used in SNR denominator.",
    )
    parser.add_argument("--audit-n-per-class", type=int, default=50)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = IIaQCConfig(
        iib_channel=args.iib_channel,
        iia_channel=args.iia_channel,
        type_i_channel=args.type_i_channel,
        erosion_px=args.erosion_px,
        ring_inner_px=args.ring_inner_px,
        ring_outer_px=args.ring_outer_px,
        min_bg_pixels=args.min_bg_pixels,
        fallback_ring_outer_px=args.fallback_ring_outer_px,
        fallback_local_px=args.fallback_local_px,
        fallback_local_quantile=args.fallback_local_quantile,
        weak_snr=args.weak_snr,
        clear_snr=args.clear_snr,
        clear_iib_p75_snr=args.clear_iib_p75_snr,
        clear_iib_p90_snr=args.clear_iib_p90_snr,
        weak_iib_p75_snr=args.weak_iib_p75_snr,
        weak_iib_p90_snr=args.weak_iib_p90_snr,
        clear_margin=args.clear_margin,
        weak_margin=args.weak_margin,
        min_interior_area=args.min_interior_area,
        noise_floor=args.noise_floor,
    )
    outputs = run_iia_signal_qc(
        image_path=args.image,
        labels_path=args.labels,
        fibers_path=args.fibers,
        output_dir=args.output_dir,
        cfg=cfg,
        image_id=args.image_id,
        audit_n_per_class=args.audit_n_per_class,
        skip_plots=args.skip_plots,
    )
    for name, path in outputs.items():
        print(f"saved {name}: {path}")


if __name__ == "__main__":
    main()
