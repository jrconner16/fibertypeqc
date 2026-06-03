from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import find_objects, gaussian_filter
from scipy.ndimage import mean as ndi_mean
from scipy.ndimage import zoom as ndi_zoom
from skimage.filters import threshold_otsu, threshold_yen

from src.label_masks import erode_labels


@dataclass
class QuantifyConfig:
    type1_channel: int = 0
    type2_channel: int = 1
    i_channel: int | None = None
    iix_channel: int | None = None
    threshold_mode: str = "quantile"  # quantile | otsu | yen | fixed
    quantile: float = 0.6
    percentile_q: float = 0.85
    use_percentile_gate: bool = True
    type1_threshold: float = 0.0
    type2_threshold: float = 0.0
    # raw | global_subtract | tile_subtract | gaussian_subtract
    typing_preprocess: str = "global_subtract"
    typing_bg_quantile: float = 0.02
    typing_tile_size: int = 512
    typing_bg_sigma: float = 24.0
    typing_smooth_sigma: float = 0.8
    typing_erode_px: int = 2
    coverage_quantile: float = 0.85
    min_coverage: float = 0.06
    review_confidence_threshold: float = 0.15
    review_margin: float = 0.05
    model_confidence_threshold: float = 0.70
    model_margin_threshold: float = 0.25
    mixed_balance_tolerance: float = 0.25
    pixel_size_x_um: float | None = None
    pixel_size_y_um: float | None = None
    csa_erode_px: tuple[int, ...] = (1, 2, 3, 4, 5)

    # Optional classifier hook (off by default).
    classifier_path: str | None = None


@dataclass
class QCConfig:
    min_labels: int = 300
    max_unknown_rate: float = 0.35
    median_area_min: float = 200.0
    median_area_max: float = 15000.0
    max_type_corr: float = 0.92


@dataclass(frozen=True)
class MarkerStats:
    mean: np.ndarray
    p75: np.ndarray
    p90: np.ndarray
    pctl: np.ndarray
    coverage: np.ndarray


@dataclass(frozen=True)
class MarkerSpec:
    marker_name: str
    legacy_prefix: str | None
    channel_index: int


def _default_marker_specs(cfg: QuantifyConfig) -> tuple[MarkerSpec, MarkerSpec]:
    return (
        MarkerSpec(marker_name="iib", legacy_prefix="type1", channel_index=int(cfg.type1_channel)),
        MarkerSpec(marker_name="iia", legacy_prefix="type2", channel_index=int(cfg.type2_channel)),
    )


def _active_marker_specs(cfg: QuantifyConfig) -> tuple[MarkerSpec, ...]:
    specs: list[MarkerSpec] = [
        MarkerSpec(marker_name="iib", legacy_prefix="type1", channel_index=int(cfg.type1_channel)),
        MarkerSpec(marker_name="iia", legacy_prefix="type2", channel_index=int(cfg.type2_channel)),
    ]
    if cfg.i_channel is not None:
        specs.append(
            MarkerSpec(marker_name="i", legacy_prefix=None, channel_index=int(cfg.i_channel))
        )
    if cfg.iix_channel is not None:
        specs.append(
            MarkerSpec(marker_name="iix", legacy_prefix=None, channel_index=int(cfg.iix_channel))
        )
    return tuple(specs)


def _auto_threshold(values: np.ndarray, mode: str, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 0.0
    if mode == "quantile":
        return float(np.quantile(values, quantile))
    if mode == "otsu":
        return float(threshold_otsu(values))
    if mode == "yen":
        return float(threshold_yen(values))
    raise ValueError(f"Unsupported auto threshold mode: {mode}")


def _marker_column(spec: MarkerSpec, suffix: str) -> str:
    if spec.legacy_prefix is None:
        raise ValueError(
            f"Marker {spec.marker_name!r} does not have a legacy output-column prefix."
        )
    return f"{spec.legacy_prefix}_{suffix}"


def _build_features(
    df: pd.DataFrame,
    marker_specs: tuple[MarkerSpec, MarkerSpec],
) -> pd.DataFrame:
    eps = 1e-6
    primary_spec, secondary_spec = marker_specs
    primary_prefix = primary_spec.legacy_prefix
    secondary_prefix = secondary_spec.legacy_prefix

    primary_mean = df[_marker_column(primary_spec, "mean")].astype(np.float32)
    secondary_mean = df[_marker_column(secondary_spec, "mean")].astype(np.float32)
    primary_pctl = df[_marker_column(primary_spec, "pctl")].astype(np.float32)
    secondary_pctl = df[_marker_column(secondary_spec, "pctl")].astype(np.float32)
    primary_p75 = df.get(
        _marker_column(primary_spec, "p75"),
        df[_marker_column(primary_spec, "pctl")],
    ).astype(np.float32)
    secondary_p75 = df.get(
        _marker_column(secondary_spec, "p75"),
        df[_marker_column(secondary_spec, "pctl")],
    ).astype(np.float32)
    primary_p90 = df.get(
        _marker_column(primary_spec, "p90"),
        df[_marker_column(primary_spec, "pctl")],
    ).astype(np.float32)
    secondary_p90 = df.get(
        _marker_column(secondary_spec, "p90"),
        df[_marker_column(secondary_spec, "pctl")],
    ).astype(np.float32)
    primary_coverage = df[_marker_column(primary_spec, "coverage")].astype(np.float32)
    secondary_coverage = df[_marker_column(secondary_spec, "coverage")].astype(np.float32)

    out = pd.DataFrame(index=df.index)
    out["area"] = df["area"].astype(np.float32)
    out[f"{primary_prefix}_mean"] = primary_mean
    out[f"{secondary_prefix}_mean"] = secondary_mean
    out[f"{primary_prefix}_p75"] = primary_p75
    out[f"{secondary_prefix}_p75"] = secondary_p75
    out[f"{primary_prefix}_p90"] = primary_p90
    out[f"{secondary_prefix}_p90"] = secondary_p90
    out[f"{primary_prefix}_pctl"] = primary_pctl
    out[f"{secondary_prefix}_pctl"] = secondary_pctl
    out[f"{primary_prefix}_coverage"] = primary_coverage
    out[f"{secondary_prefix}_coverage"] = secondary_coverage
    out["type_ratio"] = primary_mean / (secondary_mean + eps)
    out["type_diff"] = primary_mean - secondary_mean
    out["type_pctl_ratio"] = primary_pctl / (secondary_pctl + eps)
    out["type_pctl_diff"] = primary_pctl - secondary_pctl
    out["type_p75_ratio"] = primary_p75 / (secondary_p75 + eps)
    out["type_p75_diff"] = primary_p75 - secondary_p75
    out["type_p90_ratio"] = primary_p90 / (secondary_p90 + eps)
    out["type_p90_diff"] = primary_p90 - secondary_p90
    out["type_cov_ratio"] = primary_coverage / (secondary_coverage + eps)
    out["type_cov_diff"] = primary_coverage - secondary_coverage
    return out


def _label_percentiles(
    image: np.ndarray,
    labels: np.ndarray,
    label_ids: np.ndarray,
    q: float,
) -> np.ndarray:
    q_pct = float(np.clip(q, 0.0, 1.0) * 100.0)
    out = np.zeros(len(label_ids), dtype=np.float32)
    object_slices = find_objects(labels)
    for i, lid in enumerate(label_ids):
        if lid <= 0 or lid > len(object_slices):
            continue
        slc = object_slices[int(lid) - 1]
        if slc is None:
            continue
        local_labels = labels[slc]
        local_image = image[slc]
        pixels = local_image[local_labels == lid]
        out[i] = float(np.percentile(pixels, q_pct)) if pixels.size else 0.0
    return out


def _preprocess_typing_channel(
    channel: np.ndarray,
    mode: str = "global_subtract",
    bg_quantile: float = 0.02,
    tile_size: int = 512,
    bg_sigma: float = 24.0,
    smooth_sigma: float = 0.8,
) -> np.ndarray:
    x = np.asarray(channel, dtype=np.float32)
    mode = str(mode).strip().lower()
    if mode == "raw":
        pass
    elif mode == "global_subtract":
        q = float(np.clip(bg_quantile, 0.0, 0.5))
        bg = float(np.quantile(x[np.isfinite(x)], q)) if np.any(np.isfinite(x)) else 0.0
        x = np.clip(x - bg, 0.0, None)
    elif mode == "tile_subtract":
        q = float(np.clip(bg_quantile, 0.0, 0.5))
        bg = _tile_quantile_background(x, q=q, tile_size=int(tile_size))
        x = np.clip(x - bg, 0.0, None)
    elif mode == "gaussian_subtract":
        x = x - gaussian_filter(x, sigma=float(bg_sigma))
        x = np.clip(x, 0.0, None)
    else:
        raise ValueError(
            "Unsupported typing preprocessing mode "
            f"{mode!r}; use raw, global_subtract, tile_subtract, or gaussian_subtract"
        )
    if smooth_sigma > 0:
        x = gaussian_filter(x, sigma=float(smooth_sigma))
    return x.astype(np.float32)


def _tile_quantile_background(channel: np.ndarray, q: float, tile_size: int) -> np.ndarray:
    x = np.asarray(channel, dtype=np.float32)
    h, w = x.shape
    tile = max(64, int(tile_size))
    ny = int(np.ceil(h / tile))
    nx = int(np.ceil(w / tile))
    bg_small = np.zeros((ny, nx), dtype=np.float32)

    finite = x[np.isfinite(x)]
    fallback = float(np.quantile(finite, q)) if finite.size else 0.0
    for iy in range(ny):
        y0 = iy * tile
        y1 = min(h, y0 + tile)
        for ix in range(nx):
            x0 = ix * tile
            x1 = min(w, x0 + tile)
            pixels = x[y0:y1, x0:x1]
            pixels = pixels[np.isfinite(pixels)]
            bg_small[iy, ix] = float(np.quantile(pixels, q)) if pixels.size else fallback

    zoom_y = h / max(ny, 1)
    zoom_x = w / max(nx, 1)
    bg = ndi_zoom(bg_small, zoom=(zoom_y, zoom_x), order=1)
    return bg[:h, :w].astype(np.float32)


def _norm_gain(value: np.ndarray, threshold: float) -> np.ndarray:
    denom = max(abs(float(threshold)), 1e-6)
    return np.asarray((value - threshold) / denom, dtype=np.float32)


def _signal_evidence(
    mean: pd.Series,
    p75: pd.Series,
    p90: pd.Series,
    coverage: pd.Series,
    mean_threshold: float,
    p75_threshold: float,
    p90_threshold: float,
    min_coverage: float,
) -> np.ndarray:
    coherent = (
        ((mean >= mean_threshold) | (p75 >= p75_threshold))
        & (coverage >= float(min_coverage))
    )
    p90_only = (p90 >= p90_threshold) & ~coherent
    out = np.full(len(mean), "low", dtype=object)
    out[p90_only.to_numpy()] = "p90_only"
    out[coherent.to_numpy()] = "clear"
    return out


def _class_probability_name(class_name: object) -> str:
    name = str(class_name).strip().lower()
    mapping = {
        "type1": "iib",
        "type2": "iia",
        "unknown": "iix",
        "iix_candidate": "iix",
    }
    name = mapping.get(name, name)
    safe = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
    return f"prob_{safe or 'class'}"


def _canonical_fiber_type_label(label: str) -> str:
    mapping = {
        "type1": "iib",
        "type2": "iia",
        "unknown": "iix",
        "iix_candidate": "iix",
    }
    return mapping.get(str(label).strip().lower(), str(label).strip().lower())


def _legacy_marker_pair_columns(
    marker_specs: tuple[MarkerSpec, MarkerSpec],
    suffix: str,
) -> tuple[str, str]:
    primary_spec, secondary_spec = marker_specs
    return _marker_column(primary_spec, suffix), _marker_column(secondary_spec, suffix)


def _available_markers_value(marker_specs: tuple[MarkerSpec, ...]) -> str:
    return "|".join(spec.marker_name for spec in marker_specs)


def _rule_fiber_type_source(fiber_types: pd.Series) -> np.ndarray:
    out = np.full(len(fiber_types), "direct_marker", dtype=object)
    labels = fiber_types.astype(str).str.lower()
    out[labels.eq("mixed").to_numpy()] = "hybrid_marker"
    out[labels.eq("unknown").to_numpy()] = "residual_inference"
    return out


def _marker_signal_stats(
    *,
    image_chw: np.ndarray,
    labels: np.ndarray,
    label_ids: np.ndarray,
    tissue_mask: np.ndarray,
    cfg: QuantifyConfig,
    spec: MarkerSpec,
) -> MarkerStats:
    channel = _preprocess_typing_channel(
        image_chw[spec.channel_index],
        mode=cfg.typing_preprocess,
        bg_quantile=cfg.typing_bg_quantile,
        tile_size=cfg.typing_tile_size,
        bg_sigma=cfg.typing_bg_sigma,
        smooth_sigma=cfg.typing_smooth_sigma,
    )
    mean = np.asarray(ndi_mean(channel, labels=labels, index=label_ids), dtype=np.float32)
    p75 = _label_percentiles(channel, labels, label_ids, 0.75)
    p90 = _label_percentiles(channel, labels, label_ids, 0.90)
    pctl = _label_percentiles(channel, labels, label_ids, cfg.percentile_q)
    cutoff = (
        float(np.quantile(channel[tissue_mask], cfg.coverage_quantile))
        if np.any(tissue_mask)
        else 0.0
    )
    coverage = np.asarray(
        ndi_mean((channel >= cutoff).astype(np.float32), labels=labels, index=label_ids),
        dtype=np.float32,
    )
    return MarkerStats(
        mean=mean,
        p75=p75,
        p90=p90,
        pctl=pctl,
        coverage=coverage,
    )


def _collect_marker_stats(
    *,
    image_chw: np.ndarray,
    labels: np.ndarray,
    label_ids: np.ndarray,
    tissue_mask: np.ndarray,
    cfg: QuantifyConfig,
    specs: tuple[MarkerSpec, ...],
) -> dict[str, MarkerStats]:
    return {
        spec.marker_name: _marker_signal_stats(
            image_chw=image_chw,
            labels=labels,
            label_ids=label_ids,
            tissue_mask=tissue_mask,
            cfg=cfg,
            spec=spec,
        )
        for spec in specs
    }


def _legacy_typing_feature_columns(
    marker_stats: dict[str, MarkerStats],
    marker_specs: tuple[MarkerSpec, ...],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for spec in marker_specs:
        stats = marker_stats[spec.marker_name]
        prefix = spec.legacy_prefix
        out[f"{prefix}_mean"] = stats.mean
        out[f"{prefix}_p75"] = stats.p75
        out[f"{prefix}_p90"] = stats.p90
        out[f"{prefix}_pctl"] = stats.pctl
        out[f"{prefix}_coverage"] = stats.coverage
    return out


def _marker_stats_metadata(
    marker_stats: dict[str, MarkerStats],
) -> dict[str, dict[str, np.ndarray]]:
    return {
        marker_name: {
            "mean": stats.mean.copy(),
            "p75": stats.p75.copy(),
            "p90": stats.p90.copy(),
            "pctl": stats.pctl.copy(),
            "coverage": stats.coverage.copy(),
        }
        for marker_name, stats in marker_stats.items()
    }


def _legacy_signal_thresholds(
    df: pd.DataFrame,
    cfg: QuantifyConfig,
    marker_specs: tuple[MarkerSpec, MarkerSpec],
) -> dict[str, float]:
    primary_spec, secondary_spec = marker_specs
    return {
        "type1_signal": _auto_threshold(
            df[_marker_column(primary_spec, "mean")].to_numpy(),
            cfg.threshold_mode,
            cfg.quantile,
        ),
        "type2_signal": _auto_threshold(
            df[_marker_column(secondary_spec, "mean")].to_numpy(),
            cfg.threshold_mode,
            cfg.quantile,
        ),
        "type1_p75": _auto_threshold(
            df[_marker_column(primary_spec, "p75")].to_numpy(),
            cfg.threshold_mode,
            cfg.quantile,
        ),
        "type2_p75": _auto_threshold(
            df[_marker_column(secondary_spec, "p75")].to_numpy(),
            cfg.threshold_mode,
            cfg.quantile,
        ),
        "type1_p90": _auto_threshold(
            df[_marker_column(primary_spec, "p90")].to_numpy(),
            cfg.threshold_mode,
            cfg.quantile,
        ),
        "type2_p90": _auto_threshold(
            df[_marker_column(secondary_spec, "p90")].to_numpy(),
            cfg.threshold_mode,
            cfg.quantile,
        ),
    }


def _add_legacy_signal_evidence(
    df: pd.DataFrame,
    cfg: QuantifyConfig,
    thresholds: dict[str, float],
    marker_specs: tuple[MarkerSpec, MarkerSpec],
) -> None:
    primary_spec, secondary_spec = marker_specs
    df["type1_signal_evidence"] = _signal_evidence(
        df[_marker_column(primary_spec, "mean")],
        df[_marker_column(primary_spec, "p75")],
        df[_marker_column(primary_spec, "p90")],
        df[_marker_column(primary_spec, "coverage")],
        thresholds["type1_signal"],
        thresholds["type1_p75"],
        thresholds["type1_p90"],
        cfg.min_coverage,
    )
    df["type2_signal_evidence"] = _signal_evidence(
        df[_marker_column(secondary_spec, "mean")],
        df[_marker_column(secondary_spec, "p75")],
        df[_marker_column(secondary_spec, "p90")],
        df[_marker_column(secondary_spec, "coverage")],
        thresholds["type2_signal"],
        thresholds["type2_p75"],
        thresholds["type2_p90"],
        cfg.min_coverage,
    )
    df["type1_p75_threshold"] = thresholds["type1_p75"]
    df["type2_p75_threshold"] = thresholds["type2_p75"]
    df["type1_p90_threshold"] = thresholds["type1_p90"]
    df["type2_p90_threshold"] = thresholds["type2_p90"]


def _legacy_rule_classification(
    df: pd.DataFrame,
    cfg: QuantifyConfig,
    thresholds: dict[str, float],
    marker_specs: tuple[MarkerSpec, MarkerSpec],
) -> dict[str, np.ndarray | float]:
    primary_spec, secondary_spec = marker_specs
    primary_mean = df[_marker_column(primary_spec, "mean")]
    secondary_mean = df[_marker_column(secondary_spec, "mean")]
    primary_pctl = df[_marker_column(primary_spec, "pctl")]
    secondary_pctl = df[_marker_column(secondary_spec, "pctl")]
    primary_p75 = df[_marker_column(primary_spec, "p75")]
    secondary_p75 = df[_marker_column(secondary_spec, "p75")]
    primary_coverage = df[_marker_column(primary_spec, "coverage")]
    secondary_coverage = df[_marker_column(secondary_spec, "coverage")]

    if cfg.threshold_mode in {"quantile", "otsu", "yen"}:
        t1 = thresholds["type1_signal"]
        t2 = thresholds["type2_signal"]
        t1p = _auto_threshold(primary_pctl.to_numpy(), cfg.threshold_mode, cfg.quantile)
        t2p = _auto_threshold(secondary_pctl.to_numpy(), cfg.threshold_mode, cfg.quantile)
        t1p75 = thresholds["type1_p75"]
        t2p75 = thresholds["type2_p75"]
        t1p90 = thresholds["type1_p90"]
        t2p90 = thresholds["type2_p90"]
    else:
        t1 = float(cfg.type1_threshold)
        t2 = float(cfg.type2_threshold)
        t1p = t1
        t2p = t2
        t1p75 = t1
        t2p75 = t2
        t1p90 = t1
        t2p90 = t2

    has1 = primary_mean >= t1
    has2 = secondary_mean >= t2
    if cfg.use_percentile_gate:
        has1 = has1 | (primary_pctl >= t1p) | (primary_p75 >= t1p75)
        has2 = has2 | (secondary_pctl >= t2p) | (secondary_p75 >= t2p75)
    c1 = primary_coverage >= float(cfg.min_coverage)
    c2 = secondary_coverage >= float(cfg.min_coverage)
    has1 = has1 & c1
    has2 = has2 & c2

    score1 = np.maximum(
        _norm_gain(primary_mean.to_numpy(), t1),
        _norm_gain(primary_pctl.to_numpy(), t1p),
    )
    score2 = np.maximum(
        _norm_gain(secondary_mean.to_numpy(), t2),
        _norm_gain(secondary_pctl.to_numpy(), t2p),
    )
    score1 = score1 + (
        (primary_coverage.to_numpy(dtype=np.float32) / max(cfg.min_coverage, 1e-6))
        - 1.0
    )
    score2 = score2 + (
        (secondary_coverage.to_numpy(dtype=np.float32) / max(cfg.min_coverage, 1e-6))
        - 1.0
    )

    both = has1 & has2
    fiber_type = np.full(len(df), "unknown", dtype=object)
    fiber_type[has1 & ~has2] = "type1"
    fiber_type[~has1 & has2] = "type2"
    if np.any(both):
        diff = np.abs(score1 - score2)
        close = diff <= float(cfg.mixed_balance_tolerance)
        fiber_type[both & close] = "mixed"
        fiber_type[both & ~close & (score1 > score2)] = "type1"
        fiber_type[both & ~close & (score2 > score1)] = "type2"

    confidence = np.abs(score1 - score2) / (np.abs(score1) + np.abs(score2) + 1e-6)
    return {
        "fiber_type": fiber_type,
        "score_type1": score1,
        "score_type2": score2,
        "confidence": confidence.astype(np.float32),
        "type1_threshold": t1,
        "type2_threshold": t2,
        "type1_p75_threshold": t1p75,
        "type2_p75_threshold": t2p75,
        "type1_p90_threshold": t1p90,
        "type2_p90_threshold": t2p90,
        "type1_pctl_threshold": t1p,
        "type2_pctl_threshold": t2p,
        "type1_cov_threshold": float(cfg.min_coverage),
        "type2_cov_threshold": float(cfg.min_coverage),
    }


def _add_model_signal_qc(df: pd.DataFrame, cfg: QuantifyConfig) -> pd.DataFrame:
    out = df.copy()
    pred = out["fiber_type"].astype(str).str.lower()
    type1_pred = pred.isin(["type1", "iib"])
    type2_pred = pred.isin(["type2", "iia"])
    iix_pred = pred.isin(["unknown", "iix", "iix_candidate"])

    type1_evidence = out["type1_signal_evidence"].astype(str)
    type2_evidence = out["type2_signal_evidence"].astype(str)
    weak_positive = (
        (type1_pred & type1_evidence.ne("clear"))
        | (type2_pred & type2_evidence.ne("clear"))
    )
    possible_missed_positive = iix_pred & (
        type1_evidence.eq("clear") | type2_evidence.eq("clear")
    )
    if "model_confidence" in out.columns:
        low_confidence = out["model_confidence"] < float(cfg.model_confidence_threshold)
    else:
        low_confidence = pd.Series(False, index=out.index)
    if "model_margin" in out.columns:
        low_margin = out["model_margin"] < float(cfg.model_margin_threshold)
    else:
        low_margin = pd.Series(False, index=out.index)

    flags = []
    for weak, missed, low_conf, low_marg in zip(
        weak_positive,
        possible_missed_positive,
        low_confidence,
        low_margin,
        strict=True,
    ):
        row_flags = []
        if bool(weak):
            row_flags.append("model_positive_without_clear_signal")
        if bool(missed):
            row_flags.append("model_iix_with_clear_marker_signal")
        if bool(low_conf):
            row_flags.append("low_model_confidence")
        if bool(low_marg):
            row_flags.append("low_model_margin")
        flags.append("|".join(row_flags))

    out["typing_signal_qc_flags"] = flags
    out["has_signal_qc_warning"] = weak_positive | possible_missed_positive
    out["needs_review"] = low_confidence | low_margin
    return out


def apply_auto_profile(
    cfg: QuantifyConfig,
    sensitivity: float = 0.5,
    mixed_strictness: float = 0.7,
) -> QuantifyConfig:
    s = float(np.clip(sensitivity, 0.0, 1.0))
    m = float(np.clip(mixed_strictness, 0.0, 1.0))
    tuned = replace(cfg)

    # Higher sensitivity catches more positives with lower thresholds.
    tuned.quantile = float(np.clip(0.80 - 0.35 * s, 0.35, 0.90))
    tuned.use_percentile_gate = bool(s >= 0.30)
    tuned.typing_bg_sigma = float(np.clip(44.0 - 24.0 * s, 16.0, 64.0))
    tuned.typing_smooth_sigma = float(np.clip(1.2 - 0.6 * s, 0.4, 1.6))
    tuned.coverage_quantile = float(np.clip(0.90 - 0.10 * s, 0.70, 0.95))
    tuned.min_coverage = float(np.clip(0.14 - 0.10 * s, 0.03, 0.18))
    tuned.review_confidence_threshold = float(np.clip(0.25 - 0.18 * s, 0.04, 0.30))
    tuned.review_margin = float(np.clip(0.09 - 0.06 * s, 0.02, 0.12))

    # Higher mixed strictness means far fewer mixed calls.
    tuned.mixed_balance_tolerance = float(np.clip(0.30 - 0.24 * m, 0.05, 0.35))
    return tuned


def _predict_with_classifier(
    df: pd.DataFrame,
    classifier_path: str | None,
    marker_specs: tuple[MarkerSpec, MarkerSpec],
) -> tuple[np.ndarray | None, pd.DataFrame | None, str | None]:
    if not classifier_path:
        return None, None, None

    model_path = Path(classifier_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Classifier not found: {model_path}")

    import joblib

    model = joblib.load(model_path)
    feats = _build_features(df, marker_specs)

    if hasattr(model, "feature_names_in_"):
        cols = [c for c in model.feature_names_in_ if c in feats.columns]
        if not cols:
            raise ValueError("Classifier has feature_names_in_ but none match available features")
        x = feats[cols]
    else:
        x = feats[
            [
                "area",
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
                "type_ratio",
                "type_diff",
                "type_pctl_ratio",
                "type_pctl_diff",
                "type_cov_ratio",
                "type_cov_diff",
            ]
        ]

    pred = model.predict(x)
    proba_df = pd.DataFrame(index=df.index)
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x), dtype=np.float32)
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "steps"):
            classes = getattr(model.steps[-1][1], "classes_", None)
        if classes is None:
            classes = np.arange(proba.shape[1])

        seen_prob_cols: set[str] = set()
        for class_name, values in zip(classes, proba.T, strict=False):
            col = _class_probability_name(class_name)
            if col in seen_prob_cols:
                proba_df[col] = np.fmax(proba_df[col].to_numpy(dtype=np.float32), values)
            else:
                proba_df[col] = values
                seen_prob_cols.add(col)

        for col in ("prob_iib", "prob_iia", "prob_iix"):
            if col not in proba_df.columns:
                proba_df[col] = np.nan

        sorted_proba = np.sort(proba, axis=1)
        proba_df["model_confidence"] = sorted_proba[:, -1].astype(np.float32)
        if proba.shape[1] > 1:
            margin = sorted_proba[:, -1] - sorted_proba[:, -2]
        else:
            margin = sorted_proba[:, -1]
        proba_df["model_margin"] = margin.astype(np.float32)
    else:
        for col in ("prob_iib", "prob_iia", "prob_iix", "model_confidence", "model_margin"):
            proba_df[col] = np.nan

    return np.asarray(pred).astype(str), proba_df, str(model_path)


def quantify_labels(labels: np.ndarray, image_chw: np.ndarray, cfg: QuantifyConfig) -> pd.DataFrame:
    label_ids = np.unique(labels)
    label_ids = label_ids[label_ids > 0]

    if label_ids.size == 0:
        return pd.DataFrame(
            columns=[
                "label",
                "area",
                "type1_mean",
                "type2_mean",
                "type1_pctl",
                "type2_pctl",
                "type1_coverage",
                "type2_coverage",
                "typing_interior_area",
                "typing_erode_px",
                "fiber_type",
                "fiber_type_source",
                "available_markers",
                "classification_method",
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
                "score_type1",
                "score_type2",
                "confidence",
                "model_confidence",
                "model_margin",
                "prob_iib",
                "prob_iia",
                "prob_iix",
                "needs_review",
                "typing_signal_qc_flags",
                "classifier_path",
            ]
        )

    areas = np.bincount(labels.ravel())[label_ids]
    marker_specs = _default_marker_specs(cfg)
    active_marker_specs = _active_marker_specs(cfg)

    measure_labels = erode_labels(labels, int(cfg.typing_erode_px))
    interior_areas = np.bincount(
        measure_labels.ravel(),
        minlength=int(label_ids.max()) + 1,
    )[label_ids]
    tissue = measure_labels > 0
    marker_stats = _collect_marker_stats(
        image_chw=image_chw,
        labels=measure_labels,
        label_ids=label_ids,
        tissue_mask=tissue,
        cfg=cfg,
        specs=active_marker_specs,
    )
    legacy_feature_columns = _legacy_typing_feature_columns(marker_stats, marker_specs)

    df = pd.DataFrame(
        {
            "label": label_ids.astype(np.int32),
            "area": areas.astype(np.int32),
            "typing_interior_area": interior_areas.astype(np.int32),
            "typing_erode_px": int(cfg.typing_erode_px),
            "typing_preprocess": str(cfg.typing_preprocess),
            "typing_bg_quantile": float(cfg.typing_bg_quantile),
            "typing_tile_size": int(cfg.typing_tile_size),
            "typing_bg_sigma": float(cfg.typing_bg_sigma),
            "typing_smooth_sigma": float(cfg.typing_smooth_sigma),
            **legacy_feature_columns,
        }
    )
    available_markers_value = _available_markers_value(active_marker_specs)
    df.attrs["available_markers"] = tuple(spec.marker_name for spec in active_marker_specs)
    df.attrs["marker_stats"] = _marker_stats_metadata(marker_stats)
    df["available_markers"] = available_markers_value

    if cfg.pixel_size_x_um is not None and cfg.pixel_size_y_um is not None:
        pixel_area_um2 = float(cfg.pixel_size_x_um) * float(cfg.pixel_size_y_um)
        df["pixel_size_x_um"] = float(cfg.pixel_size_x_um)
        df["pixel_size_y_um"] = float(cfg.pixel_size_y_um)
        df["area_um2"] = df["area"].astype(np.float32) * pixel_area_um2
        for erode_px in cfg.csa_erode_px:
            erode_px = int(erode_px)
            if erode_px <= 0:
                continue
            csa_labels = erode_labels(labels, erode_px)
            csa_areas = np.bincount(
                csa_labels.ravel(),
                minlength=int(label_ids.max()) + 1,
            )[label_ids]
            df[f"area_erode_{erode_px}px"] = csa_areas.astype(np.int32)
            df[f"area_erode_{erode_px}px_um2"] = (
                csa_areas.astype(np.float32) * pixel_area_um2
            )

    legacy_thresholds = _legacy_signal_thresholds(df, cfg, marker_specs)
    _add_legacy_signal_evidence(df, cfg, legacy_thresholds, marker_specs)

    pred, proba_df, model_path = _predict_with_classifier(df, cfg.classifier_path, marker_specs)
    if pred is not None:
        df["fiber_type"] = pred
        if proba_df is not None:
            for col in proba_df.columns:
                df[col] = proba_df[col].to_numpy()
        df["classification_method"] = "model"
        df["fiber_type_source"] = "model_prediction"
        df["type1_threshold"] = legacy_thresholds["type1_signal"]
        df["type2_threshold"] = legacy_thresholds["type2_signal"]
        df["type1_pctl_threshold"] = np.nan
        df["type2_pctl_threshold"] = np.nan
        df["type1_cov_threshold"] = np.nan
        df["type2_cov_threshold"] = np.nan
        df["score_type1"] = np.nan
        df["score_type2"] = np.nan
        if "model_confidence" not in df.columns:
            df["model_confidence"] = np.nan
        if "model_margin" not in df.columns:
            df["model_margin"] = np.nan
        df["confidence"] = df["model_confidence"]
        df["classifier_path"] = model_path
        df = _add_model_signal_qc(df, cfg)
        return df

    rule_outputs = _legacy_rule_classification(df, cfg, legacy_thresholds, marker_specs)
    df["fiber_type"] = rule_outputs["fiber_type"]
    df["fiber_type_source"] = _rule_fiber_type_source(df["fiber_type"])
    method = f"rules:{cfg.threshold_mode}"
    if cfg.use_percentile_gate:
        method += f"+p{int(round(cfg.percentile_q * 100))}"
    method += f"+cov{int(round(cfg.min_coverage * 100))}"
    method += f"+mx{int(round(cfg.mixed_balance_tolerance * 100))}"
    df["classification_method"] = method
    for col in (
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
        "score_type1",
        "score_type2",
        "confidence",
    ):
        df[col] = rule_outputs[col]
    df["model_confidence"] = np.nan
    df["model_margin"] = np.nan
    df["prob_iib"] = np.nan
    df["prob_iia"] = np.nan
    df["prob_iix"] = np.nan
    df["needs_review"] = (
        (df["fiber_type"] == "unknown")
        | (df["confidence"] < float(cfg.review_confidence_threshold))
        | (np.abs(df["score_type1"] - df["score_type2"]) < float(cfg.review_margin))
    )
    df["typing_signal_qc_flags"] = ""
    df["classifier_path"] = ""
    return df


def class_stats_with_ci(
    fibers: pd.DataFrame,
    classes: tuple[str, ...] = ("type1", "type2", "mixed", "unknown"),
    bootstrap_reps: int = 1000,
    seed: int = 0,
    canonicalize_labels: bool = False,
) -> dict[str, float]:
    out: dict[str, float] = {}
    n = int(len(fibers))
    out["n_fibers"] = float(n)

    if n == 0:
        for c in classes:
            out[f"prop_{c}"] = np.nan
            out[f"ci95_low_{c}"] = np.nan
            out[f"ci95_high_{c}"] = np.nan
        return out

    labels = fibers["fiber_type"].astype(str)
    if canonicalize_labels:
        labels = labels.map(_canonical_fiber_type_label)
    labels = labels.to_numpy()
    rng = np.random.default_rng(seed)

    for c in classes:
        p = float(np.mean(labels == c))
        out[f"prop_{c}"] = p

        if bootstrap_reps <= 0:
            out[f"ci95_low_{c}"] = np.nan
            out[f"ci95_high_{c}"] = np.nan
            continue

        boot = np.empty(bootstrap_reps, dtype=np.float32)
        for i in range(bootstrap_reps):
            sample = labels[rng.integers(0, n, size=n)]
            boot[i] = np.mean(sample == c)

        out[f"ci95_low_{c}"] = float(np.percentile(boot, 2.5))
        out[f"ci95_high_{c}"] = float(np.percentile(boot, 97.5))

    return out


def qc_flags_from_fibers(
    fibers: pd.DataFrame,
    cfg: QCConfig,
    marker_specs: tuple[MarkerSpec, MarkerSpec] | None = None,
) -> dict[str, str | float | int | bool]:
    n = int(len(fibers))
    if n == 0:
        return {
            "qc_status": "warn",
            "qc_reasons": "no_fibers",
            "unknown_rate": np.nan,
            "type_corr": np.nan,
            "median_area": np.nan,
            "flag_low_labels": True,
            "flag_high_unknown_rate": True,
            "flag_median_area_outlier": True,
            "flag_high_type_corr": False,
        }

    median_area = float(np.median(fibers["area"]))
    unknown_rate = float((fibers["fiber_type"] == "unknown").mean())

    if marker_specs is None:
        marker_specs = (
            MarkerSpec(marker_name="iib", legacy_prefix="type1", channel_index=0),
            MarkerSpec(marker_name="iia", legacy_prefix="type2", channel_index=1),
        )
    primary_mean_col, secondary_mean_col = _legacy_marker_pair_columns(marker_specs, "mean")
    t1 = fibers[primary_mean_col].to_numpy(dtype=np.float32)
    t2 = fibers[secondary_mean_col].to_numpy(dtype=np.float32)
    if np.std(t1) < 1e-8 or np.std(t2) < 1e-8:
        type_corr = 1.0
    else:
        type_corr = float(np.corrcoef(t1, t2)[0, 1])

    flag_low_labels = n < cfg.min_labels
    flag_high_unknown = unknown_rate > cfg.max_unknown_rate
    flag_area = not (cfg.median_area_min <= median_area <= cfg.median_area_max)
    flag_corr = type_corr > cfg.max_type_corr

    reasons = []
    if flag_low_labels:
        reasons.append("low_labels")
    if flag_high_unknown:
        reasons.append("high_unknown_rate")
    if flag_area:
        reasons.append("median_area_outlier")
    if flag_corr:
        reasons.append("high_type_corr")

    return {
        "qc_status": "pass" if len(reasons) == 0 else "warn",
        "qc_reasons": "|".join(reasons),
        "unknown_rate": unknown_rate,
        "type_corr": type_corr,
        "median_area": median_area,
        "flag_low_labels": bool(flag_low_labels),
        "flag_high_unknown_rate": bool(flag_high_unknown),
        "flag_median_area_outlier": bool(flag_area),
        "flag_high_type_corr": bool(flag_corr),
    }
