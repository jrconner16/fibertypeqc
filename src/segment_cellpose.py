from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from cellpose import models


@dataclass
class CellposeConfig:
    pretrained_model: str = "cpsam"
    diameter: float | None = 30
    bsize: int = 256
    resample: bool = False
    use_mps: bool = True
    normalize: bool = False


_CACHED_MODELS: dict[tuple[str, str], models.CellposeModel] = {}


def _device_name(use_mps: bool) -> str:
    if use_mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_cellpose_model(cfg: CellposeConfig) -> models.CellposeModel:
    dev = _device_name(cfg.use_mps)
    key = (cfg.pretrained_model, dev)
    if key in _CACHED_MODELS:
        return _CACHED_MODELS[key]

    device = torch.device(dev) if dev != "cpu" else None
    model = models.CellposeModel(
        gpu=(dev != "cpu"),
        device=device,
        pretrained_model=cfg.pretrained_model,
    )
    _CACHED_MODELS[key] = model
    return model


def run_cellpose(membrane_image: np.ndarray, cfg: CellposeConfig) -> tuple[np.ndarray, float]:
    model = get_cellpose_model(cfg)

    if cfg.bsize != 256:
        raise ValueError("For Cellpose v4 cpsam, use bsize=256 to avoid shape mismatch errors")

    print(f"Cellpose device: {_device_name(cfg.use_mps)}", flush=True)
    t0 = perf_counter()
    result = model.eval(
        membrane_image,
        diameter=cfg.diameter,
        bsize=cfg.bsize,
        resample=cfg.resample,
        normalize=cfg.normalize,
    )
    elapsed_s = perf_counter() - t0

    masks = result[0] if isinstance(result, tuple) else result
    return masks.astype(np.int32), elapsed_s
