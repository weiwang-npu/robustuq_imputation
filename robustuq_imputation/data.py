from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch

from .config import ImputerConfig

ArrayInput: TypeAlias = ArrayLike | torch.Tensor
FloatArray: TypeAlias = NDArray[np.float32]
BoolArray: TypeAlias = NDArray[np.bool_]


def as_windows(data: ArrayInput, config: ImputerConfig, *, complete: bool = False) -> FloatArray:
    """Validate [N, L, K] and return an independent, contiguous float32 array."""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    array = np.asarray(data, dtype=np.float32)
    expected = (config.seq_len, config.feature_dim)
    if array.ndim != 3 or array.shape[1:] != expected or len(array) == 0:
        raise ValueError(f"Expected nonempty windows [N, {expected[0]}, {expected[1]}]")
    if complete and not np.isfinite(array).all():
        raise ValueError("Training requires complete, finite windows")
    return np.array(array, copy=True, order="C")


def prepare_observations(
    data: ArrayInput,
    config: ImputerConfig,
    observed_mask: ArrayInput | None = None,
) -> tuple[FloatArray, BoolArray]:
    """Replace missing entries with zero without treating real zeros as missing.

    An explicit mask takes precedence over NaN inference. Only entries marked
    observed must be finite; arbitrary placeholders at missing entries are safe.
    """
    array = as_windows(data, config)
    if observed_mask is None:
        if np.isinf(array).any():
            raise ValueError("Use NaN, not infinity, to mark missing observations")
        mask = ~np.isnan(array)
    else:
        if isinstance(observed_mask, torch.Tensor):
            observed_mask = observed_mask.detach().cpu().numpy()
        mask = np.asarray(observed_mask)
        if mask.shape != array.shape or not np.isin(mask, [0, 1]).all():
            raise ValueError("observed_mask must have the same shape as data and contain only 0/1")
        mask = mask.astype(bool)
        if not np.isfinite(array[mask]).all():
            raise ValueError("Observed positions must be finite")
    return np.where(mask, array, 0).astype(np.float32), mask


def make_observed_mask(
    shape: Sequence[int],
    missing_ratio: float = 0.1,
    malfunction_ratio: float = 0.1,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> BoolArray:
    """Combine independent point missingness with whole-sensor failures.

    Each point is observed with probability 1 - missing_ratio. For each window,
    floor(K * malfunction_ratio) uniformly selected sensors are fully masked.
    Random point counts fluctuate; an entirely observed window is possible.
    """
    if len(shape) not in (2, 3) or any(int(size) != size or size < 1 for size in shape):
        raise ValueError("shape must be [L, K] or [N, L, K] with positive sizes")
    if not (0 <= missing_ratio <= 1 and 0 <= malfunction_ratio <= 1):
        raise ValueError("Missing ratios must be in [0, 1]")
    if seed is not None and rng is not None:
        raise ValueError("Pass either seed or rng, not both")
    rng = np.random.default_rng(seed) if rng is None else rng
    single = len(shape) == 2
    masks = rng.random((1, *shape) if single else shape) >= missing_ratio
    features = masks.shape[-1]
    failed = int(malfunction_ratio * features)
    for mask in masks:
        sensors = rng.choice(features, failed, replace=False)
        mask[:, sensors] = False
    return masks[0] if single else masks
