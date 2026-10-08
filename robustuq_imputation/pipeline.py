from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch

from .config import ImputerConfig
from .data import ArrayInput, FloatArray, prepare_observations
from .models import ConditionalDiffusion


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _resolve_device(device: str | torch.device) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are cpu and cuda")
    return resolved


class RobustUQImputer:

    def __init__(
        self,
        config: ImputerConfig,
        *,
        diffusion: ConditionalDiffusion,
        device: str | torch.device = "auto",
    ) -> None:
        if not isinstance(config, ImputerConfig):
            raise TypeError("config must be an ImputerConfig")
        if not isinstance(diffusion, ConditionalDiffusion):
            raise TypeError("diffusion must be a ConditionalDiffusion model")
        self.config = config
        self.device = _resolve_device(device)
        self.diffusion = diffusion.to(self.device).eval()
        self.encoder = self.diffusion.ConvTranEncoder

    @torch.no_grad()
    def encode(
        self,
        data: ArrayInput,
        observed_mask: ArrayInput | None = None,
        *,
        batch_size: int = 128,
    ) -> FloatArray:
        """Return [N, L, d_model] features, with missing inputs zero-filled."""
        _positive_integer(batch_size, "batch_size")
        values, _ = prepare_observations(data, self.config, observed_mask)
        self.encoder.eval()
        outputs = []
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(self.device)
            outputs.append(self.encoder(batch).cpu().numpy())
        return np.concatenate(outputs)

    def _sample_batches(
        self,
        data: ArrayInput,
        observed_mask: ArrayInput | None,
        n_samples: int,
        batch_size: int,
        seed: int | None,
    ) -> Iterator[FloatArray]:
        _positive_integer(n_samples, "n_samples")
        _positive_integer(batch_size, "batch_size")
        values, masks = prepare_observations(data, self.config, observed_mask)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(self.device)
            mask = torch.from_numpy(masks[start : start + batch_size]).to(self.device)
            if mask.all():
                samples = batch[:, None].expand(-1, n_samples, -1, -1)
            else:
                samples = self.diffusion.sample(batch, mask, n_samples, generator=generator)
            if not torch.isfinite(samples).all():
                raise FloatingPointError(
                    "Non-finite diffusion samples; check model parameters, schedule and input scale"
                )
            yield samples.cpu().numpy()

    def sample(
        self,
        data: ArrayInput,
        observed_mask: ArrayInput | None = None,
        *,
        n_samples: int = 20,
        batch_size: int = 32,
        seed: int | None = None,
    ) -> FloatArray:
        """Return completed stochastic draws [N, S, L, K].

        A seed reproduces draws for the same device, environment and batch size.
        Fully observed batches pass through without consuming sampling RNG.
        """
        batches = self._sample_batches(data, observed_mask, n_samples, batch_size, seed)
        return np.concatenate(list(batches))

    def transform(
        self,
        data: ArrayInput,
        observed_mask: ArrayInput | None = None,
        *,
        n_samples: int = 20,
        batch_size: int = 32,
        seed: int | None = None,
    ) -> FloatArray:
        """Return median imputation [N, L, K], reducing each batch in memory.

        For an even sample count, NumPy averages the two central draws.
        """
        batches = self._sample_batches(data, observed_mask, n_samples, batch_size, seed)
        return np.concatenate([np.median(samples, axis=1) for samples in batches])
