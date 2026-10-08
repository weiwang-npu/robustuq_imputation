"""Conditional diffusion training and sampling over missing entries."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import ImputerConfig
from .denoiser import DenoisingNetwork


class ConditionalDiffusion(nn.Module):
    """Freeze the representation encoder and learn a conditional noise predictor.

    Public tensors use [B, L, K]; the denoiser uses [B, 1, K, L].
    """

    def __init__(self, config: ImputerConfig, encoder: nn.Module) -> None:
        super().__init__()
        self.ConvTranEncoder = encoder
        self.ConvTranEncoder.requires_grad_(False).eval()
        self.diffmodel = DenoisingNetwork(config.to_dict())
        self.num_steps = config.num_steps

        # Compute schedules in float64; cast coefficients to the input dtype when used.
        beta = torch.linspace(
            config.beta_start, config.beta_end, self.num_steps, dtype=torch.float64
        )
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar_previous = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bar[:-1]])
        # Index t-1 represents diffusion step t in 1..T; alpha_bar_0 = 1.
        self.register_buffer("beta", beta, persistent=False)
        self.register_buffer("alpha", alpha, persistent=False)
        self.register_buffer("alpha_bar", alpha_bar, persistent=False)
        self.register_buffer(
            "posterior_variance",
            beta * (1 - alpha_bar_previous) / (1 - alpha_bar),
            persistent=False,
        )

    def train(self, mode: bool = True) -> ConditionalDiffusion:
        super().train(mode)
        # Disabling gradients alone does not freeze BatchNorm or dropout.
        self.ConvTranEncoder.eval()
        return self

    def condition(self, values: Tensor, observed_mask: Tensor) -> Tensor:
        """Encode only observed values, returning [B, 1, d_model, L]."""
        with torch.no_grad():
            representation = self.ConvTranEncoder(values * observed_mask)
        return representation.transpose(1, 2).unsqueeze(1)

    def forward(self, values: Tensor, observed_mask: Tensor) -> Tensor:
        """Return the batch mean of squared L2 noise error over missing entries."""
        missing = (~observed_mask.bool()).transpose(1, 2)
        if not missing.any():
            raise ValueError("Diffusion training needs at least one masked position")

        target = values.transpose(1, 2) * missing
        timestep = torch.randint(self.num_steps, (len(values),), device=values.device)
        alpha_bar = self.alpha_bar[timestep].to(values.dtype).view(-1, 1, 1)
        noise = torch.randn_like(target) * missing
        noisy_target = alpha_bar.sqrt() * target + (1 - alpha_bar).sqrt() * noise
        predicted_noise = self.diffmodel(
            self.condition(values, observed_mask), noisy_target.unsqueeze(1), timestep
        )
        return ((noise - predicted_noise).square() * missing).sum(dim=(1, 2)).mean()

    @torch.no_grad()
    def sample(
        self,
        values: Tensor,
        observed_mask: Tensor,
        n_samples: int,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Return completed draws [B, S, L, K], retaining observed entries."""
        self.eval()
        condition = self.condition(values, observed_mask)
        observed = values.transpose(1, 2)
        missing = (~observed_mask.bool()).transpose(1, 2)
        results = []

        for _ in range(n_samples):
            current = (
                torch.randn(
                    observed.shape,
                    dtype=observed.dtype,
                    device=values.device,
                    generator=generator,
                )
                * missing
            )

            for step in range(self.num_steps - 1, -1, -1):
                timestep = torch.tensor([step], device=values.device)
                predicted_noise = self.diffmodel(
                    condition, (missing * current).unsqueeze(1), timestep
                )
                mean_scale = (1 / self.alpha[step].sqrt()).to(values.dtype)
                prediction_scale = (self.beta[step] / (1 - self.alpha_bar[step]).sqrt()).to(
                    values.dtype
                )
                current = mean_scale * (current - prediction_scale * predicted_noise) * missing

                # At t=1 the posterior variance is zero, so no noise is added.
                if step:
                    standard_deviation = self.posterior_variance[step].sqrt().to(values.dtype)
                    noise = torch.randn(
                        current.shape,
                        dtype=current.dtype,
                        device=current.device,
                        generator=generator,
                    )
                    current = current + standard_deviation * noise * missing

            completed = torch.where(missing, current, observed).transpose(1, 2)
            results.append(completed)

        return torch.stack(results, dim=1)
