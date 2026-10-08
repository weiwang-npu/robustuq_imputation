"""Noise prediction with temporal/spatial attention and gated residual blocks.

External conditions have shape [B, 1, d_model, L]; noisy targets have shape
[B, 1, K, L]. Residual blocks retain the [B, C, K, L] layout.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _make_transformer_encoder(
    heads: int = 8, layers: int = 1, channels: int = 64
) -> nn.TransformerEncoder:
    """Build attention over either the time axis or the sensor axis."""
    layer = nn.TransformerEncoderLayer(
        d_model=channels,
        nhead=heads,
        dim_feedforward=64,
        activation="gelu",
        batch_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers)


def _make_conv1d(in_channels: int, out_channels: int, kernel_size: int) -> nn.Conv1d:
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    """Sinusoidal embedding of diffusion steps 1..T, followed by two projections."""

    def __init__(
        self,
        num_steps: int,
        embedding_dim: int = 128,
        projection_dim: int | None = None,
    ) -> None:
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step: Tensor) -> Tensor:
        embedding = self.embedding[diffusion_step]
        embedding = F.silu(self.projection1(embedding))
        return F.silu(self.projection2(embedding))

    @staticmethod
    def _build_embedding(num_steps: int, dim: int = 64) -> Tensor:
        # Row zero encodes diffusion step t=1.
        steps = torch.arange(1, num_steps + 1).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        phase = steps * frequencies
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)


class ConditionerEmbedding(nn.Module):
    """Project representation channels to sensors at each time step."""

    def __init__(self, input_dim: int, emb_dim: int) -> None:
        super().__init__()
        self.Embedding = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Map [B, 1, input_dim, L] to [B, 1, emb_dim, L]."""
        representation = x.squeeze(1).permute(0, 2, 1)
        projected = self.Embedding(representation)
        return projected.permute(0, 2, 1).unsqueeze(1)


class ResidualBlock(nn.Module):
    """Time attention, sensor attention, conditioning and gated activation."""

    def __init__(
        self, side_dim: int, channels: int, diffusion_embedding_dim: int, nheads: int
    ) -> None:
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = _make_conv1d(side_dim, 2 * channels, 1)
        self.mid_projection = _make_conv1d(channels, 2 * channels, 1)
        self.output_projection = _make_conv1d(channels, 2 * channels, 1)
        self.time_layer = _make_transformer_encoder(heads=nheads, layers=1, channels=channels)
        self.feature_layer = _make_transformer_encoder(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y: Tensor, base_shape: tuple[int, ...]) -> Tensor:
        """Attend over L independently for each of the B*K sensor sequences."""
        batch, channels, sensors, length = base_shape
        sequences = (
            y.reshape(batch, channels, sensors, length)
            .permute(0, 2, 1, 3)
            .reshape(batch * sensors, channels, length)
        )
        attended = self.time_layer(sequences.permute(0, 2, 1)).permute(0, 2, 1)
        return (
            attended.reshape(batch, sensors, channels, length)
            .permute(0, 2, 1, 3)
            .reshape(batch, channels, sensors * length)
        )

    def forward_feature(self, y: Tensor, base_shape: tuple[int, ...]) -> Tensor:
        """Attend over K independently for each of the B*L time slices."""
        batch, channels, sensors, length = base_shape
        sequences = (
            y.reshape(batch, channels, sensors, length)
            .permute(0, 3, 1, 2)
            .reshape(batch * length, channels, sensors)
        )
        attended = self.feature_layer(sequences.permute(0, 2, 1)).permute(0, 2, 1)
        return (
            attended.reshape(batch, length, channels, sensors)
            .permute(0, 2, 3, 1)
            .reshape(batch, channels, sensors * length)
        )

    def forward(self, x: Tensor, cond_info: Tensor, diffusion_emb: Tensor) -> tuple[Tensor, Tensor]:
        """Return residual and skip tensors, both shaped [B, C, K, L]."""
        batch, channels, sensors, length = x.shape
        base_shape = x.shape
        residual_input = x.reshape(batch, channels, sensors * length)

        step_embedding = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        hidden = residual_input + step_embedding
        hidden = self.forward_time(hidden, base_shape)
        hidden = self.forward_feature(hidden, base_shape)
        hidden = self.mid_projection(hidden)

        condition_channels = cond_info.shape[1]
        condition = cond_info.reshape(batch, condition_channels, sensors * length)
        hidden = hidden + self.cond_projection(condition)

        gate, candidate = torch.chunk(hidden, 2, dim=1)
        hidden = torch.sigmoid(gate) * torch.tanh(candidate)
        residual, skip = torch.chunk(self.output_projection(hidden), 2, dim=1)

        residual_input = residual_input.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (residual_input + residual) / math.sqrt(2.0), skip


class DenoisingNetwork(nn.Module):
    """Predict Gaussian noise from missing targets, observed features and time."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.channels = config["channels"]
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.conditioner_projection = ConditionerEmbedding(config["d_model"], config["feature_dim"])
        self.input_projection = _make_conv1d(1, self.channels, 1)
        self.output_projection1 = _make_conv1d(self.channels, self.channels, 1)
        self.output_projection2 = _make_conv1d(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=1,
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, cond_info: Tensor, noisy_target: Tensor, diffusion_step: Tensor) -> Tensor:
        """Map condition [B, 1, D, L] and target [B, 1, K, L] to [B, K, L]."""
        batch, input_channels, sensors, length = noisy_target.shape
        hidden = noisy_target.reshape(batch, input_channels, sensors * length)
        hidden = F.relu(self.input_projection(hidden))
        hidden = hidden.reshape(batch, self.channels, sensors, length)

        step_embedding = self.diffusion_embedding(diffusion_step)
        condition = self.conditioner_projection(cond_info)
        skip_connections = []
        for layer in self.residual_layers:
            hidden, skip = layer(hidden, condition, step_embedding)
            skip_connections.append(skip)

        hidden = torch.sum(torch.stack(skip_connections), dim=0)
        hidden = hidden.reshape(batch, self.channels, sensors * length)
        hidden = F.relu(self.output_projection1(hidden))
        return self.output_projection2(hidden).reshape(batch, sensors, length)
