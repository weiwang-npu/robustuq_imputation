"""Convolutional embedding and Transformer representation learning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from .position import AbsolutePositionalEncoding


class ConvTranEncoder(nn.Module):
    """Encode [B, L, K] windows into [B, L, d_model] representations."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        d_model = config["d_model"]
        self.embed_layer1 = nn.Sequential(
            nn.Conv2d(1, config["emb_size"], kernel_size=(1, 8), padding="same"),
            nn.BatchNorm2d(config["emb_size"]),
            nn.GELU(),
        )
        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(config["emb_size"], d_model, kernel_size=(config["feature_dim"], 1)),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )
        self.pos_enc = AbsolutePositionalEncoding(
            d_model, dropout=config["dropout"], max_len=config["seq_len"]
        )
        # Construct separately so the layers have independent initial weights.
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=config["num_heads"],
                    dim_feedforward=config["dim_feedforward"],
                    dropout=config["dropout"],
                    activation="relu",
                    batch_first=True,
                    norm_first=False,
                )
                for _ in range(config["num_encoder_layers"])
            ]
        )

    def forward(self, values: Tensor) -> Tensor:
        """Embed each time step while attending across the whole window."""
        embedded = self.embed_layer1(values.transpose(1, 2).unsqueeze(1))
        embedded = self.embed_layer2(embedded).squeeze(2).transpose(1, 2)
        representation = self.pos_enc(embedded)
        for layer in self.encoder_layers:
            representation = layer(representation)
        return representation


class MaskedReconstructionModel(nn.Module):
    """Temporary linear reconstruction head, discarded after pretraining."""

    def __init__(self, encoder: nn.Module, d_model: int, feature_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.output_layer = nn.Linear(d_model, feature_dim)

    def forward(self, values: Tensor) -> Tensor:
        return self.output_layer(self.encoder(values))

    def loss(self, values: Tensor, observed_mask: Tensor) -> Tensor:
        """Return squared reconstruction error averaged over masked entries."""
        missing = ~observed_mask.bool()
        if not missing.any():
            raise ValueError("Reconstruction training needs at least one masked position")
        reconstructed = self(values * observed_mask)
        return (reconstructed - values).square().masked_select(missing).mean()
