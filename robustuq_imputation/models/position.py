import math

import torch
from torch import Tensor, nn


class AbsolutePositionalEncoding(nn.Module):
    """Add fixed time positions to a [B, L, D] embedding, then apply dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("pe", encoding.unsqueeze(0), persistent=False)

    def forward(self, values: Tensor) -> Tensor:
        return self.dropout(values + self.pe[:, : values.shape[1]])
