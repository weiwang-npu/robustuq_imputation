import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ImputerConfig:
    
    # Window dimensions and missingness probabilities.
    seq_len: int = 30
    feature_dim: int = 14
    missing_ratio: float = 0.1
    malfunction_ratio: float = 0.1

    # Convolutional embedding and representation encoder.
    emb_size: int = 256
    d_model: int = 64
    num_encoder_layers: int = 2
    num_heads: int = 8
    dim_feedforward: int = 256
    dropout: float = 0.1

    # Linear diffusion schedule and residual denoising network.
    beta_start: float = 0.0004
    beta_end: float = 0.05
    num_steps: int = 50
    layers: int = 4
    channels: int = 64
    nheads: int = 8
    diffusion_embedding_dim: int = 128

    def __post_init__(self) -> None:
        integer_fields = (
            "seq_len",
            "feature_dim",
            "emb_size",
            "d_model",
            "num_encoder_layers",
            "num_heads",
            "dim_feedforward",
            "num_steps",
            "layers",
            "channels",
            "nheads",
            "diffusion_embedding_dim",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.num_heads or self.channels % self.nheads:
            raise ValueError("Embedding/channel dimensions must be divisible by attention heads")
        if self.d_model % 2 or self.diffusion_embedding_dim < 4 or self.diffusion_embedding_dim % 2:
            raise ValueError("d_model must be even; diffusion_embedding_dim must be even and >= 4")
        for name in ("missing_ratio", "malfunction_ratio"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 < self.beta_start <= self.beta_end < 1:
            raise ValueError("Require 0 < beta_start <= beta_end < 1")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached mapping for the lower-level network constructors."""
        return asdict(self)
