"""Turn-detection model architectures.

- WhisperTinyTurn: pretrained Whisper-Tiny encoder truncated to an 8s input
  window (positional embeddings sliced 1500 -> 400), + attention pooling +
  small MLP head. ~8M params.
- TinyMelNet: from-scratch comparison, ~1M params: depthwise-separable Conv1d
  stack over mel frames + BiGRU + the same pooling/head.

Both take log-mel (B, 80, 800) and return a single logit per example
(sigmoid -> P(turn complete)).
"""

import torch
import torch.nn as nn

from turn_detector.features import N_ENCODER_POSITIONS


class AttnPool(nn.Module):
    """Learned-query attention pooling over time: (B, T, D) -> (B, D)."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(x @ self.query * self.scale, dim=1)
        return (weights.unsqueeze(-1) * x).sum(dim=1)


def make_head(dim: int, hidden: int = 256, dropout: float = 0.1) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )


def truncate_whisper_encoder(encoder, n_positions: int = N_ENCODER_POSITIONS):
    """Slice Whisper's 30s positional table to our window length, in place."""
    old = encoder.embed_positions
    new = nn.Embedding(n_positions, old.embedding_dim)
    new.weight.data.copy_(old.weight.data[:n_positions])
    encoder.embed_positions = new
    encoder.config.max_source_positions = n_positions
    if hasattr(encoder, "max_source_positions"):
        encoder.max_source_positions = n_positions
    return encoder


class WhisperTinyTurn(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        dim = encoder.config.d_model
        self.pool = AttnPool(dim)
        self.head = make_head(dim)

    @classmethod
    def from_pretrained(cls, name: str = "openai/whisper-tiny"):
        from transformers import WhisperModel
        encoder = WhisperModel.from_pretrained(name).encoder
        return cls(truncate_whisper_encoder(encoder))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(mel).last_hidden_state       # (B, 400, 384)
        return self.head(self.pool(hidden)).squeeze(-1)


class DSConvBlock(nn.Module):
    """Depthwise-separable Conv1d + BN + GELU."""

    def __init__(self, channels: int, kernel: int = 5, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, kernel, stride=stride,
            padding=kernel // 2, groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class TinyMelNet(nn.Module):
    def __init__(self, width: int = 192, gru_hidden: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(80, width, 5, stride=2, padding=2),
            nn.BatchNorm1d(width),
            nn.GELU(),
            DSConvBlock(width, stride=2),
            DSConvBlock(width, stride=2),
            DSConvBlock(width, stride=1),
        )                                                   # (B, width, 100)
        self.gru = nn.GRU(width, gru_hidden, batch_first=True, bidirectional=True)
        self.pool = AttnPool(2 * gru_hidden)
        self.head = make_head(2 * gru_hidden)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.stem(mel).transpose(1, 2)                  # (B, 100, width)
        x, _ = self.gru(x)                                  # (B, 100, 2*hidden)
        return self.head(self.pool(x)).squeeze(-1)


def build_model(arch: str) -> nn.Module:
    if arch == "whisper":
        return WhisperTinyTurn.from_pretrained()
    if arch == "tinymel":
        return TinyMelNet()
    raise ValueError(f"unknown arch: {arch}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
