"""Sparse Autoencoder for disentangling DINOv2 feature space."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """Single-layer SAE with ReLU or TopK activation.

    topk > 0: exact K features active per sample (Gao et al., 2024)
    topk = 0: ReLU + L1 penalty (default)
    tied_weights: decoder uses encoder.weight.T
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 8192,
        tied_weights: bool = False,
        topk: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tied_weights = tied_weights
        self.topk = topk

        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        if not tied_weights:
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        else:
            self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        if not self.tied_weights:
            nn.init.kaiming_uniform_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.encoder(x)
        if self.topk > 0:
            h = torch.zeros_like(pre)
            topk_vals, topk_idx = pre.topk(self.topk, dim=1)
            h.scatter_(1, topk_idx, F.relu(topk_vals))
            return h
        return F.relu(pre)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        if self.tied_weights:
            return h @ self.encoder.weight + self.decoder_bias
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        return self.decode(h), h
