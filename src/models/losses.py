"""Loss functions for SAE training."""

import torch
import torch.nn.functional as F


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x)


def cosine_reconstruction_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """Cosine + MSE loss. Optimises angular alignment to match the unit-sphere
    retrieval geometry used by FAISS IndexFlatIP."""
    cos = F.cosine_similarity(x, x_hat, dim=1)
    return (1.0 - cos).mean() + alpha * F.mse_loss(x_hat, x)


def sparsity_loss(h: torch.Tensor) -> torch.Tensor:
    return h.abs().mean()
