"""Tests for SAE loss functions."""

import pytest
import torch

from src.models.losses import cosine_reconstruction_loss, reconstruction_loss, sparsity_loss


class TestReconstructionLoss:
    def test_zero_when_perfect(self):
        x = torch.randn(8, 32)
        assert reconstruction_loss(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_different(self):
        x = torch.randn(8, 32)
        x_hat = torch.zeros_like(x)
        assert reconstruction_loss(x, x_hat).item() > 0

    def test_scalar_output(self):
        x = torch.randn(4, 16)
        loss = reconstruction_loss(x, x + 0.1)
        assert loss.ndim == 0

    def test_symmetric(self):
        x = torch.randn(8, 32)
        y = torch.randn(8, 32)
        assert reconstruction_loss(x, y).item() == pytest.approx(
            reconstruction_loss(y, x).item(), rel=1e-5
        )


class TestCosineReconstructionLoss:
    def test_near_zero_for_aligned(self):
        x = torch.nn.functional.normalize(torch.randn(8, 32), dim=1)
        loss = cosine_reconstruction_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_for_different(self):
        x = torch.randn(8, 32)
        x_hat = torch.zeros(8, 32) + 0.1
        assert cosine_reconstruction_loss(x, x_hat).item() > 0

    def test_scalar_output(self):
        x = torch.randn(4, 16)
        loss = cosine_reconstruction_loss(x, x * 2)
        assert loss.ndim == 0

    def test_alpha_controls_mse_weight(self):
        x = torch.nn.functional.normalize(torch.randn(8, 32), dim=1)
        x_hat = x * 2  # same direction, different magnitude
        loss_default = cosine_reconstruction_loss(x, x_hat)
        loss_no_mse = cosine_reconstruction_loss(x, x_hat, alpha=0.0)
        assert loss_no_mse.item() == pytest.approx(0.0, abs=1e-5)
        assert loss_default.item() > loss_no_mse.item()


class TestSparsityLoss:
    def test_zero_for_zeros(self):
        h = torch.zeros(8, 64)
        assert sparsity_loss(h).item() == pytest.approx(0.0, abs=1e-8)

    def test_positive_for_nonzero(self):
        h = torch.abs(torch.randn(8, 64))
        assert sparsity_loss(h).item() > 0

    def test_scalar_output(self):
        h = torch.randn(4, 32)
        loss = sparsity_loss(h)
        assert loss.ndim == 0

    @pytest.mark.parametrize("scale", [0.1, 1.0, 10.0])
    def test_scales_with_magnitude(self, scale):
        h_base = torch.abs(torch.randn(8, 64))
        assert sparsity_loss(h_base * scale).item() == pytest.approx(
            sparsity_loss(h_base).item() * scale, rel=1e-4
        )
