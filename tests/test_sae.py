"""Tests for SparseAutoencoder."""

import torch
import pytest

from src.models.sae import SparseAutoencoder


class TestSparseAutoencoderShapes:
    def test_forward_returns_tuple(self, small_sae, input_dim):
        x = torch.randn(8, input_dim)
        out = small_sae(x)
        assert isinstance(out, tuple) and len(out) == 2

    def test_reconstruction_shape(self, small_sae, input_dim):
        x = torch.randn(8, input_dim)
        x_hat, h = small_sae(x)
        assert x_hat.shape == x.shape

    def test_hidden_shape(self, small_sae, input_dim, hidden_dim):
        x = torch.randn(8, input_dim)
        _, h = small_sae(x)
        assert h.shape == (8, hidden_dim)

    def test_encode_shape(self, small_sae, input_dim, hidden_dim):
        x = torch.randn(4, input_dim)
        h = small_sae.encode(x)
        assert h.shape == (4, hidden_dim)

    def test_decode_shape(self, small_sae, input_dim, hidden_dim):
        h = torch.randn(4, hidden_dim)
        x_hat = small_sae.decode(h)
        assert x_hat.shape == (4, input_dim)


class TestFeatureDirections:
    def test_untied_uses_unit_decoder_columns(self, input_dim, hidden_dim):
        # the steering atom is the decoder column, not the encoder row (docs/adr/0002)
        sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, tied_weights=False)
        dirs = sae.feature_directions([0, 3])
        assert dirs.shape == (2, input_dim)
        expected = torch.nn.functional.normalize(sae.decoder.weight.detach().t()[[0, 3]], dim=-1)
        assert torch.allclose(dirs, expected, atol=1e-6)
        assert torch.allclose(dirs.norm(dim=-1), torch.ones(2), atol=1e-6)

    def test_tied_uses_encoder_rows(self, input_dim, hidden_dim):
        sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, tied_weights=True)
        dirs = sae.feature_directions([1])
        expected = torch.nn.functional.normalize(sae.encoder.weight.detach()[[1]], dim=-1)
        assert torch.allclose(dirs, expected, atol=1e-6)


class TestReLUActivation:
    def test_hidden_non_negative(self, small_sae, input_dim):
        x = torch.randn(16, input_dim)
        _, h = small_sae(x)
        assert (h >= 0).all()

    def test_sparse_by_construction(self, small_sae, input_dim):
        x = torch.randn(256, input_dim)
        _, h = small_sae(x)
        sparsity = (h == 0).float().mean().item()
        assert sparsity > 0.0, "Expected some dead neurons with random inputs"


class TestTopKActivation:
    def test_exactly_k_active(self, small_sae_topk, input_dim):
        sae = small_sae_topk
        x = torch.randn(8, input_dim)
        h = sae.encode(x)
        n_active = (h > 0).sum(dim=1)
        assert (n_active <= sae.topk).all()

    def test_topk_shape(self, small_sae_topk, input_dim, hidden_dim):
        x = torch.randn(4, input_dim)
        _, h = small_sae_topk(x)
        assert h.shape == (4, hidden_dim)

    def test_topk_non_negative(self, small_sae_topk, input_dim):
        x = torch.randn(16, input_dim)
        _, h = small_sae_topk(x)
        assert (h >= 0).all()


class TestTiedWeights:
    def test_forward_shape(self, small_sae_tied, input_dim):
        x = torch.randn(8, input_dim)
        x_hat, h = small_sae_tied(x)
        assert x_hat.shape == x.shape
        assert h.shape[1] == small_sae_tied.hidden_dim

    def test_no_decoder_attribute(self, small_sae_tied):
        assert not hasattr(small_sae_tied, "decoder")

    def test_has_decoder_bias(self, small_sae_tied):
        assert hasattr(small_sae_tied, "decoder_bias")


class TestParameterizedConfigs:
    @pytest.mark.parametrize("input_dim,hidden_dim", [(16, 32), (64, 128), (128, 512)])
    def test_various_dims(self, input_dim, hidden_dim):
        sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
        x = torch.randn(4, input_dim)
        x_hat, h = sae(x)
        assert x_hat.shape == (4, input_dim)
        assert h.shape == (4, hidden_dim)

    @pytest.mark.parametrize("topk", [1, 4, 8])
    def test_topk_values(self, topk):
        sae = SparseAutoencoder(input_dim=32, hidden_dim=64, topk=topk)
        x = torch.randn(8, 32)
        _, h = sae(x)
        assert (h >= 0).all()
        assert ((h > 0).sum(dim=1) <= topk).all()
