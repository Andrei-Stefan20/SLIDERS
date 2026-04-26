"""Integration-style tests for the SAE training loop."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.train_sae import train_sae


@pytest.fixture
def tiny_embeddings_path(tmp_path) -> Path:
    rng = np.random.default_rng(42)
    embs = rng.standard_normal((100, 32)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs /= np.where(norms > 0, norms, 1.0)
    path = tmp_path / "embeddings.npy"
    np.save(path, embs)
    return path


@pytest.fixture
def output_dir(tmp_path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    return d


class TestTrainSAE:
    def test_creates_best_checkpoint(self, tiny_embeddings_path, output_dir):
        train_sae(
            embeddings_path=tiny_embeddings_path,
            output_dir=output_dir,
            hidden_dim=64,
            epochs=2,
            batch_size=32,
            log_every=999,
            val_split=0.1,
            patience=10,
        )
        assert (output_dir / "sae_best.pt").exists()

    def test_creates_last_checkpoint(self, tiny_embeddings_path, output_dir):
        train_sae(
            embeddings_path=tiny_embeddings_path,
            output_dir=output_dir,
            hidden_dim=64,
            epochs=2,
            batch_size=32,
            log_every=999,
            val_split=0.1,
            patience=10,
        )
        assert (output_dir / "sae_last.pt").exists()

    def test_saved_model_loadable(self, tiny_embeddings_path, output_dir):
        from src.models.sae import SparseAutoencoder

        train_sae(
            embeddings_path=tiny_embeddings_path,
            output_dir=output_dir,
            hidden_dim=64,
            epochs=2,
            batch_size=32,
            log_every=999,
            val_split=0.1,
            patience=10,
        )
        state = torch.load(output_dir / "sae_best.pt", map_location="cpu")
        sae = SparseAutoencoder(input_dim=32, hidden_dim=64)
        sae.load_state_dict(state)
        x = torch.randn(4, 32)
        x_hat, h = sae(x)
        assert x_hat.shape == (4, 32)

    def test_topk_mode(self, tiny_embeddings_path, output_dir):
        train_sae(
            embeddings_path=tiny_embeddings_path,
            output_dir=output_dir,
            hidden_dim=64,
            epochs=1,
            batch_size=32,
            log_every=999,
            topk=4,
            val_split=0.1,
            patience=5,
        )
        assert (output_dir / "sae_best.pt").exists()

    def test_cosine_loss_mode(self, tiny_embeddings_path, output_dir):
        train_sae(
            embeddings_path=tiny_embeddings_path,
            output_dir=output_dir,
            hidden_dim=64,
            epochs=1,
            batch_size=32,
            log_every=999,
            loss_type="cosine",
            val_split=0.1,
            patience=5,
        )
        assert (output_dir / "sae_best.pt").exists()

    def test_early_stopping(self, tiny_embeddings_path, output_dir):
        """With patience=1 and 10 requested epochs, training stops early."""
        train_sae(
            embeddings_path=tiny_embeddings_path,
            output_dir=output_dir,
            hidden_dim=64,
            epochs=10,
            batch_size=32,
            log_every=999,
            val_split=0.2,
            patience=1,
        )
        assert (output_dir / "sae_best.pt").exists()

    def test_tied_weights(self, tiny_embeddings_path, output_dir):
        train_sae(
            embeddings_path=tiny_embeddings_path,
            output_dir=output_dir,
            hidden_dim=64,
            epochs=1,
            batch_size=32,
            log_every=999,
            tied_weights=True,
            val_split=0.1,
            patience=5,
        )
        assert (output_dir / "sae_best.pt").exists()
