"""Shared fixtures for the SLIDERS test suite."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


@pytest.fixture
def input_dim() -> int:
    return 32


@pytest.fixture
def hidden_dim() -> int:
    return 64


@pytest.fixture
def small_sae(input_dim, hidden_dim):
    from src.models.sae import SparseAutoencoder

    return SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)


@pytest.fixture
def small_sae_topk(input_dim, hidden_dim):
    from src.models.sae import SparseAutoencoder

    return SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, topk=4)


@pytest.fixture
def small_sae_tied(input_dim, hidden_dim):
    from src.models.sae import SparseAutoencoder

    return SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, tied_weights=True)


@pytest.fixture
def fake_embeddings(input_dim) -> np.ndarray:
    rng = np.random.default_rng(42)
    embs = rng.standard_normal((200, input_dim)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.where(norms > 0, norms, 1.0)


@pytest.fixture
def fake_activations(hidden_dim) -> np.ndarray:
    """Sparse non-negative activations — most entries are zero."""
    rng = np.random.default_rng(0)
    acts = rng.standard_normal((200, hidden_dim)).astype(np.float32)
    acts = np.maximum(acts, 0.0)  # ReLU
    return acts


@pytest.fixture
def fake_image_paths(tmp_path) -> list[str]:
    """Create 200 tiny PNG files and return their paths."""
    from PIL import Image

    paths = []
    for i in range(200):
        p = tmp_path / f"img_{i:04d}.png"
        img = Image.fromarray(
            np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        )
        img.save(p)
        paths.append(str(p))
    return paths


@pytest.fixture
def embeddings_npy(tmp_path, fake_embeddings) -> Path:
    path = tmp_path / "embeddings.npy"
    np.save(path, fake_embeddings)
    return path


@pytest.fixture
def mock_clip_encoder():
    encoder = MagicMock()

    def fake_preprocess(img):
        return torch.zeros(3, 224, 224)

    def fake_encode_images(images: torch.Tensor) -> torch.Tensor:
        b = images.shape[0]
        out = torch.randn(b, 768)
        out = torch.nn.functional.normalize(out, dim=-1)
        return out

    encoder.preprocess = fake_preprocess
    encoder.encode_images = fake_encode_images
    return encoder


@pytest.fixture
def mock_dino_encoder(input_dim):
    encoder = MagicMock()

    def fake_encode(images: torch.Tensor) -> torch.Tensor:
        b = images.shape[0]
        return torch.randn(b, input_dim)

    encoder.encode = fake_encode
    return encoder
