"""Tests for RetrievalService."""

from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from src.ui.retrieval_service import RetrievalService, SearchResult
from src.ui.state import AppState


@pytest.fixture
def mock_state(input_dim, hidden_dim, fake_embeddings, fake_activations, fake_image_paths):
    import faiss

    index = faiss.IndexFlatIP(input_dim)
    cast(Any, index).add(fake_embeddings)

    from src.models.sae import SparseAutoencoder
    sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)

    state = AppState(
        dino=MagicMock(),
        sae=sae,
        index=index,
        sae_index=None,
        embeddings=fake_embeddings,
        activations=fake_activations,
        image_paths=fake_image_paths,
        feature_ids=list(range(10)),
        feature_names=[f"feature_{i}" for i in range(10)],
        feature_descriptions=[""] * 10,
    )
    return state


@pytest.fixture
def service(mock_state):
    return RetrievalService(mock_state)


class TestEncodeImage:
    def test_returns_unit_norm_array(self, service, input_dim):
        service._state.dino.encode.return_value = torch.randn(1, input_dim)

        query_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        emb = service.encode_image(query_img)

        assert emb.shape == (input_dim,)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-4

    def test_accepts_rgb_array(self, service, input_dim):
        service._state.dino.encode.return_value = torch.randn(1, input_dim)
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        emb = service.encode_image(img)
        assert emb.ndim == 1


class TestRetrieve:
    def test_returns_list_of_images(self, service, input_dim):
        query_emb = np.random.randn(input_dim).astype(np.float32)
        query_emb /= np.linalg.norm(query_emb)
        slider_values = [0.0] * len(service._state.feature_ids)

        results = service.retrieve(query_emb, slider_values, k=5)

        assert isinstance(results, list)
        assert len(results) <= 5
        for r in results:
            assert isinstance(r, SearchResult)
            assert isinstance(r.image, Image.Image)

    def test_returns_empty_for_zero_sliders(self, service, input_dim):
        query_emb = np.random.randn(input_dim).astype(np.float32)
        query_emb /= np.linalg.norm(query_emb)
        slider_values = [0.0] * len(service._state.feature_ids)

        results = service.retrieve(query_emb, slider_values, k=3)
        assert isinstance(results, list)

    def test_uses_direction_sliders_when_class_directions_set(
        self, service, input_dim, fake_embeddings
    ):
        n_dirs = 5
        dirs = np.random.randn(n_dirs, input_dim).astype(np.float32)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        service._state.class_directions = dirs
        service._state.feature_ids = list(range(n_dirs))
        service._state.feature_names = [f"dir_{i}" for i in range(n_dirs)]

        query_emb = fake_embeddings[0]
        results = service.retrieve(query_emb, [0.0] * n_dirs, k=4)
        assert isinstance(results, list)
