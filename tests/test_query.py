"""Tests for the retrieval query interface."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.retrieval.query import _mmr_rerank, _sae_activation_rerank, search, search_with_sliders


def _make_mock_index(n_vectors: int, dim: int, seed: int = 0):
    """Return a mock faiss.Index whose .search() returns plausible results."""
    rng = np.random.default_rng(seed)
    corpus = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    norms = np.linalg.norm(corpus, axis=1, keepdims=True)
    corpus = corpus / np.where(norms > 0, norms, 1.0)

    def fake_search(query: np.ndarray, k: int):
        sims = (corpus @ query.T).squeeze()  # (n_vectors,)
        top_k = np.argsort(sims)[::-1][:k]
        return sims[top_k].reshape(1, -1), top_k.reshape(1, -1)

    idx = MagicMock()
    idx.search.side_effect = fake_search
    return idx


@pytest.fixture
def mock_faiss_index(fake_embeddings):
    return _make_mock_index(len(fake_embeddings), fake_embeddings.shape[1])


@pytest.fixture
def query_emb(input_dim):
    rng = np.random.default_rng(99)
    q = rng.standard_normal(input_dim).astype(np.float32)
    return q / np.linalg.norm(q)


class TestSearch:
    def test_returns_k_results(self, mock_faiss_index, query_emb):
        dists, idxs = search(mock_faiss_index, query_emb, k=5)
        assert len(dists) == 5
        assert len(idxs) == 5

    def test_indices_in_bounds(self, mock_faiss_index, query_emb, fake_embeddings):
        _, idxs = search(mock_faiss_index, query_emb, k=10)
        assert all(0 <= i < len(fake_embeddings) for i in idxs)

    def test_distances_in_range(self, mock_faiss_index, query_emb):
        dists, _ = search(mock_faiss_index, query_emb, k=5)
        assert all(-1.0 <= d <= 1.0 + 1e-5 for d in dists)

    def test_k_equals_1(self, mock_faiss_index, query_emb):
        dists, idxs = search(mock_faiss_index, query_emb, k=1)
        assert len(dists) == 1
        assert len(idxs) == 1


class TestMMRRerank:
    def test_preserves_all_indices(self, fake_embeddings):
        dists = np.random.rand(10).astype(np.float32)
        idxs = np.arange(10)
        out_dists, out_idxs = _mmr_rerank(idxs, dists, fake_embeddings, mmr_lambda=0.5)
        assert set(out_idxs.tolist()) == set(idxs.tolist())

    def test_lambda_1_unchanged(self, fake_embeddings):
        dists = np.random.rand(5).astype(np.float32)
        idxs = np.arange(5)
        out_dists, out_idxs = _mmr_rerank(idxs, dists, fake_embeddings, mmr_lambda=1.0)
        np.testing.assert_array_equal(out_idxs, idxs)

    def test_single_element_unchanged(self, fake_embeddings):
        dists = np.array([0.9], dtype=np.float32)
        idxs = np.array([3])
        out_dists, out_idxs = _mmr_rerank(idxs, dists, fake_embeddings, mmr_lambda=0.5)
        np.testing.assert_array_equal(out_idxs, idxs)

    def test_output_is_permutation(self, fake_embeddings):
        n = 8
        dists = np.random.rand(n).astype(np.float32)
        idxs = np.arange(n)
        out_dists, out_idxs = _mmr_rerank(idxs, dists, fake_embeddings, mmr_lambda=0.6)
        assert len(out_idxs) == n


class TestSAEActivationRerank:
    def test_reranks_indices(self, fake_activations):
        idxs = np.arange(10)
        dists = np.random.rand(10).astype(np.float32)
        out_dists, out_idxs = _sae_activation_rerank(
            idxs, dists, fake_activations,
            active_feature_ids=[0, 1],
            active_alphas=[1.0, -1.0],
        )
        assert set(out_idxs.tolist()) == set(idxs.tolist())

    def test_output_length_preserved(self, fake_activations):
        n = 15
        idxs = np.arange(n)
        dists = np.random.rand(n).astype(np.float32)
        out_dists, out_idxs = _sae_activation_rerank(
            idxs, dists, fake_activations,
            active_feature_ids=[0],
            active_alphas=[2.0],
        )
        assert len(out_idxs) == n

    def test_positive_alpha_promotes_high_activation(self, fake_activations):
        n = 20
        idxs = np.arange(n)
        dists = np.ones(n, dtype=np.float32)
        _, out_idxs = _sae_activation_rerank(
            idxs, dists, fake_activations,
            active_feature_ids=[0],
            active_alphas=[1.0],
        )
        assert fake_activations[out_idxs[0], 0] >= fake_activations[out_idxs[-1], 0]


class TestSearchWithSliders:
    def test_empty_slider_config_uses_plain_search(
        self, mock_faiss_index, query_emb, small_sae
    ):
        dists, idxs = search_with_sliders(
            index=mock_faiss_index,
            query_emb=query_emb,
            sae_model=small_sae,
            slider_config={},
            k=5,
        )
        assert len(idxs) == 5

    def test_with_slider_config_returns_k(
        self, mock_faiss_index, query_emb, small_sae
    ):
        dists, idxs = search_with_sliders(
            index=mock_faiss_index,
            query_emb=query_emb,
            sae_model=small_sae,
            slider_config={0: 1.0, 1: -0.5},
            k=5,
        )
        assert len(idxs) == 5

    def test_with_corpus_activations(
        self, mock_faiss_index, query_emb, small_sae, fake_activations, fake_embeddings
    ):
        dists, idxs = search_with_sliders(
            index=mock_faiss_index,
            query_emb=query_emb,
            sae_model=small_sae,
            slider_config={0: 1.0},
            k=5,
            corpus_activations=fake_activations,
            corpus_embeddings=fake_embeddings,
            mmr_lambda=0.7,
        )
        assert len(idxs) == 5

    def test_with_tied_weights_sae(
        self, mock_faiss_index, query_emb, small_sae_tied
    ):
        dists, idxs = search_with_sliders(
            index=mock_faiss_index,
            query_emb=query_emb,
            sae_model=small_sae_tied,
            slider_config={0: 0.5},
            k=3,
        )
        assert len(idxs) == 3
