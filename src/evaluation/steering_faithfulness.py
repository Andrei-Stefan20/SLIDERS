"""Measures whether steering sliders actually shift retrieval results in the expected direction."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import faiss
    from src.models.sae import SparseAutoencoder


def steering_faithfulness(
    sae: "SparseAutoencoder",
    index: "faiss.Index",
    corpus_activations: np.ndarray,
    query_embs: np.ndarray,
    feature_id: int,
    alpha: float = 2.0,
    k: int = 20,
    n_queries: int = 100,
) -> float:
    """Returns mean_steered_activation / corpus_mean_activation. Above 1.0 means steering works."""
    from src.retrieval.query import search_with_sliders

    corpus_mean = float(corpus_activations[:, feature_id].mean())
    if corpus_mean < 1e-8:
        return 1.0

    scores_steered: list[float] = []

    for q in query_embs[:n_queries]:
        _, idxs_steered = search_with_sliders(
            index, q, sae, {feature_id: alpha}, k=k,
            corpus_activations=corpus_activations,
        )
        act_steered = float(corpus_activations[idxs_steered, feature_id].mean())
        scores_steered.append(act_steered)

    return float(np.mean(scores_steered)) / corpus_mean


def direction_steering_faithfulness(
    index: "faiss.Index",
    corpus_embeddings: np.ndarray,
    directions: np.ndarray,
    direction_idx: int,
    query_embs: np.ndarray,
    alpha: float = 2.0,
    k: int = 20,
    n_queries: int = 100,
) -> float:
    """Same as steering_faithfulness but for difference-in-means directions instead of SAE features."""
    from src.retrieval.query import search_with_direction_sliders

    d = directions[direction_idx]
    d_norm = d / (np.linalg.norm(d) + 1e-8)

    corpus_mean_sim = float((corpus_embeddings @ d_norm).mean())
    if abs(corpus_mean_sim) < 1e-8:
        return 1.0

    n = len(directions)
    scores_steered: list[float] = []

    for q in query_embs[:n_queries]:
        alphas = [0.0] * n
        alphas[direction_idx] = alpha
        _, idxs_steered = search_with_direction_sliders(
            index, q, directions, alphas, k=k,
        )
        sim_steered = float((corpus_embeddings[idxs_steered] @ d_norm).mean())
        scores_steered.append(sim_steered)

    return float(np.mean(scores_steered)) / abs(corpus_mean_sim)


def batch_steering_faithfulness(
    sae: "SparseAutoencoder",
    index: "faiss.Index",
    corpus_activations: np.ndarray,
    query_embs: np.ndarray,
    feature_ids: list[int],
    alpha: float = 2.0,
    k: int = 20,
    n_queries: int = 100,
) -> dict[int, float]:
    """Compute steering faithfulness for multiple features."""
    return {
        fid: steering_faithfulness(
            sae, index, corpus_activations, query_embs, fid, alpha, k, n_queries
        )
        for fid in feature_ids
    }
