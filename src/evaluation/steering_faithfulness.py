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
    """Feature activation in steered vs unsteered retrieval. >1.0 = steering pulls
    retrieval toward the concept. Pure embedding-space steering — passing
    corpus_activations would rerank by the feature we measure (tautological)."""
    from src.retrieval.query import search, search_with_sliders

    scores_base: list[float] = []
    scores_steered: list[float] = []

    for q in query_embs[:n_queries]:
        _, idxs_base = search(index, q, k=k)
        scores_base.append(float(corpus_activations[idxs_base, feature_id].mean()))

        _, idxs_steered = search_with_sliders(index, q, sae, {feature_id: alpha}, k=k)
        scores_steered.append(float(corpus_activations[idxs_steered, feature_id].mean()))

    baseline = float(np.mean(scores_base))
    steered = float(np.mean(scores_steered))
    if baseline < 1e-8 and steered < 1e-8:
        return 1.0
    return steered / (baseline + 1e-8)


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
    from src.retrieval.query import search, search_with_direction_sliders

    d = directions[direction_idx]
    d_norm = d / (np.linalg.norm(d) + 1e-8)

    n = len(directions)
    scores_base: list[float] = []
    scores_steered: list[float] = []

    for q in query_embs[:n_queries]:
        _, idxs_base = search(index, q, k=k)
        scores_base.append(float((corpus_embeddings[idxs_base] @ d_norm).mean()))

        alphas = [0.0] * n
        alphas[direction_idx] = alpha
        _, idxs_steered = search_with_direction_sliders(
            index, q, directions, alphas, k=k,
        )
        scores_steered.append(float((corpus_embeddings[idxs_steered] @ d_norm).mean()))

    # cosine lift, not ratio: cosine is signed, a ratio flips/explodes near zero
    return float(np.mean(scores_steered)) - float(np.mean(scores_base))


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
    return {
        fid: steering_faithfulness(
            sae, index, corpus_activations, query_embs, fid, alpha, k, n_queries
        )
        for fid in feature_ids
    }
