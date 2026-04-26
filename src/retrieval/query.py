"""High-level query interface over a FAISS index."""

import faiss
import numpy as np

from src.models.sae import SparseAutoencoder
from src.retrieval.steering import steer_query


def search_with_direction_sliders(
    index: faiss.Index,
    query_emb: np.ndarray,
    directions: np.ndarray,
    slider_values: list[float],
    k: int = 10,
    corpus_embeddings: np.ndarray | None = None,
    mmr_lambda: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Steer query using pre-computed directions (e.g. difference-in-means).

    Unlike search_with_sliders, this requires no SAE model. Directions are
    passed directly in the original embedding space.

    Args:
        directions: Shape (N, D), one direction per slider.
        slider_values: Alpha for each direction; zero entries are skipped.
    """
    active = [(d, a) for d, a in zip(directions, slider_values) if abs(a) > 1e-6]
    if not active:
        return search(index, query_emb, k=k)

    active_dirs = np.stack([d for d, _ in active])
    active_alphas = [a for _, a in active]

    steered = steer_query(query_emb, active_dirs, active_alphas)
    fetch_k = max(k * 3, 60) if corpus_embeddings is not None else k
    dists, idxs = search(index, steered, k=fetch_k)

    if corpus_embeddings is not None:
        dists, idxs = _mmr_rerank(idxs, dists, corpus_embeddings, mmr_lambda=mmr_lambda)

    return dists[:k], idxs[:k]


def search(
    index: faiss.Index,
    query_emb: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    query = np.ascontiguousarray(query_emb, dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(query, k)
    return distances[0], indices[0]


def _sae_activation_rerank(
    indices: np.ndarray,
    distances: np.ndarray,
    corpus_activations: np.ndarray,
    active_feature_ids: list[int],
    active_alphas: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Re-rank by dot product between corpus SAE activations and active slider weights.

    Fixes the mismatch between the DINOv2 search space and the SAE feature space:
    FAISS finds globally similar images, but this promotes images that actually
    have high activation on the requested features.
    """
    alphas_arr = np.array(active_alphas, dtype=np.float32)
    sae_scores = (corpus_activations[indices][:, active_feature_ids] * alphas_arr).sum(axis=1)
    order = np.argsort(sae_scores)[::-1]
    return distances[order], indices[order]


def _mmr_rerank(
    indices: np.ndarray,
    distances: np.ndarray,
    corpus_embeddings: np.ndarray,
    mmr_lambda: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximal Marginal Relevance re-ranking for diversity (Carbonell & Goldstein, 1998).

    Selects: argmax [ lambda * relevance - (1-lambda) * max_sim_to_selected ]
    mmr_lambda=1.0 is pure relevance, mmr_lambda=0.0 is pure diversity.
    """
    if mmr_lambda >= 1.0 or len(indices) <= 1:
        return distances, indices

    embs = np.ascontiguousarray(corpus_embeddings[indices], dtype=np.float32)
    sim_matrix = embs @ embs.T

    selected: list[int] = []
    remaining = list(range(len(indices)))

    best_start = int(np.argmax(distances))
    selected.append(best_start)
    remaining.remove(best_start)

    while remaining:
        max_sim = sim_matrix[remaining][:, selected].max(axis=1)
        mmr_scores = mmr_lambda * distances[remaining] - (1.0 - mmr_lambda) * max_sim
        best = remaining[int(np.argmax(mmr_scores))]
        selected.append(best)
        remaining.remove(best)

    order = np.array(selected)
    return distances[order], indices[order]


def search_with_sliders(
    index: faiss.Index,
    query_emb: np.ndarray,
    sae_model: SparseAutoencoder,
    slider_config: dict[int, float],
    k: int = 10,
    corpus_activations: np.ndarray | None = None,
    corpus_embeddings: np.ndarray | None = None,
    mmr_lambda: float = 0.7,
    sae_index: faiss.Index | None = None,
    sae_index_weight: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Steer the query, search FAISS, then optionally apply two re-ranking passes.

    Re-ranking passes (applied in order when their inputs are provided):
      1. SAE activation re-ranking (corpus_activations), for precision
      2. MMR diversity re-ranking (corpus_embeddings), for variety

    When sae_index is provided, results from both the primary and SAE-space
    indices are merged before re-ranking.
    """
    if not slider_config:
        return search(index, query_emb, k=k)

    if sae_model.tied_weights:
        decoder_weight = sae_model.encoder.weight.detach().cpu().numpy().T
    else:
        decoder_weight = sae_model.decoder.weight.detach().cpu().numpy()

    feature_ids = list(slider_config.keys())
    alphas = [slider_config[fid] for fid in feature_ids]
    directions = decoder_weight[:, feature_ids].T  # (n_sliders, input_dim)
    steered = steer_query(query_emb, directions, alphas)

    fetch_k = max(k * 3, 60) if (corpus_activations is not None or corpus_embeddings is not None) else k
    dists, idxs = search(index, steered, k=fetch_k)

    if sae_index is not None:
        import torch
        with torch.no_grad():
            acts = sae_model.encode(torch.from_numpy(steered.reshape(1, -1))).numpy()[0]
        norm = np.linalg.norm(acts)
        acts_normed = acts / norm if norm > 1e-8 else acts
        sae_dists, sae_idxs = search(sae_index, acts_normed, k=fetch_k)

        scores: dict[int, float] = {}
        max_d = max(dists.max(), 1e-8)
        max_s = max(sae_dists.max(), 1e-8)
        for d, idx in zip(dists, idxs):
            scores[int(idx)] = scores.get(int(idx), 0.0) + (1.0 - sae_index_weight) * float(d) / max_d
        for s, idx in zip(sae_dists, sae_idxs):
            scores[int(idx)] = scores.get(int(idx), 0.0) + sae_index_weight * float(s) / max_s
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:fetch_k]
        idxs = np.array([i for i, _ in sorted_items])
        dists = np.array([v for _, v in sorted_items])

    if corpus_activations is not None and feature_ids:
        dists, idxs = _sae_activation_rerank(idxs, dists, corpus_activations, feature_ids, alphas)

    if corpus_embeddings is not None:
        dists, idxs = _mmr_rerank(idxs, dists, corpus_embeddings, mmr_lambda=mmr_lambda)

    return dists[:k], idxs[:k]
