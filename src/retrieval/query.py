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
    alphas_arr = np.array(active_alphas, dtype=np.float32)
    sae_scores = (corpus_activations[indices][:, active_feature_ids] * alphas_arr).sum(axis=1)
    order = np.argsort(sae_scores)[::-1]
    # Returns relevance scores, not cosine distances; MMR uses them downstream.
    return sae_scores[order].astype(np.float32), indices[order]


def _mmr_rerank(
    indices: np.ndarray,
    distances: np.ndarray,
    corpus_embeddings: np.ndarray,
    mmr_lambda: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    if mmr_lambda >= 1.0 or len(indices) <= 1:
        return distances, indices

    embs = np.ascontiguousarray(corpus_embeddings[indices], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.where(norms > 1e-8, norms, 1.0)
    sim_matrix = embs @ embs.T

    selected: list[int] = []
    remaining = list(range(len(indices)))

    best_start = int(np.argmax(distances))
    selected.append(best_start)
    remaining.remove(best_start)

    while remaining:
        max_sim = sim_matrix[remaining][:, selected].max(axis=1)
        mmr_scores = mmr_lambda * distances[remaining] - (1.0 - mmr_lambda) * max_sim
        best_pos = int(np.argmax(mmr_scores))
        selected.append(remaining[best_pos])
        remaining.pop(best_pos)

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
    feature_scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not slider_config:
        return search(index, query_emb, k=k)

    feature_ids = list(slider_config.keys())
    raw_alphas = [slider_config[fid] for fid in feature_ids]
    if feature_scales is not None:
        alphas = [a * float(feature_scales[fid]) for a, fid in zip(raw_alphas, feature_ids)]
    else:
        alphas = list(raw_alphas)

    directions = sae_model.feature_directions(feature_ids).cpu().numpy()
    steered = steer_query(query_emb, directions, alphas)

    fetch_k = max(k * 3, 60) if (corpus_activations is not None or corpus_embeddings is not None) else k
    dists, idxs = search(index, steered, k=fetch_k)

    if sae_index is not None:
        import torch

        with torch.no_grad():
            acts = sae_model.encode(torch.from_numpy(query_emb.reshape(1, -1))).numpy()[0]
        for fid, a in zip(feature_ids, raw_alphas):
            acts[fid] = max(0.0, float(acts[fid]) + a)
        norm = np.linalg.norm(acts)
        acts_normed = acts / norm if norm > 1e-8 else acts
        sae_dists, sae_idxs = search(sae_index, acts_normed, k=fetch_k)

        if corpus_activations is not None and feature_ids:
            # Union both pools; the activation rerank below sets the real order.
            pool = list(dict.fromkeys([int(i) for i in idxs] + [int(i) for i in sae_idxs]))
            idxs = np.array(pool)
            dists = np.linspace(1.0, 0.0, num=len(pool), dtype=np.float32)
        else:
            rrf_k = 60.0
            scores: dict[int, float] = {}
            for rank, idx in enumerate(idxs):
                scores[int(idx)] = scores.get(int(idx), 0.0) + (1.0 - sae_index_weight) / (rrf_k + rank)
            for rank, idx in enumerate(sae_idxs):
                scores[int(idx)] = scores.get(int(idx), 0.0) + sae_index_weight / (rrf_k + rank)
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:fetch_k]
            idxs = np.array([i for i, _ in sorted_items])
            dists = np.array([v for _, v in sorted_items])

    if corpus_activations is not None and feature_ids:
        dists, idxs = _sae_activation_rerank(idxs, dists, corpus_activations, feature_ids, alphas)

    if corpus_embeddings is not None:
        dists, idxs = _mmr_rerank(idxs, dists, corpus_embeddings, mmr_lambda=mmr_lambda)

    return dists[:k], idxs[:k]
