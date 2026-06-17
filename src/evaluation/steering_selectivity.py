"""Steering selectivity: when a slider is pushed, does it move ONLY its own attribute?

Faithfulness shows the steered feature rises; this asks whether other features rise too.
A clean slider concentrates the activation increase on its own feature (on-target
fraction near 1); a entangled one drags many unrelated features along.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import faiss
    from src.models.sae import SparseAutoencoder


def steering_selectivity(
    sae: "SparseAutoencoder",
    index: "faiss.Index",
    corpus_activations: np.ndarray,
    query_embs: np.ndarray,
    feature_id: int,
    alpha: float = 2.0,
    k: int = 20,
    n_queries: int = 100,
) -> float:
    """On-target fraction: feature_id's mean activation increase in the steered top-k,
    divided by the total positive increase across all features. ~1.0 = selective slider,
    ~0 = the lift is spread over many other features."""
    from src.retrieval.query import search, search_with_sliders

    deltas = np.zeros(corpus_activations.shape[1], dtype=np.float64)
    n = 0
    for q in query_embs[:n_queries]:
        _, idxs_base = search(index, q, k=k)
        _, idxs_steer = search_with_sliders(index, q, sae, {feature_id: alpha}, k=k)
        deltas += corpus_activations[idxs_steer].mean(0) - corpus_activations[idxs_base].mean(0)
        n += 1
    if n == 0:
        return float("nan")
    deltas /= n
    on_target = max(float(deltas[feature_id]), 0.0)
    total_positive = float(np.clip(deltas, 0.0, None).sum())
    if total_positive < 1e-12:
        return float("nan")
    return on_target / total_positive


def batch_steering_selectivity(
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
        fid: steering_selectivity(
            sae, index, corpus_activations, query_embs, fid, alpha, k, n_queries
        )
        for fid in feature_ids
    }
