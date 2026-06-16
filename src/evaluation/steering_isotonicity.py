from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import faiss
    from src.models.sae import SparseAutoencoder


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    cov = np.corrcoef(rx, ry)
    return float(cov[0, 1]) if not np.isnan(cov[0, 1]) else 0.0


def steering_isotonicity(
    sae: "SparseAutoencoder",
    index: "faiss.Index",
    corpus_activations: np.ndarray,
    query_embs: np.ndarray,
    feature_id: int,
    alphas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
    k: int = 20,
    n_queries: int = 50,
) -> dict:
    from src.retrieval.query import search_with_sliders

    corpus_mean = float(corpus_activations[:, feature_id].mean())
    if corpus_mean < 1e-8:
        return {"feature_id": feature_id, "spearman_rho": 1.0, "alpha_means": {}}

    rng = np.random.default_rng(42)
    queries = query_embs[
        rng.choice(len(query_embs), size=min(n_queries, len(query_embs)), replace=False)
    ]

    alpha_means: dict[float, float] = {}
    for alpha in alphas:
        mean_acts = []
        for q in queries:
            # pure steering: a corpus_activations rerank would force monotonicity
            _, idxs = search_with_sliders(index, q, sae, {feature_id: alpha}, k=k)
            mean_acts.append(float(corpus_activations[idxs, feature_id].mean()))
        alpha_means[alpha] = float(np.mean(mean_acts)) / corpus_mean

    xs = np.array(list(alpha_means.keys()))
    ys = np.array(list(alpha_means.values()))

    return {
        "feature_id": feature_id,
        "spearman_rho": _spearman_rho(xs, ys),
        "alpha_means": alpha_means,
    }


def batch_steering_isotonicity(
    sae: "SparseAutoencoder",
    index: "faiss.Index",
    corpus_activations: np.ndarray,
    query_embs: np.ndarray,
    feature_ids: list[int],
    alphas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
    k: int = 20,
    n_queries: int = 50,
) -> list[dict]:
    return [
        steering_isotonicity(sae, index, corpus_activations, query_embs, fid, alphas, k, n_queries)
        for fid in feature_ids
    ]
