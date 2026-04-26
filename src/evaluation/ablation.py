"""PCA baseline ablation, compares SAE steering against principal component directions."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import faiss
    from src.models.sae import SparseAutoencoder


def pca_directions(embeddings: np.ndarray, n_components: int = 20) -> np.ndarray:
    """Compute top-n PCA directions from embeddings using SVD.

    Args:
        embeddings: Shape (N, D), should be zero-meaned or normalised.
        n_components: Number of principal components to return.

    Returns:
        Shape (n_components, D), each row is a unit-norm direction.
    """
    centered = embeddings - embeddings.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[:n_components].astype(np.float32)


def _pca_direction_faithfulness(
    index: "faiss.Index",
    corpus_embeddings: np.ndarray,
    direction: np.ndarray,
    query_embs: np.ndarray,
    alpha: float = 2.0,
    k: int = 20,
    n_queries: int = 100,
) -> float:
    from src.retrieval.query import search
    from src.retrieval.steering import steer_query

    scores_base: list[float] = []
    scores_steer: list[float] = []

    direction = direction / (np.linalg.norm(direction) + 1e-8)

    for q in query_embs[:n_queries]:
        _, idxs_base = search(index, q, k=k)
        sim_base = float((corpus_embeddings[idxs_base] @ direction).mean())

        q_steered = steer_query(q, direction[np.newaxis, :], [alpha])
        _, idxs_steer = search(index, q_steered, k=k)
        sim_steer = float((corpus_embeddings[idxs_steer] @ direction).mean())

        scores_base.append(sim_base)
        scores_steer.append(sim_steer)

    baseline = float(np.mean(scores_base))
    steered = float(np.mean(scores_steer))
    return steered / (baseline + 1e-8)


def evaluate_pca_baseline(
    index: "faiss.Index",
    corpus_embeddings: np.ndarray,
    query_embs: np.ndarray,
    n_components: int = 20,
    alpha: float = 2.0,
    k: int = 20,
    n_queries: int = 100,
) -> dict:
    """Compute mean steering faithfulness using PCA directions as a baseline.

    Compare this number against the SAE steering faithfulness score:
    - SAE score >> PCA score: SAE adds interpretable steering value
    - SAE score ~= PCA score: SAE is no better than linear PCA decomposition

    Returns:
        Dict with keys: pca_mean_faithfulness, pca_per_component (list of floats),
        n_components, explained_variance_ratio.
    """
    directions = pca_directions(corpus_embeddings, n_components)

    centered = corpus_embeddings - corpus_embeddings.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    total_var = float((s ** 2).sum())
    explained = [(float(sv ** 2) / total_var) for sv in s[:n_components]]

    per_component = [
        _pca_direction_faithfulness(
            index, corpus_embeddings, directions[i], query_embs, alpha, k, n_queries
        )
        for i in range(n_components)
    ]

    return {
        "pca_mean_faithfulness": float(np.mean(per_component)),
        "pca_per_component": per_component,
        "n_components": n_components,
        "explained_variance_ratio": explained,
        "cumulative_variance": float(sum(explained)),
    }


def evaluate_sae_vs_pca(
    sae: "SparseAutoencoder",
    index: "faiss.Index",
    corpus_embeddings: np.ndarray,
    corpus_activations: np.ndarray,
    query_embs: np.ndarray,
    feature_ids: list[int],
    n_pca_components: int = 20,
    alpha: float = 2.0,
    k: int = 20,
    n_queries: int = 100,
) -> dict:
    """Full ablation: SAE steering faithfulness vs PCA baseline.

    Returns:
        Dict with sae_mean_faithfulness, pca_mean_faithfulness,
        improvement_ratio (SAE/PCA), and per-feature details.
    """
    from src.evaluation.steering_faithfulness import batch_steering_faithfulness

    sae_scores = batch_steering_faithfulness(
        sae, index, corpus_activations, query_embs,
        feature_ids, alpha, k, n_queries,
    )
    sae_mean = float(np.mean(list(sae_scores.values()))) if sae_scores else 0.0

    pca_result = evaluate_pca_baseline(
        index, corpus_embeddings, query_embs, n_pca_components, alpha, k, n_queries
    )
    pca_mean = pca_result["pca_mean_faithfulness"]

    return {
        "sae_mean_faithfulness": sae_mean,
        "pca_mean_faithfulness": pca_mean,
        "improvement_ratio": sae_mean / (pca_mean + 1e-8),
        "sae_per_feature": sae_scores,
        "pca_result": pca_result,
    }
