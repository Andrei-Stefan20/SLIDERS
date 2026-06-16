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
    """Cosine lift (steered minus unsteered) of retrieved items toward direction.
    >0 = steering works. Lift not ratio: cosine is signed, a ratio flips/explodes
    near zero. Same metric for any unit direction (PCA or SAE), so they compare."""
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

    return float(np.mean(scores_steer)) - float(np.mean(scores_base))


def evaluate_pca_baseline(
    index: "faiss.Index",
    corpus_embeddings: np.ndarray,
    query_embs: np.ndarray,
    n_components: int = 20,
    alpha: float = 2.0,
    k: int = 20,
    n_queries: int = 100,
) -> dict:
    """Per-component steering lift for the top PCA directions — the baseline
    evaluate_sae_vs_pca measures SAE features against with the same metric."""
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


def _sae_feature_direction(sae: "SparseAutoencoder", feature_id: int) -> np.ndarray:
    """A feature's unit-norm decoder column — the direction search_with_sliders steers along."""
    return sae.feature_directions([feature_id])[0].cpu().numpy()


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
    """SAE vs PCA steering, same cosine-lift metric for both. Headline is
    steering_advantage = SAE median lift - PCA median lift (>0 = SAE steers better
    than raw PCs). Median, since signed lifts can have outlier queries."""
    sae_per_feature = [
        _pca_direction_faithfulness(
            index, corpus_embeddings, _sae_feature_direction(sae, fid),
            query_embs, alpha, k, n_queries,
        )
        for fid in feature_ids
    ]
    sae_mean = float(np.mean(sae_per_feature)) if sae_per_feature else 0.0
    sae_median = float(np.median(sae_per_feature)) if sae_per_feature else 0.0

    pca_result = evaluate_pca_baseline(
        index, corpus_embeddings, query_embs, n_pca_components, alpha, k, n_queries
    )
    pca_mean = pca_result["pca_mean_faithfulness"]
    pca_median = float(np.median(pca_result["pca_per_component"]))

    return {
        "sae_mean_faithfulness": sae_mean,
        "sae_median_faithfulness": sae_median,
        "pca_mean_faithfulness": pca_mean,
        "pca_median_faithfulness": pca_median,
        "steering_advantage": sae_median - pca_median,
        "steering_advantage_mean": sae_mean - pca_mean,
        "sae_per_feature": dict(zip(feature_ids, sae_per_feature)),
        "pca_result": pca_result,
    }
