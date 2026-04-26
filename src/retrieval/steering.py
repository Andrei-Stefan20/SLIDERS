"""Query steering via SAE feature directions (the SLIDERS mechanism)."""

import numpy as np

from src.utils.exceptions import EmbeddingDimensionMismatch, InvalidSliderConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


def steer_query(
    query_emb: np.ndarray,
    directions: np.ndarray,
    alphas: list[float],
) -> np.ndarray:
    """Return q' = normalize(q + sum_i alpha_i * direction_i)."""
    if query_emb.ndim != 1:
        raise ValueError(f"query_emb must be 1-D, got shape {query_emb.shape}")

    if directions.ndim != 2:
        raise ValueError(f"directions must be 2-D, got shape {directions.shape}")

    if directions.shape[1] != query_emb.shape[0]:
        raise EmbeddingDimensionMismatch(
            f"directions.shape[1]={directions.shape[1]} != "
            f"query_emb.shape[0]={query_emb.shape[0]}"
        )

    if len(alphas) != directions.shape[0]:
        raise InvalidSliderConfig(
            f"len(alphas)={len(alphas)} != directions.shape[0]={directions.shape[0]}"
        )

    alphas_arr = np.asarray(alphas, dtype=np.float32)
    steered = query_emb.astype(np.float32) + (alphas_arr[:, None] * directions).sum(axis=0)

    norm = np.linalg.norm(steered)
    if norm < 1e-8:
        logger.warning("Steered query norm ~0 (directions cancelled out), returning unsteered query.")
        return query_emb.astype(np.float32)
    return steered / norm
