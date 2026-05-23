from pathlib import PurePath
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import faiss
    from src.models.sae import SparseAutoencoder

from src.evaluation.recall_at_k import recall_at_k


def targeted_recall_delta(
    sae: "SparseAutoencoder",
    index: "faiss.Index",
    corpus_activations: np.ndarray,
    query_embs: np.ndarray,
    image_paths: list[str],
    feature_id: int,
    dominant_class: str,
    alpha: float = 2.0,
    k: int = 10,
    n_queries: int = 100,
) -> dict:
    from src.retrieval.query import search, search_with_sliders

    class_indices = [
        i for i, p in enumerate(image_paths)
        if PurePath(p).parent.name == dominant_class
    ]
    if len(class_indices) < 2:
        return {
            "feature_id": feature_id, "dominant_class": dominant_class,
            "baseline_recall": 0.0, "steered_recall": 0.0, "delta_recall": 0.0,
        }

    rng = np.random.default_rng(0)
    qidxs = rng.choice(class_indices, size=min(n_queries, len(class_indices)), replace=False)

    base_recalls: list[float] = []
    steer_recalls: list[float] = []

    for qi in qidxs:
        q = query_embs[qi]
        relevant = [j for j in class_indices if j != qi]

        _, idxs_base = search(index, q, k=k + 1)
        filtered_base = [int(i) for i in idxs_base if i != qi][:k]

        _, idxs_steer = search_with_sliders(
            index, q, sae, {feature_id: alpha}, k=k + 1,
            corpus_activations=corpus_activations,
        )
        filtered_steer = [int(i) for i in idxs_steer if i != qi][:k]

        base_recalls.append(recall_at_k(filtered_base, relevant, k))
        steer_recalls.append(recall_at_k(filtered_steer, relevant, k))

    base_mean = float(np.mean(base_recalls))
    steer_mean = float(np.mean(steer_recalls))
    return {
        "feature_id": feature_id,
        "dominant_class": dominant_class,
        "baseline_recall": base_mean,
        "steered_recall": steer_mean,
        "delta_recall": steer_mean - base_mean,
    }


def batch_targeted_recall(
    sae: "SparseAutoencoder",
    index: "faiss.Index",
    corpus_activations: np.ndarray,
    query_embs: np.ndarray,
    image_paths: list[str],
    mono_scores: list[dict],
    alpha: float = 2.0,
    k: int = 10,
    n_queries: int = 100,
) -> list[dict]:
    return [
        targeted_recall_delta(
            sae, index, corpus_activations, query_embs, image_paths,
            m["feature_id"], m["dominant_class"], alpha, k, n_queries,
        )
        for m in mono_scores
    ]
