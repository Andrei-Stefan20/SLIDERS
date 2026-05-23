from pathlib import PurePath
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import faiss
    from src.models.sae import SparseAutoencoder

from src.evaluation.recall_at_k import (
    average_precision,
    precision_at_k,
    recall_at_k,
)


def _build_ground_truth(image_paths: list[str]) -> list[list[int]]:
    labels = [PurePath(p).parent.name for p in image_paths]
    label_to_indices: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(i)
    return [[j for j in label_to_indices[labels[i]] if j != i] for i in range(len(image_paths))]


def _score_results(
    retrieved_per_query: list[list[int]],
    ground_truth: list[list[int]],
    query_indices: list[int],
    k_values: tuple[int, ...],
) -> dict[str, float]:
    max_k = max(k_values)
    p, r, ap = {k: [] for k in k_values}, {k: [] for k in k_values}, []
    for retrieved, qi in zip(retrieved_per_query, query_indices):
        relevant = ground_truth[qi]
        for k in k_values:
            p[k].append(precision_at_k(retrieved, relevant, k))
            r[k].append(recall_at_k(retrieved, relevant, k))
        ap.append(average_precision(retrieved, relevant, max_k))
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"P@{k}"] = float(np.mean(p[k]))
        metrics[f"R@{k}"] = float(np.mean(r[k]))
    metrics[f"mAP@{max_k}"] = float(np.mean(ap))
    return metrics


def compare_retrieval_methods(
    index: "faiss.Index",
    norm_embs: np.ndarray,
    image_paths: list[str],
    sae: "SparseAutoencoder",
    corpus_activations: np.ndarray,
    query_indices: list[int],
    k_values: tuple[int, ...] = (5, 10),
    steer_alpha: float = 2.0,
    n_steer_features: int = 5,
    n_pca_components: int = 5,
) -> dict[str, dict[str, float]]:
    import torch
    from src.evaluation.ablation import pca_directions
    from src.retrieval.query import search, search_with_sliders
    from src.retrieval.steering import steer_query

    ground_truth = _build_ground_truth(image_paths)
    max_k = max(k_values)
    fetch_k = max_k + 1

    all_pca_dirs = pca_directions(norm_embs, max(n_pca_components, 50))

    unsteered, pca_steered, sae_steered = [], [], []

    for qi in query_indices:
        q = norm_embs[qi]

        _, idxs = search(index, q, k=fetch_k)
        unsteered.append([int(i) for i in idxs if i != qi][:max_k])

        projections = np.abs(all_pca_dirs @ q)
        top_pca = np.argsort(projections)[::-1][:n_pca_components]
        q_pca = steer_query(q, all_pca_dirs[top_pca], [steer_alpha] * n_pca_components)
        _, idxs = search(index, q_pca, k=fetch_k)
        pca_steered.append([int(i) for i in idxs if i != qi][:max_k])

        with torch.no_grad():
            q_acts = sae.encode(torch.from_numpy(q.reshape(1, -1))).numpy()[0]
        top_fids = np.argsort(q_acts)[::-1][:n_steer_features].tolist()
        slider_config = {fid: steer_alpha for fid in top_fids}
        _, idxs = search_with_sliders(
            index, q, sae, slider_config, k=fetch_k,
            corpus_activations=corpus_activations,
        )
        sae_steered.append([int(i) for i in idxs if i != qi][:max_k])

    return {
        "Unsteered (DINOv2)": _score_results(unsteered, ground_truth, query_indices, k_values),
        f"PCA steering (top-{n_pca_components})": _score_results(pca_steered, ground_truth, query_indices, k_values),
        f"SAE steering (top-{n_steer_features})": _score_results(sae_steered, ground_truth, query_indices, k_values),
    }


def print_comparison_table(results: dict[str, dict[str, float]]) -> None:
    methods = list(results.keys())
    if not methods:
        return
    metrics = list(results[methods[0]].keys())

    col_w = max(len(m) for m in methods) + 2
    header = f"  {'Method':{col_w}}" + "".join(f"  {m:>10}" for m in metrics)
    print(header)
    print("  " + "-" * (col_w + 12 * len(metrics)))
    for method, scores in results.items():
        row = f"  {method:{col_w}}" + "".join(f"  {scores[m]:>10.4f}" for m in metrics)
        print(row)
