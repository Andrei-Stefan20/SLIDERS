"""Rank SAE features for naming using diverse selection strategies."""

import numpy as np


def compute_sparsity(activations: np.ndarray) -> np.ndarray:
    """Fraction of zeros per feature (higher = more sparse = more monosemantic)."""
    return (activations == 0).mean(axis=0)


def compute_selectivity(
    activations: np.ndarray,
    class_labels: np.ndarray,
) -> np.ndarray:
    """Max class mean activation / mean over all classes."""
    n_features = activations.shape[1]
    classes = np.unique(class_labels)
    class_means = np.zeros((len(classes), n_features))
    for i, c in enumerate(classes):
        mask = class_labels == c
        if mask.sum() > 0:
            class_means[i] = activations[mask].mean(axis=0)
    overall_mean = class_means.mean(axis=0) + 1e-8
    return class_means.max(axis=0) / overall_mean


def _sample_and_normalize(activations: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    n_samples = min(activations.shape[0], 5000)
    rng = np.random.default_rng(seed=42)
    sample_idx = rng.choice(activations.shape[0], n_samples, replace=False)
    acts = activations[sample_idx][:, candidates]
    acts = acts - acts.mean(axis=0, keepdims=True)
    return acts / (acts.std(axis=0, keepdims=True) + 1e-8)


def _greedy_mmr(
    candidates: np.ndarray,
    scores_norm: np.ndarray,
    acts_normed: np.ndarray,
    n_features: int,
    lambda_mmr: float,
) -> list[int]:
    n_samples = acts_normed.shape[0]
    selected: list[int] = []
    selected_local: list[int] = []

    first = int(np.argmax(scores_norm))
    selected.append(int(candidates[first]))
    selected_local.append(first)

    while len(selected) < n_features and len(selected) < len(candidates):
        dots = (acts_normed.T @ acts_normed[:, selected_local]) / n_samples
        max_corr = np.abs(dots).max(axis=1)
        scores = lambda_mmr * scores_norm - (1 - lambda_mmr) * max_corr
        for i in selected_local:
            scores[i] = -np.inf
        best_idx = int(np.argmax(scores))
        selected.append(int(candidates[best_idx]))
        selected_local.append(best_idx)

    return selected


def rank_by_selectivity_mmr(
    activations: np.ndarray,
    class_labels: np.ndarray,
    n_features: int = 20,
    sparsity_min: float = 0.90,
    sparsity_max: float = 0.995,
    lambda_mmr: float = 0.5,
) -> list[int]:
    """Select diverse features ranked by class selectivity + MMR.

    Selectivity = max class mean activation / global mean.
    High selectivity means the feature fires strongly on specific classes.
    MMR then ensures diversity among the selected.
    """
    sparsity = compute_sparsity(activations)
    candidates = np.where(
        (sparsity >= sparsity_min) & (sparsity <= sparsity_max)
    )[0]
    if len(candidates) == 0:
        candidates = np.arange(activations.shape[1])

    selectivity = compute_selectivity(activations[:, candidates], class_labels)
    scores_norm = selectivity / (selectivity.max() + 1e-8)
    acts_normed = _sample_and_normalize(activations, candidates)
    return _greedy_mmr(candidates, scores_norm, acts_normed, n_features, lambda_mmr)


def rank_diverse_mmr(
    activations: np.ndarray,
    n_features: int = 20,
    sparsity_min: float = 0.90,
    sparsity_max: float = 0.995,
    lambda_mmr: float = 0.5,
) -> list[int]:
    """Select diverse features via MMR with sparsity pre-filtering.

    1. Filter features by sparsity in [sparsity_min, sparsity_max].
    2. Rank survivors by variance.
    3. Greedy MMR: pick next feature maximising
       lambda * variance_norm - (1 - lambda) * max_corr_with_selected.
    """
    sparsity = compute_sparsity(activations)
    candidates = np.where(
        (sparsity >= sparsity_min) & (sparsity <= sparsity_max)
    )[0]

    if len(candidates) == 0:
        variance = activations.var(axis=0)
        return [int(i) for i in np.argsort(variance)[::-1][:n_features]]

    variance = activations[:, candidates].var(axis=0)
    scores_norm = variance / (variance.max() + 1e-8)
    acts_normed = _sample_and_normalize(activations, candidates)
    return _greedy_mmr(candidates, scores_norm, acts_normed, n_features, lambda_mmr)
