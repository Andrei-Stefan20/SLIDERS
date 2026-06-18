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
    direction_sims: np.ndarray | None = None,
) -> list[int]:
    """Greedy MMR selection. Diversity penalty is the max similarity to the already
    selected: decoder-direction cosine when direction_sims (unit-norm candidate rows)
    is given, else activation-pattern correlation. Directions catch near-duplicate
    sliders that correlation misses."""
    n_samples = acts_normed.shape[0]
    selected: list[int] = []
    selected_local: list[int] = []

    first = int(np.argmax(scores_norm))
    selected.append(int(candidates[first]))
    selected_local.append(first)

    while len(selected) < n_features and len(selected) < len(candidates):
        if direction_sims is not None:
            sims = np.abs(direction_sims @ direction_sims[selected_local].T)
        else:
            sims = np.abs((acts_normed.T @ acts_normed[:, selected_local]) / n_samples)
        max_sim = sims.max(axis=1)
        scores = lambda_mmr * scores_norm - (1 - lambda_mmr) * max_sim
        for i in selected_local:
            scores[i] = -np.inf
        best_idx = int(np.argmax(scores))
        selected.append(int(candidates[best_idx]))
        selected_local.append(best_idx)

    return selected


def _candidate_direction_sims(
    directions: np.ndarray | None, candidates: np.ndarray
) -> np.ndarray | None:
    """Unit-norm decoder rows for the candidates, or None if directions not given."""
    if directions is None:
        return None
    d = directions[candidates].astype(np.float32)
    return d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)


def _semantic_fingerprints(
    activations: np.ndarray, embeddings: np.ndarray, candidates: np.ndarray, k: int = 20
) -> np.ndarray:
    """Per-candidate semantic signature: unit-norm mean embedding of its top-k activating
    images. Two features that fire on visually similar images (and would get the same VLM
    name) have similar fingerprints even when their decoder directions or activation
    patterns differ — this is the signal that catches name-level duplicate sliders."""
    fps = np.zeros((len(candidates), embeddings.shape[1]), dtype=np.float32)
    kk = min(k, activations.shape[0])
    for i, f in enumerate(candidates):
        col = activations[:, f]
        top = np.argpartition(col, -kk)[-kk:]
        m = embeddings[top].mean(0)
        n = np.linalg.norm(m)
        fps[i] = m / n if n > 1e-8 else m
    return fps


def _prepare_diversity(
    activations: np.ndarray,
    candidates: np.ndarray,
    scores_norm: np.ndarray,
    n_features: int,
    directions: np.ndarray | None,
    embeddings: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Pick the MMR diversity signal. Priority: semantic fingerprints (embeddings) >
    decoder directions > activation correlation. With embeddings, shortlist to a
    score-ranked pool first so the per-feature fingerprint stays cheap."""
    if embeddings is not None:
        pool = np.argsort(scores_norm)[::-1][:max(n_features * 5, 60)]
        candidates = candidates[pool]
        scores_norm = scores_norm[pool]
        sims = _semantic_fingerprints(activations, embeddings, candidates)
    elif directions is not None:
        sims = _candidate_direction_sims(directions, candidates)
    else:
        sims = None
    acts_normed = _sample_and_normalize(activations, candidates)
    return candidates, scores_norm, acts_normed, sims


def rank_by_selectivity_mmr(
    activations: np.ndarray,
    class_labels: np.ndarray,
    n_features: int = 20,
    sparsity_min: float = 0.90,
    sparsity_max: float = 0.995,
    lambda_mmr: float = 0.5,
    directions: np.ndarray | None = None,
    embeddings: np.ndarray | None = None,
) -> list[int]:
    """Select diverse features ranked by class selectivity + MMR.

    Selectivity = max class mean activation / global mean.
    High selectivity means the feature fires strongly on specific classes.
    MMR then ensures diversity; pass `embeddings` to diversify by semantic fingerprint
    (mean top-image embedding) or `directions` (decoder columns) instead of activation
    correlation.
    """
    sparsity = compute_sparsity(activations)
    candidates = np.where(
        (sparsity >= sparsity_min) & (sparsity <= sparsity_max)
    )[0]
    if len(candidates) == 0:
        candidates = np.arange(activations.shape[1])

    selectivity = compute_selectivity(activations[:, candidates], class_labels)
    scores_norm = selectivity / (selectivity.max() + 1e-8)
    candidates, scores_norm, acts_normed, sims = _prepare_diversity(
        activations, candidates, scores_norm, n_features, directions, embeddings
    )
    return _greedy_mmr(candidates, scores_norm, acts_normed, n_features, lambda_mmr, sims)


def rank_diverse_mmr(
    activations: np.ndarray,
    n_features: int = 20,
    sparsity_min: float = 0.90,
    sparsity_max: float = 0.995,
    lambda_mmr: float = 0.5,
    directions: np.ndarray | None = None,
    embeddings: np.ndarray | None = None,
) -> list[int]:
    """Select diverse features via MMR with sparsity pre-filtering.

    1. Filter features by sparsity in [sparsity_min, sparsity_max].
    2. Rank survivors by variance.
    3. Greedy MMR: pick next feature maximising
       lambda * variance_norm - (1 - lambda) * max_sim_with_selected.

    The diversity term defaults to activation correlation. Pass `embeddings` to diversify
    by **semantic fingerprint** (mean top-image embedding) instead — this removes sliders
    that fire on visually similar images and would get the same VLM name, which directional
    or activation-correlation diversity miss. `directions` (decoder columns) is an
    alternative geometric signal.
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
    candidates, scores_norm, acts_normed, sims = _prepare_diversity(
        activations, candidates, scores_norm, n_features, directions, embeddings
    )
    return _greedy_mmr(candidates, scores_norm, acts_normed, n_features, lambda_mmr, sims)
