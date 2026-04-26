"""Measures how monosemantic (single-concept) each SAE feature is."""

import math
from collections import Counter
from pathlib import PurePath

import numpy as np


def monosemanticity_score(
    activations: np.ndarray,
    image_paths: list[str],
    feature_id: int,
    k: int = 20,
) -> dict:
    """Compute class-purity score for one SAE feature.

    A monosemantic feature activates predominantly on one class.
    Score 1.0 = all top-k images from the same class.
    Score 0.0 = top-k images distributed uniformly across all classes.

    Args:
        activations: Shape (N, hidden_dim).
        image_paths: Length-N list of image path strings.
        feature_id: Which feature to score.
        k: How many top-activating images to examine.

    Returns:
        Dict with keys: feature_id, purity_score, dominant_class,
        dominant_class_fraction, n_classes_in_top_k, entropy.
    """
    feature_acts = activations[:, feature_id]
    top_idx = np.argsort(feature_acts)[::-1][:k]

    classes = [PurePath(image_paths[i]).parent.name for i in top_idx]
    counts = Counter(classes)

    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log(p + 1e-8) for p in probs)
    max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
    purity = 1.0 if max_entropy <= 0 else 1.0 - entropy / max_entropy
    dominant_class, dominant_count = counts.most_common(1)[0]

    return {
        "feature_id": feature_id,
        "purity_score": float(purity),
        "dominant_class": dominant_class,
        "dominant_class_fraction": float(dominant_count / total),
        "n_classes_in_top_k": len(counts),
        "entropy": entropy,
    }


def batch_monosemanticity(
    activations: np.ndarray,
    image_paths: list[str],
    feature_ids: list[int],
    k: int = 20,
) -> list[dict]:
    """Compute monosemanticity scores for multiple features."""
    return [
        monosemanticity_score(activations, image_paths, fid, k)
        for fid in feature_ids
    ]


def mean_purity(scores: list[dict]) -> float:
    """Mean purity score across a list of monosemanticity results."""
    if not scores:
        return 0.0
    return float(np.mean([s["purity_score"] for s in scores]))
