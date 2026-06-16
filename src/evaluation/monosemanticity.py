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
    n_total_classes: int | None = None,
) -> dict:
    """Class purity of a feature's top-k images. 1.0 = all one class, 0.0 = spread
    evenly. Pass n_total_classes to normalise entropy by the dataset's class count
    (comparable across features); without it, falls back to the observed classes."""
    feature_acts = activations[:, feature_id]
    top_idx = np.argsort(feature_acts)[::-1][:k]

    classes = [PurePath(image_paths[i]).parent.name for i in top_idx]
    counts = Counter(classes)

    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log(p + 1e-8) for p in probs)
    n_ref = n_total_classes if n_total_classes and n_total_classes > 1 else len(counts)
    max_entropy = math.log(n_ref) if n_ref > 1 else 1.0
    purity = 1.0 if max_entropy <= 0 else 1.0 - entropy / max_entropy
    purity = float(max(0.0, min(1.0, purity)))
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
    n_total_classes: int | None = None,
) -> list[dict]:
    """Compute monosemanticity scores for multiple features."""
    return [
        monosemanticity_score(activations, image_paths, fid, k, n_total_classes)
        for fid in feature_ids
    ]


def mean_purity(scores: list[dict]) -> float:
    """Mean purity score across a list of monosemanticity results."""
    if not scores:
        return 0.0
    return float(np.mean([s["purity_score"] for s in scores]))


def n_distinct_classes(image_paths: list[str]) -> int:
    """Number of distinct parent-folder class labels in the corpus."""
    return len({PurePath(p).parent.name for p in image_paths})


def shuffled_label_purity_baseline(
    activations: np.ndarray,
    image_paths: list[str],
    feature_ids: list[int],
    k: int = 20,
    n_total_classes: int | None = None,
    n_shuffles: int = 5,
    seed: int = 0,
) -> list[float]:
    """Chance-level purity under shuffled labels — one value per shuffle. Real
    purity only means something as the gap above this."""
    rng = np.random.default_rng(seed)
    paths = list(image_paths)
    out: list[float] = []
    for _ in range(n_shuffles):
        # shuffle paths, not labels: monosemanticity_score reads the class off the path
        permuted = [paths[j] for j in rng.permutation(len(paths))]
        scores = batch_monosemanticity(
            activations, permuted, feature_ids, k, n_total_classes
        )
        out.append(mean_purity(scores))
    return out
