"""Identifies the most and least activating images for each SAE feature."""

from pathlib import Path
from typing import NamedTuple

import numpy as np


class FeatureImages(NamedTuple):
    feature_id: int
    top_paths: list[Path]
    top_activations: list[float]
    bottom_paths: list[Path]
    bottom_activations: list[float]


def get_top_images(
    activations: np.ndarray,
    image_paths: list[Path | str],
    feature_id: int,
    k: int = 10,
) -> FeatureImages:
    feature_acts = activations[:, feature_id]
    top_idx = np.argsort(feature_acts)[::-1][:k].tolist()
    bottom_idx = np.argsort(feature_acts)[:k].tolist()
    paths = [Path(p) for p in image_paths]
    return FeatureImages(
        feature_id=feature_id,
        top_paths=[paths[i] for i in top_idx],
        top_activations=[float(feature_acts[i]) for i in top_idx],
        bottom_paths=[paths[i] for i in bottom_idx],
        bottom_activations=[float(feature_acts[i]) for i in bottom_idx],
    )


def rank_features_by_variance(activations: np.ndarray) -> list[int]:
    """Return feature indices sorted by activation variance. Dead features excluded."""
    variances = activations.var(axis=0)
    ranked = np.argsort(variances)[::-1]
    return [int(i) for i in ranked if variances[i] > 0]


