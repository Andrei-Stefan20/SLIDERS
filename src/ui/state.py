from dataclasses import dataclass, field

import faiss
import numpy as np

from src.encoders.dino_encoder import DINOEncoder
from src.models.sae import SparseAutoencoder


@dataclass
class AppState:
    dino: DINOEncoder
    sae: SparseAutoencoder
    index: faiss.Index
    sae_index: faiss.Index | None
    embeddings: np.ndarray
    activations: np.ndarray | None
    image_paths: list[str]
    feature_ids: list[int]
    feature_names: list[str]
    feature_descriptions: list[str]
    image_classes: list[str] = field(default_factory=list)

    class_directions: np.ndarray | None = None
    class_names: list[str] | None = None

    preview_top: list[str] | None = None
    preview_bottom: list[str] | None = None

    path_to_idx: dict[str, int] = field(default_factory=dict)
    feature_scales: np.ndarray | None = None
