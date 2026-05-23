from collections import Counter
from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.encoders.dino_encoder import DINO_TRANSFORM
from src.retrieval.query import search, search_with_direction_sliders, search_with_sliders
from src.ui.state import AppState
from src.utils.io import normalize_embeddings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    image: Image.Image
    path: str


class RetrievalService:

    def __init__(self, state: AppState) -> None:
        self._state = state
        self.last_majority_class: str | None = None

    def encode_image(self, query_image: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(query_image).convert("RGB")
        emb = (
            self._state.dino.encode(DINO_TRANSFORM(pil).unsqueeze(0))
            .squeeze(0)
            .numpy()
        )
        return normalize_embeddings(emb.reshape(1, -1)).squeeze(0)

    def encode_image_raw(self, query_image: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(query_image).convert("RGB")
        return (
            self._state.dino.encode(DINO_TRANSFORM(pil).unsqueeze(0))
            .squeeze(0)
            .numpy()
            .astype(np.float32)
        )

    def _majority_class_from_indices(self, indices: np.ndarray) -> str | None:
        state = self._state
        labels = [
            state.image_classes[int(idx)]
            for idx in indices
            if 0 <= int(idx) < len(state.image_classes)
        ]
        if not labels:
            return None

        counts = Counter(labels)
        return counts.most_common(1)[0][0]

    def _filter_to_class(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
        target_class: str | None,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not target_class:
            return distances[:k], indices[:k]

        state = self._state
        pairs = [
            (dist, idx)
            for dist, idx in zip(distances, indices)
            if 0 <= int(idx) < len(state.image_classes)
            and state.image_classes[int(idx)] == target_class
        ]
        if not pairs:
            return distances[:k], indices[:k]

        kept_dists = np.asarray([dist for dist, _ in pairs], dtype=distances.dtype)
        kept_idxs = np.asarray([idx for _, idx in pairs], dtype=indices.dtype)
        return kept_dists[:k], kept_idxs[:k]

    def classify_majority_class(
        self,
        query_emb: np.ndarray,
        k: int = 60,
    ) -> str | None:
        state = self._state
        if not state.image_classes:
            return None

        _, indices = search(state.index, query_emb, k=min(k, len(state.image_paths)))
        return self._majority_class_from_indices(indices)

    def retrieve(
        self,
        query_emb: np.ndarray,
        slider_values: list[float],
        mmr_lambda: float = 0.7,
        k: int = 20,
    ) -> list[SearchResult]:
        """Retrieve images by steering query_emb along sliders and running FAISS search.

        Args:
            query_emb: Pre-encoded query embedding (unit norm).
            slider_values: One float per feature/direction slot.
            mmr_lambda: Diversity–relevance trade-off (1.0 = pure relevance).
            k: Number of results.

        Returns:
            List of SearchResult objects.
        """
        state = self._state

        majority_class = self.classify_majority_class(
            query_emb,
            k=max(k * 8, 120),
        )
        self.last_majority_class = majority_class

        search_k = min(max(k * 8, 120), len(state.image_paths))

        if state.class_directions is not None:
            distances, indices = search_with_direction_sliders(
                index=state.index,
                query_emb=query_emb,
                directions=state.class_directions,
                slider_values=[float(v) for v in slider_values],
                k=search_k,
                corpus_embeddings=state.embeddings if mmr_lambda < 1.0 else None,
                mmr_lambda=mmr_lambda,
            )
        else:
            slider_config = {
                fid: float(alpha)
                for fid, alpha in zip(state.feature_ids, slider_values)
                if abs(alpha) > 1e-6
            }
            distances, indices = search_with_sliders(
                index=state.index,
                query_emb=query_emb,
                sae_model=state.sae,
                slider_config=slider_config,
                k=search_k,
                corpus_activations=state.activations,
                corpus_embeddings=state.embeddings if mmr_lambda < 1.0 else None,
                mmr_lambda=mmr_lambda,
                sae_index=state.sae_index,
            )

        distances, indices = self._filter_to_class(distances, indices, majority_class, k)

        results: list[SearchResult] = []
        for idx in indices:
            if 0 <= idx < len(state.image_paths):
                try:
                    path = state.image_paths[idx]
                    image = Image.open(path).convert("RGB")
                    results.append(SearchResult(image=image, path=path))
                except Exception as e:
                    logger.warning(f"Could not load image {state.image_paths[idx]}: {e}")
        return results
