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


@dataclass
class RetrievalResult:
    items: list[SearchResult]
    majority_class: str | None
    was_filtered: bool


class RetrievalService:

    def __init__(self, state: AppState) -> None:
        self._state = state

    def encode_image_raw(self, query_image: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(query_image).convert("RGB")
        return (
            self._state.dino.encode(DINO_TRANSFORM(pil).unsqueeze(0))
            .squeeze(0)
            .numpy()
            .astype(np.float32)
        )

    def encode_image(self, query_image: np.ndarray) -> np.ndarray:
        raw = self.encode_image_raw(query_image)
        return normalize_embeddings(raw.reshape(1, -1)).squeeze(0)

    @staticmethod
    def normalize(raw_emb: np.ndarray) -> np.ndarray:
        return normalize_embeddings(raw_emb.reshape(1, -1)).squeeze(0)

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
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        if not target_class:
            return distances[:k], indices[:k], False

        state = self._state
        pairs = [
            (dist, idx)
            for dist, idx in zip(distances, indices)
            if 0 <= int(idx) < len(state.image_classes)
            and state.image_classes[int(idx)] == target_class
        ]
        if not pairs:
            return distances[:k], indices[:k], False

        kept_dists = np.asarray([dist for dist, _ in pairs], dtype=distances.dtype)
        kept_idxs = np.asarray([idx for _, idx in pairs], dtype=indices.dtype)
        return kept_dists[:k], kept_idxs[:k], True

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

    def _retrieve_patch(self, query_patches: np.ndarray, slider_values: list[float], k: int) -> RetrievalResult:
        from src.retrieval.patch_retrieval import maxsim_search, steer_patches

        state = self._state
        qp = np.asarray(query_patches, dtype=np.float32)
        slider_config = {
            fid: float(a) for fid, a in zip(state.feature_ids, slider_values) if abs(a) > 1e-6
        }
        if slider_config:
            fids = list(slider_config)
            dirs = state.sae.feature_directions(fids).cpu().numpy()
            qp = steer_patches(qp, dirs, [slider_config[f] for f in fids])

        _, indices = maxsim_search(state.patch_corpus, state.index, qp, k=k)
        majority_class = self._majority_class_from_indices(indices)
        dists = np.ones(len(indices), dtype=np.float32)
        _, indices, was_filtered = self._filter_to_class(dists, indices, majority_class, k)

        items: list[SearchResult] = []
        for idx in indices:
            if 0 <= int(idx) < len(state.image_paths):
                try:
                    path = state.image_paths[int(idx)]
                    items.append(SearchResult(image=Image.open(path).convert("RGB"), path=path))
                except Exception as e:
                    logger.warning(f"Could not load image {state.image_paths[int(idx)]}: {e}")
        return RetrievalResult(items=items, majority_class=majority_class, was_filtered=was_filtered)

    def retrieve(
        self,
        query_emb: np.ndarray,
        slider_values: list[float],
        mmr_lambda: float = 0.7,
        k: int = 20,
    ) -> RetrievalResult:
        state = self._state

        if state.patch_corpus is not None:
            return self._retrieve_patch(query_emb, slider_values, k)

        if state.class_directions is not None:
            distances, indices = search_with_direction_sliders(
                index=state.index,
                query_emb=query_emb,
                directions=state.class_directions,
                slider_values=[float(v) for v in slider_values],
                k=k,
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
                k=k,
                corpus_activations=state.activations,
                corpus_embeddings=state.embeddings if mmr_lambda < 1.0 else None,
                mmr_lambda=mmr_lambda,
                sae_index=state.sae_index,
                feature_scales=state.feature_scales,
            )

        majority_class = self._majority_class_from_indices(indices)
        distances, indices, was_filtered = self._filter_to_class(
            distances, indices, majority_class, k
        )

        items: list[SearchResult] = []
        for idx in indices:
            if 0 <= idx < len(state.image_paths):
                try:
                    path = state.image_paths[idx]
                    image = Image.open(path).convert("RGB")
                    items.append(SearchResult(image=image, path=path))
                except Exception as e:
                    logger.warning(f"Could not load image {state.image_paths[idx]}: {e}")
        return RetrievalResult(items=items, majority_class=majority_class, was_filtered=was_filtered)
