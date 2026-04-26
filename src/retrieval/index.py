"""FAISS index construction, persistence, and loading."""

from pathlib import Path

import faiss
import numpy as np


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """IndexFlatIP over L2-normalised embeddings (cosine similarity)."""
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)  # type: ignore[call-arg]
    return index


def build_sae_index(activations: np.ndarray) -> faiss.Index:
    """IndexFlatIP over L2-normalised SAE activation vectors.

    Searching in this space is geometrically consistent with slider steering,
    which operates along SAE decoder directions.
    """
    acts = activations.astype(np.float32)
    norms = np.linalg.norm(acts, axis=1, keepdims=True)
    acts_normed = np.ascontiguousarray(acts / np.where(norms > 1e-8, norms, 1.0))
    index = faiss.IndexFlatIP(acts_normed.shape[1])
    index.add(acts_normed)  # type: ignore[call-arg]
    return index


def save_index(index: faiss.Index, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path | str) -> faiss.Index:
    return faiss.read_index(str(path))
