"""FAISS index construction, persistence, and loading."""

from pathlib import Path

import faiss
import numpy as np


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """IndexFlatIP over L2-normalised embeddings (cosine similarity)."""
    embs = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = np.ascontiguousarray(embs / np.where(norms > 1e-8, norms, 1.0))
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)  # type: ignore[call-arg]
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


def build_patch_index(
    patches: np.ndarray, use_pq: bool = False, nlist: int = 4096, m: int = 32
) -> faiss.Index:
    """IndexFlatIP over L2-normalised patch vectors for late-interaction candidate
    generation. With ``use_pq`` build an IVF-PQ instead (≈1-2 GB for ~10M patches vs
    ~45 GB flat); the exact MaxSim rescoring then runs from the patch memmap, so PQ's
    lossiness only affects which candidates are fetched, not the final ranking."""
    embs = np.ascontiguousarray(_l2(patches))
    d = embs.shape[1]
    if not use_pq:
        index = faiss.IndexFlatIP(d)
        index.add(embs)
        return index
    nlist = min(nlist, max(1, embs.shape[0] // 39))
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
    index.train(embs)
    index.add(embs)
    index.nprobe = 16
    return index


def _l2(embs: np.ndarray) -> np.ndarray:
    embs = np.asarray(embs, dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.where(norms > 1e-8, norms, 1.0)


def save_index(index: faiss.Index, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path | str) -> faiss.Index:
    return faiss.read_index(str(path))
