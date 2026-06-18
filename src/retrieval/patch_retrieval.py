"""Patch-level retrieval via late interaction (ColBERT-style MaxSim).

A query image is a set of patch vectors; corpus images are scored by
MaxSim = sum over query patches of the best matching corpus patch. A FAISS index
over all patches generates candidate images, then MaxSim is computed exactly from the
patch memmap for those candidates. Steering adds an SAE feature direction to the query
patches before search (the patch-space analog of CLS slider steering)."""

from pathlib import Path

import faiss
import numpy as np

from src.retrieval.patch_store import PatchReader


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n > 1e-8, n, 1.0)


class PatchCorpus:
    """Memmapped patch embeddings with O(1) access to an image's patches.

    Patches of one image are contiguous (see scripts/extract_embeddings.py), so image
    ``i`` owns rows ``[i*P : (i+1)*P]``. Reads go through PatchReader, so int8-quantized
    storage is dequantized transparently."""

    def __init__(self, patch_emb_path: Path | str) -> None:
        self.reader = PatchReader(patch_emb_path)
        self.data = self.reader.data
        self.image_ids = self.reader.image_ids
        self.meta = self.reader.meta
        self.patches_per_image = self.reader.patches_per_image
        self.n_images = self.reader.n_images

    def image_patches(self, image_id: int, normalize: bool = True) -> np.ndarray:
        s = image_id * self.patches_per_image
        rows = self.reader.rows(slice(s, s + self.patches_per_image))
        return l2_normalize(rows) if normalize else rows


def maxsim_score(query_patches: np.ndarray, doc_patches: np.ndarray) -> float:
    """Late-interaction score: sum over query patches of the best doc-patch similarity.
    Both inputs must be unit-normalized."""
    sim = query_patches @ doc_patches.T  # (P_q, P_d)
    return float(sim.max(axis=1).sum())


def steer_patches(
    query_patches: np.ndarray, directions: np.ndarray, alphas: list[float]
) -> np.ndarray:
    """Add sum(alpha_i * direction_i) to every query patch, then renormalize each."""
    delta = (np.asarray(alphas, dtype=np.float32)[:, None] * directions).sum(axis=0)
    return l2_normalize(np.asarray(query_patches, dtype=np.float32) + delta)


def maxsim_search(
    corpus: PatchCorpus,
    index: faiss.Index,
    query_patches: np.ndarray,
    k: int = 10,
    probe: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, image_ids) of the top-k corpus images for a query image.

    ``probe`` candidate patches are fetched per query patch; their images form the
    candidate set, then each is scored exactly by MaxSim."""
    qn = np.ascontiguousarray(l2_normalize(query_patches))
    _, patch_idxs = index.search(qn, probe)
    flat = patch_idxs.reshape(-1)
    cand = np.unique(corpus.image_ids[flat[flat >= 0]])
    if len(cand) == 0:
        return np.empty(0, np.float32), np.empty(0, np.int64)
    scored = [(maxsim_score(qn, corpus.image_patches(int(im))), int(im)) for im in cand]
    scored.sort(key=lambda x: -x[0])
    top = scored[:k]
    return (
        np.array([s for s, _ in top], dtype=np.float32),
        np.array([i for _, i in top], dtype=np.int64),
    )
