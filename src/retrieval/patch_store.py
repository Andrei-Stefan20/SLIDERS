"""int8-quantized patch storage with transparent on-read dequantization.

Patch memmaps dominate disk (≈45 GB float32 / 22 GB float16 for ~11M patches).
Per-row symmetric int8 quantization stores 1 byte/value + one float32 scale per patch
(≈11 GB), and `PatchReader` hands every consumer (training, indexing, naming, retrieval)
plain float32 rows so nothing else needs to know the storage is quantized."""

from pathlib import Path

import numpy as np

from src.utils.io import load_patch_sidecars, patch_scales_path


def quantize_int8(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric int8 quantization. Returns (codes int8, scales float32)."""
    v = np.asarray(vectors, dtype=np.float32)
    scales = np.abs(v).max(axis=1) / 127.0
    scales = np.where(scales > 1e-12, scales, 1.0).astype(np.float32)
    codes = np.round(v / scales[:, None]).clip(-127, 127).astype(np.int8)
    return codes, scales


def dequantize_int8(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return codes.astype(np.float32) * np.asarray(scales, dtype=np.float32)[:, None]


class PatchReader:
    """Reads a patch memmap as float32, dequantizing int8 storage on the fly."""

    def __init__(self, embeddings_path: Path | str) -> None:
        self.data = np.load(embeddings_path, mmap_mode="r")
        self.image_ids, self.meta = load_patch_sidecars(embeddings_path)
        self.patches_per_image = int(self.meta["patches_per_image"])
        self.n_images = int(self.meta["n_images"])
        self.is_int8 = self.data.dtype == np.int8
        self.scales = (
            np.load(patch_scales_path(embeddings_path), mmap_mode="r")
            if self.is_int8 else None
        )

    def __len__(self) -> int:
        return len(self.data)

    @property
    def dim(self) -> int:
        return self.data.shape[1]

    def rows(self, sl) -> np.ndarray:
        """Float32 rows for a slice or index array, dequantized if stored as int8."""
        block = np.asarray(self.data[sl], dtype=np.float32)
        if self.is_int8:
            block = block * np.asarray(self.scales[sl], dtype=np.float32)[:, None]
        return block
