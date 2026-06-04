"""Dataset classes for loading raw images and pre-extracted embeddings."""

from pathlib import Path
from typing import cast

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from src.encoders.dino_encoder import DINO_TRANSFORM as IMAGENET_TRANSFORM


class ImageFolderFlat(Dataset):
    """Recursively loads all images from a directory tree as (tensor, path) pairs."""

    EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

    def __init__(
        self,
        root: Path | str,
        transform: T.Compose | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform or IMAGENET_TRANSFORM
        self.paths: list[Path] = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in self.EXTENSIONS
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        img = Image.open(self.paths[idx]).convert("RGB")
        return cast(torch.Tensor, self.transform(img)), str(self.paths[idx])


class EmbeddingDataset(Dataset):
    """Wraps a pre-extracted .npy embedding matrix as a PyTorch Dataset.

    Each __getitem__ returns a single float32 embedding vector.
    The full array is memory-mapped when mmap=True.
    """

    def __init__(self, npy_path: Path | str, mmap: bool = False) -> None:
        path = Path(npy_path)
        if mmap:
            self._data: np.ndarray = np.load(path, mmap_mode="r")
        else:
            self._data = np.load(path).astype(np.float32)
        if self._data.ndim != 2:
            raise ValueError(
                f"EmbeddingDataset expects shape (N, D), got {self._data.shape}"
            )

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self._data[idx].astype(np.float32))
