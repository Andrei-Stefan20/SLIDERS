from pathlib import Path
from typing import cast

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from src.encoders.dino_encoder import DINO_TRANSFORM as IMAGENET_TRANSFORM
from src.utils.io import normalize_embeddings


class ImageFolderFlat(Dataset):

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

    def __init__(
        self, npy_path: Path | str, mmap: bool = False, normalize: bool = True
    ) -> None:
        path = Path(npy_path)
        if mmap:
            # Stay memory-mapped; normalize per-row in __getitem__ instead.
            self._data: np.ndarray = np.load(path, mmap_mode="r")
            self._normalize_rows = normalize
        else:
            data = np.load(path).astype(np.float32)
            self._data = normalize_embeddings(data) if normalize else data
            self._normalize_rows = False
        if self._data.ndim != 2:
            raise ValueError(
                f"EmbeddingDataset expects shape (N, D), got {self._data.shape}"
            )

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self._data[idx].astype(np.float32)
        if self._normalize_rows:
            norm = float(np.linalg.norm(row))
            if norm > 0:
                row = row / norm
        return torch.from_numpy(row)
