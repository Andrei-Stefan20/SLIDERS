"""Localize SAE features spatially using DINOv2 patch tokens."""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from src.encoders.dino_encoder import DINO_TRANSFORM, DINOEncoder
from src.models.sae import SparseAutoencoder

_GEOM_TRANSFORM = T.Compose([T.Resize(256), T.CenterCrop(224)])


def _aligned_224(img: Image.Image) -> Image.Image:
    return cast(Image.Image, _GEOM_TRANSFORM(img))


def localize_feature(
    image_path: Path | str,
    dino: DINOEncoder,
    sae: SparseAutoencoder,
    feature_id: int,
    crop_size: int = 96,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    if not dino.use_patches:
        raise ValueError("localize_feature requires DINOEncoder(use_patches=True)")

    img = Image.open(image_path).convert("RGB")
    img_tensor = cast(torch.Tensor, DINO_TRANSFORM(img)).unsqueeze(0)

    with torch.no_grad():
        patch_tokens = dino.encode(img_tensor)
        patch_acts = sae.encode(patch_tokens.squeeze(0))

    feat_acts = patch_acts[:, feature_id].numpy()
    best_patch_idx = int(np.argmax(feat_acts))

    grid_size = int(np.sqrt(len(feat_acts)))
    row = best_patch_idx // grid_size
    col = best_patch_idx % grid_size
    patch_px = 224 // grid_size

    center_x = col * patch_px + patch_px // 2
    center_y = row * patch_px + patch_px // 2

    half = crop_size // 2
    x1 = max(0, center_x - half)
    y1 = max(0, center_y - half)
    x2 = min(224, center_x + half)
    y2 = min(224, center_y + half)

    img_aligned = _aligned_224(img)
    crop = img_aligned.crop((x1, y1, x2, y2))
    return crop, (x1, y1, x2, y2)


def localize_feature_batch(
    image_paths: Sequence[Path | str],
    dino: DINOEncoder,
    sae: SparseAutoencoder,
    feature_id: int,
    crop_size: int = 96,
) -> list[Image.Image]:
    return [
        localize_feature(p, dino, sae, feature_id, crop_size)[0]
        for p in image_paths
    ]
