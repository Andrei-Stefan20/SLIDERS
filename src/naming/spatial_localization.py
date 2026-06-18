"""Localize SAE features spatially using DINOv2 patch tokens."""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

from src.encoders.dino_encoder import DINO_TRANSFORM, DINOEncoder
from src.models.sae import SparseAutoencoder

_GEOM_TRANSFORM = T.Compose([T.Resize(256), T.CenterCrop(224)])


def _aligned_224(img: Image.Image) -> Image.Image:
    return cast(Image.Image, _GEOM_TRANSFORM(img))


def _patch_box(patch_idx: int, grid_size: int, crop_size: int) -> tuple[int, int, int, int]:
    patch_px = 224 // grid_size
    center_x = (patch_idx % grid_size) * patch_px + patch_px // 2
    center_y = (patch_idx // grid_size) * patch_px + patch_px // 2
    half = crop_size // 2
    return (
        max(0, center_x - half), max(0, center_y - half),
        min(224, center_x + half), min(224, center_y + half),
    )


def _highlight_crop(
    img_aligned: Image.Image, patch_idx: int, grid_size: int, crop_size: int
) -> Image.Image:
    """Context crop with the active patch cell outlined.

    The SAE activation correlates with a patch but attention mixes in context, so the
    true cause may be nearby rather than the patch itself (arXiv:2509.00749). Keeping
    context and just marking the patch lets the VLM use both."""
    cx1, cy1, cx2, cy2 = _patch_box(patch_idx, grid_size, crop_size)
    px1, py1, px2, py2 = _patch_box(patch_idx, grid_size, 224 // grid_size)
    crop = img_aligned.crop((cx1, cy1, cx2, cy2)).convert("RGB")
    draw = ImageDraw.Draw(crop)
    box = (px1 - cx1, py1 - cy1, px2 - cx1 - 1, py2 - cy1 - 1)
    draw.rectangle(box, outline=(255, 0, 0), width=2)
    return crop


def _montage(crops: list[Image.Image]) -> Image.Image:
    """Arrange crops into a near-square grid."""
    if not crops:
        return Image.new("RGB", (1, 1))
    cols = int(np.ceil(np.sqrt(len(crops))))
    rows = int(np.ceil(len(crops) / cols))
    cw = max(c.width for c in crops)
    ch = max(c.height for c in crops)
    out = Image.new("RGB", (cols * cw, rows * ch), (255, 255, 255))
    for i, c in enumerate(crops):
        out.paste(c, ((i % cols) * cw, (i // cols) * ch))
    return out


def _patch_activations(
    image_path: Path | str, dino: DINOEncoder, sae: SparseAutoencoder, feature_id: int
) -> tuple[Image.Image, np.ndarray, int]:
    if not dino.use_patches:
        raise ValueError("localization requires DINOEncoder(use_patches=True)")
    img = Image.open(image_path).convert("RGB")
    img_tensor = cast(torch.Tensor, DINO_TRANSFORM(img)).unsqueeze(0)
    with torch.no_grad():
        patch_tokens = dino.encode(img_tensor)
        patch_acts = sae.encode(patch_tokens.squeeze(0))
    feat_acts = patch_acts[:, feature_id].numpy()
    return _aligned_224(img), feat_acts, int(np.sqrt(len(feat_acts)))


def localize_feature(
    image_path: Path | str,
    dino: DINOEncoder,
    sae: SparseAutoencoder,
    feature_id: int,
    crop_size: int = 96,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    img_aligned, feat_acts, grid_size = _patch_activations(image_path, dino, sae, feature_id)
    box = _patch_box(int(np.argmax(feat_acts)), grid_size, crop_size)
    return img_aligned.crop(box), box


def feature_activation_grid(
    image_path: Path | str,
    dino: DINOEncoder,
    sae: SparseAutoencoder,
    feature_id: int,
) -> tuple[Image.Image, np.ndarray]:
    """Aligned 224x224 RGB image plus its (G, G) per-patch activation map for a feature.

    Row-major grid (patch ``i`` -> row ``i//G``, col ``i%G``), so it overlays directly on
    the aligned image as a heatmap (used by the report's activation maps)."""
    img_aligned, feat_acts, grid_size = _patch_activations(image_path, dino, sae, feature_id)
    return img_aligned, feat_acts.reshape(grid_size, grid_size)


def localize_feature_topk(
    image_path: Path | str,
    dino: DINOEncoder,
    sae: SparseAutoencoder,
    feature_id: int,
    n_patches: int = 4,
    crop_size: int = 96,
) -> Image.Image:
    """Montage of context crops around an image's top-N activating patches, each with
    the active patch outlined.

    A patch-trained feature is a region, so the VLM sees several of the image's strongest
    patches (in context, marked) rather than the single best one (that's localize_feature).
    """
    img_aligned, feat_acts, grid_size = _patch_activations(image_path, dino, sae, feature_id)
    n = min(n_patches, len(feat_acts))
    top = np.argsort(feat_acts)[::-1][:n]
    crops = [_highlight_crop(img_aligned, int(i), grid_size, crop_size) for i in top]
    return _montage(crops)


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


def localize_feature_topk_batch(
    image_paths: Sequence[Path | str],
    dino: DINOEncoder,
    sae: SparseAutoencoder,
    feature_id: int,
    n_patches: int = 4,
    crop_size: int = 96,
) -> list[Image.Image]:
    """One multi-patch montage per image (see ``localize_feature_topk``)."""
    return [
        localize_feature_topk(p, dino, sae, feature_id, n_patches, crop_size)
        for p in image_paths
    ]
