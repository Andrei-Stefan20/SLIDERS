"""Patch-level naming: per-image aggregation (max) and multi-patch montage."""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.name_features import _aggregate_patch_activations  # noqa: E402
from src.models.sae import SparseAutoencoder  # noqa: E402
from src.naming.spatial_localization import (  # noqa: E402
    _highlight_crop,
    _montage,
    _patch_box,
)
from src.utils.io import save_patch_sidecars  # noqa: E402


@pytest.fixture
def tmp_path():
    d = Path(tempfile.mkdtemp(dir=str(ROOT)))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_aggregate_takes_per_image_max(tmp_path):
    # 2 images x 3 patches x 4 dim.
    rng = np.random.default_rng(0)
    patches = rng.standard_normal((6, 4)).astype(np.float32)
    emb_path = tmp_path / "ds_patch_embeddings.npy"
    np.save(emb_path, patches)
    ids = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    save_patch_sidecars(emb_path, ids, {"grid_size": 0, "patches_per_image": 3, "n_images": 2})

    sae = SparseAutoencoder(input_dim=4, hidden_dim=3)
    sae.eval()
    max_acts, mean_emb = _aggregate_patch_activations(emb_path, sae, batch_rows=4)

    assert max_acts.shape == (2, 3)
    assert mean_emb.shape == (2, 4)

    # Reference: encode each image's normalized patches, take the column-wise max.
    for img, rows in ((0, patches[:3]), (1, patches[3:])):
        n = rows / np.linalg.norm(rows, axis=1, keepdims=True)
        with torch.no_grad():
            ref = sae.encode(torch.from_numpy(n.astype(np.float32))).numpy().max(0)
        assert np.allclose(max_acts[img], ref, atol=1e-5)


def test_montage_packs_all_crops():
    crops = [Image.new("RGB", (10, 10), (i, 0, 0)) for i in range(4)]
    m = _montage(crops)
    assert m.size == (20, 20)  # 4 crops -> 2x2 grid of 10px tiles


def test_patch_box_within_bounds():
    x1, y1, x2, y2 = _patch_box(patch_idx=255, grid_size=16, crop_size=96)
    assert 0 <= x1 < x2 <= 224
    assert 0 <= y1 < y2 <= 224


def test_highlight_crop_keeps_context_and_marks_patch():
    img = Image.new("RGB", (224, 224), (120, 120, 120))
    crop = _highlight_crop(img, patch_idx=120, grid_size=16, crop_size=96)
    assert crop.size == (96, 96)  # full context kept, not shrunk to the patch
    arr = np.asarray(crop)
    # the red outline introduces pixels the flat gray image had none of
    assert ((arr[:, :, 0] > 200) & (arr[:, :, 1] < 60) & (arr[:, :, 2] < 60)).any()
