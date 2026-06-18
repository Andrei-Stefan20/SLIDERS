"""Patch-token memmap extraction: shapes, patch->image mapping, split routing."""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_embeddings import write_patch_memmaps  # noqa: E402
from src.utils.io import load_patch_sidecars  # noqa: E402


@pytest.fixture
def tmp_path():
    """Local temp dir under the repo; the system temp dir is locked on this Windows box."""
    d = Path(tempfile.mkdtemp(dir=str(ROOT)))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)

P, DIM = 4, 3  # 4 patches/image (2x2 grid), 3-dim


def _batches(n_images, batch=2):
    """Each image's patches carry its global index as their value, so routing is checkable."""
    paths = [f"img_{i}.jpg" for i in range(n_images)]
    for start in range(0, n_images, batch):
        idxs = range(start, min(start + batch, n_images))
        emb = np.stack([np.full((P, DIM), float(i), dtype=np.float32) for i in idxs])
        yield emb, [paths[i] for i in idxs]


def test_single_file_no_split(tmp_path):
    n = 5
    written = write_patch_memmaps(_batches(n), n, tmp_path, "ds", val_split=0.0)
    assert len(written) == 1
    data = np.load(written[0], mmap_mode="r")
    assert data.shape == (n * P, DIM)

    ids, meta = load_patch_sidecars(written[0])
    assert (meta["grid_size"], meta["patches_per_image"], meta["n_images"]) == (2, P, n)
    assert data.dtype == np.float16  # default storage dtype halves the footprint
    # Each image's P contiguous rows hold its index; ids map back to that image.
    for img in range(n):
        rows = data[img * P : (img + 1) * P]
        assert np.all(rows == float(img))
        assert np.all(ids[img * P : (img + 1) * P] == img)


def test_salient_subsampling_keeps_top_n_by_norm(tmp_path):
    # one image, P patches with increasing norm; keep the 2 highest-norm patches.
    paths = ["img_0.jpg"]
    block = np.zeros((1, P, DIM), np.float32)
    for p in range(P):
        block[0, p, 0] = float(p + 1)  # norms 1,2,3,4
    written = write_patch_memmaps(
        iter([(block, paths)]), 1, tmp_path, "ds", val_split=0.0, max_patches_per_image=2
    )
    data = np.load(written[0])
    _, meta = load_patch_sidecars(written[0])
    assert meta["patches_per_image"] == 2
    assert data.shape == (2, DIM)
    # the two kept patches are the highest-norm ones (values 3 and 4)
    assert sorted(data[:, 0].tolist()) == [3.0, 4.0]


def test_int8_storage_roundtrips(tmp_path):
    from src.retrieval.patch_store import PatchReader

    block = np.random.default_rng(0).standard_normal((1, P, DIM)).astype(np.float32)
    written = write_patch_memmaps(
        iter([(block, ["img.jpg"])]), 1, tmp_path, "ds", val_split=0.0, dtype=np.int8
    )
    data = np.load(written[0])
    assert data.dtype == np.int8  # ~1 byte/value
    recon = PatchReader(written[0]).rows(slice(0, P))
    assert np.allclose(recon, block[0], atol=0.05)  # per-row int8 quant error is tiny


def test_embedding_dataset_dequantizes_int8(tmp_path):
    from src.data.loader import EmbeddingDataset

    block = np.random.default_rng(1).standard_normal((1, P, DIM)).astype(np.float32)
    written = write_patch_memmaps(
        iter([(block, ["img.jpg"])]), 1, tmp_path, "ds", val_split=0.0, dtype=np.int8
    )
    ds = EmbeddingDataset(written[0], mmap=True, normalize=False)
    assert np.allclose(ds[0].numpy(), block[0, 0], atol=0.05)


def test_image_level_split_partitions_all_images(tmp_path):
    n = 10
    write_patch_memmaps(_batches(n), n, tmp_path, "ds", val_split=0.3)
    train = np.load(tmp_path / "ds_train_patch_embeddings.npy", mmap_mode="r")
    val = np.load(tmp_path / "ds_val_patch_embeddings.npy", mmap_mode="r")

    # 3 images held out, 7 train; every patch row is intact (P identical rows per image).
    assert val.shape == (3 * P, DIM)
    assert train.shape == (7 * P, DIM)

    seen = {float(train[i * P, 0]) for i in range(7)} | {float(val[i * P, 0]) for i in range(3)}
    assert seen == {float(i) for i in range(n)}  # train ∪ val == all images, no overlap
