"""Direction-aware MMR drops near-duplicate sliders that activation-correlation keeps."""

import numpy as np

from src.naming.feature_ranking import rank_diverse_mmr


def _sparse_acts_and_dirs():
    # 6 features, ~5% active each (sparsity ~0.95, inside the [0.90, 0.995] filter).
    rng = np.random.default_rng(0)
    n, f, dim = 200, 6, 8
    acts = np.zeros((n, f), dtype=np.float32)
    mags = [5.0, 4.8, 1.0, 1.0, 1.0, 1.0]  # feat 0 highest variance, 1 next
    used: set[int] = set()
    for j, m in enumerate(mags):
        # disjoint active rows -> features 0 and 1 are activation-uncorrelated
        choices = [i for i in range(n) if i not in used]
        idx = rng.choice(choices, size=10, replace=False)
        used.update(int(i) for i in idx)
        acts[idx, j] = m

    dirs = np.zeros((f, dim), dtype=np.float32)
    dirs[0] = [1, 0, 0, 0, 0, 0, 0, 0]
    dirs[1] = [1, 0.01, 0, 0, 0, 0, 0, 0]  # near-identical direction to feature 0
    for j in range(2, f):
        dirs[j, j] = 1.0
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return acts, dirs


def test_direction_dedup_drops_duplicate():
    acts, dirs = _sparse_acts_and_dirs()
    with_dirs = rank_diverse_mmr(acts, n_features=3, directions=dirs)
    assert with_dirs[0] == 0
    assert 1 not in with_dirs  # duplicate direction excluded

    # without directions the activation-correlation MMR keeps both (they're uncorrelated)
    without = rank_diverse_mmr(acts, n_features=3)
    assert 0 in without and 1 in without


def test_semantic_dedup_drops_same_concept_feature():
    # Features 0 and 1 fire on DIFFERENT images that are semantically similar (cluster A),
    # so they're activation-uncorrelated but a VLM would name them the same. Feature 2
    # fires on cluster B. Semantic-fingerprint MMR should keep 0 and 2, drop 1.
    rng = np.random.default_rng(1)
    n, f, dim = 200, 6, 8
    a, b = np.zeros(dim, np.float32), np.zeros(dim, np.float32)
    a[0], b[1] = 1.0, 1.0
    embs = np.zeros((n, dim), np.float32)
    embs[:100] = a + 0.05 * rng.standard_normal((100, dim)).astype(np.float32)   # cluster A
    embs[100:] = b + 0.05 * rng.standard_normal((100, dim)).astype(np.float32)   # cluster B

    acts = np.zeros((n, f), np.float32)
    acts[0:20, 0] = 5.0      # cluster A images
    acts[20:40, 1] = 4.8     # different cluster A images (uncorrelated with feature 0)
    acts[100:120, 2] = 4.6   # cluster B images
    for j in range(3, f):    # low-variance filler on cluster B
        acts[120 + (j - 3) * 10: 130 + (j - 3) * 10, j] = 1.0

    sel = rank_diverse_mmr(acts, n_features=2, embeddings=embs)
    assert sel[0] == 0
    assert 1 not in sel and 2 in sel
