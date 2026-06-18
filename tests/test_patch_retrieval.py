"""Patch MaxSim retrieval + patch evaluation on a toy two-class corpus."""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.patch_eval import patch_retrieval_results  # noqa: E402
from src.evaluation.recall_at_k import mean_precision_at_k  # noqa: E402
from src.retrieval.index import build_patch_index  # noqa: E402
from src.retrieval.patch_retrieval import (  # noqa: E402
    PatchCorpus,
    maxsim_score,
    steer_patches,
)
from src.utils.io import save_patch_sidecars  # noqa: E402

P, D = 4, 8


@pytest.fixture
def tmp_path():
    d = Path(tempfile.mkdtemp(dir=str(ROOT)))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _two_class_corpus(tmp_path, name, n_per_class=3):
    """Class A patches cluster near axis 0, class B near axis 1."""
    rng = np.random.default_rng(abs(hash(name)) % 1000)
    rows, ids, labels = [], [], []
    img = 0
    for cls, axis in (("A", 0), ("B", 1)):
        for _ in range(n_per_class):
            base = np.zeros((P, D), np.float32)
            base[:, axis] = 1.0
            rows.append(base + 0.05 * rng.standard_normal((P, D)).astype(np.float32))
            ids.extend([img] * P)
            labels.append(cls)
            img += 1
    patches = np.concatenate(rows, axis=0)
    path = tmp_path / f"{name}_patch_embeddings.npy"
    np.save(path, patches)
    save_patch_sidecars(path, np.array(ids, np.int32),
                        {"grid_size": 2, "patches_per_image": P, "n_images": img})
    return PatchCorpus(path), labels


def test_maxsim_score_prefers_aligned_patches():
    q = np.eye(D, dtype=np.float32)[:P]          # unit patches on first P axes
    same = q.copy()
    diff = np.eye(D, dtype=np.float32)[P:2 * P]   # disjoint axes
    assert maxsim_score(q, same) > maxsim_score(q, diff)


def test_steer_patches_moves_toward_direction():
    q = np.zeros((P, D), np.float32)
    q[:, 0] = 1.0
    direction = np.zeros((1, D), np.float32)
    direction[0, 1] = 1.0
    steered = steer_patches(q, direction, [5.0])
    assert (steered[:, 1] > steered[:, 0]).all()  # pushed toward axis 1


def test_maxsim_retrieval_recovers_class(tmp_path):
    corpus, corpus_labels = _two_class_corpus(tmp_path, "corpus")
    query, query_labels = _two_class_corpus(tmp_path, "query", n_per_class=1)
    index = build_patch_index(np.asarray(corpus.data), use_pq=False)

    results = patch_retrieval_results(corpus, index, query, query_labels, corpus_labels, k=3)
    # the top hit for each query is a same-class image
    prec = mean_precision_at_k(results, k_values=[1])
    assert prec["precision@1"] == 1.0


def test_purity_from_labels_extremes():
    from src.evaluation.patch_eval import _purity_from_labels

    pure = _purity_from_labels(["A"] * 10, n_total_classes=4)
    assert pure["purity_score"] == 1.0 and pure["dominant_class_fraction"] == 1.0
    mixed = _purity_from_labels(["A", "B", "C", "D"], n_total_classes=4)
    assert mixed["purity_score"] < 0.05  # uniform over all classes -> ~0


def test_spearman_monotonic():
    from src.evaluation.patch_eval import _spearman

    assert _spearman(np.array([1, 2, 3, 4.0]), np.array([2, 4, 6, 8.0])) == pytest.approx(1.0)
    assert _spearman(np.array([1, 2, 3, 4.0]), np.array([8, 6, 4, 2.0])) == pytest.approx(-1.0)


def test_patch_monosemanticity_and_isotonicity_run(tmp_path):
    from src.evaluation.patch_eval import (
        live_features_from_sample,
        patch_isotonicity,
        patch_monosemanticity,
    )
    from src.models.sae import SparseAutoencoder

    corpus, corpus_labels = _two_class_corpus(tmp_path, "corpus")
    query, _ = _two_class_corpus(tmp_path, "query", n_per_class=1)
    index = build_patch_index(np.asarray(corpus.data), use_pq=False)
    sae = SparseAutoencoder(input_dim=D, hidden_dim=16)
    sae.eval()
    feats = live_features_from_sample(corpus, sae, n_features=2, sample_patches=24)
    if not feats:
        pytest.skip("random SAE has no live feature on this toy corpus")

    mono = patch_monosemanticity(corpus, sae, feats, corpus_labels, k=3, n_total_classes=2)
    assert all(0.0 <= m["purity_score"] <= 1.0 for m in mono)

    iso = patch_isotonicity(corpus, index, query, sae, feats[0], k=3, n_queries=2)
    assert -1.0 <= iso["spearman_rho"] <= 1.0


def test_ui_patch_retrieve_majority_class(tmp_path):
    from src.models.sae import SparseAutoencoder
    from src.ui.retrieval_service import RetrievalService
    from src.ui.state import AppState

    corpus, labels = _two_class_corpus(tmp_path, "corpus")
    index = build_patch_index(np.asarray(corpus.data), use_pq=False)
    sae = SparseAutoencoder(input_dim=D, hidden_dim=16)
    sae.eval()
    image_paths = [f"/data/{labels[i]}/img{i}.jpg" for i in range(corpus.n_images)]
    state = AppState(
        dino=None, sae=sae, index=index, sae_index=None, embeddings=None, activations=None,
        image_paths=image_paths, feature_ids=[0, 1], feature_names=["f0", "f1"],
        feature_descriptions=["", ""], image_classes=labels, patch_corpus=corpus,
        path_to_idx={p: i for i, p in enumerate(image_paths)},
    )
    svc = RetrievalService(state)
    # query = a class-A image's patches; MaxSim should make class A the majority
    res = svc.retrieve(corpus.image_patches(0), [0.0, 0.0], k=3)
    assert res.majority_class == labels[0]


def test_patch_steering_faithfulness_runs(tmp_path):
    from src.evaluation.patch_eval import (
        live_features_from_sample,
        patch_steering_faithfulness,
    )
    from src.models.sae import SparseAutoencoder

    corpus, _ = _two_class_corpus(tmp_path, "corpus")
    query, _ = _two_class_corpus(tmp_path, "query", n_per_class=1)
    index = build_patch_index(np.asarray(corpus.data), use_pq=False)
    sae = SparseAutoencoder(input_dim=D, hidden_dim=16)
    sae.eval()
    feats = live_features_from_sample(corpus, sae, n_features=1, sample_patches=24)
    if not feats:
        pytest.skip("random SAE has no live feature on this toy corpus")
    r = patch_steering_faithfulness(corpus, index, query, sae, feats[0], alpha=2.0, k=3, n_queries=2)
    assert isinstance(r, float)  # runs end-to-end (value may be nan on a toy corpus)
