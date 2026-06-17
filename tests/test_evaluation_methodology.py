"""Regression tests for the evaluation methodology fixes."""

import importlib.util

import numpy as np
import torch

from src.evaluation.ablation import evaluate_sae_vs_pca
from src.evaluation.monosemanticity import (
    monosemanticity_score,
    n_distinct_classes,
    shuffled_label_purity_baseline,
)
from src.evaluation.stats import bootstrap_ci, summarize
from src.evaluation.steering_faithfulness import steering_faithfulness
from src.evaluation.steering_selectivity import steering_selectivity
from src.models.sae import SparseAutoencoder
from src.retrieval.index import build_index

_EV = importlib.util.spec_from_file_location("_ev", "scripts/evaluate.py")
ev = importlib.util.module_from_spec(_EV)
_EV.loader.exec_module(ev)


def _clustered_corpus(n_per=40, dim=32, n_clusters=3, seed=0):
    rng = np.random.default_rng(seed)
    embs, paths = [], []
    for ci in range(n_clusters):
        anchor = np.zeros(dim, dtype=np.float32)
        anchor[ci * 10:(ci + 1) * 10] = 1.0
        e = anchor + rng.standard_normal((n_per, dim)).astype(np.float32) * 0.1
        embs.append(e / np.linalg.norm(e, axis=1, keepdims=True))
        paths += [f"data/c{ci}/img{i}.jpg" for i in range(n_per)]
    return np.concatenate(embs).astype(np.float32), paths


class TestStats:
    def test_summarize_reports_distribution_and_frac(self):
        s = summarize([0.0, 1.0, 2.0, 3.0, 4.0], gt_threshold=1.5)
        assert s["mean"] == 2.0
        assert s["median"] == 2.0
        assert s["n"] == 5
        assert s["frac_gt"] == 0.6  # 2,3,4 are > 1.5
        assert s["ci_low"] <= s["mean"] <= s["ci_high"]

    def test_bootstrap_ci_singleton_and_empty(self):
        assert bootstrap_ci([5.0]) == (5.0, 5.0)
        lo, hi = bootstrap_ci([])
        assert np.isnan(lo) and np.isnan(hi)


class TestMonosemanticityNormalisation:
    def test_purity_normalised_by_total_classes(self):
        # top-k spans 2 classes evenly: 0 against the 2 observed, ~0.7 against 10 total
        acts = np.zeros((100, 1), dtype=np.float32)
        acts[:10, 0] = 1.0
        paths = [f"d/clsA/x{i}.jpg" if i < 5 else f"d/clsB/x{i}.jpg" for i in range(10)]
        paths += [f"d/cls{2 + (i % 8)}/x{i}.jpg" for i in range(10, 100)]
        s_total = monosemanticity_score(acts, paths, 0, k=10, n_total_classes=10)
        s_observed = monosemanticity_score(acts, paths, 0, k=10)
        assert s_observed["purity_score"] < 1e-6
        assert 0.6 < s_total["purity_score"] < 0.8
        assert s_total["purity_score"] > s_observed["purity_score"]

    def test_perfectly_pure_feature_scores_one(self):
        acts = np.zeros((50, 1), dtype=np.float32)
        acts[:10, 0] = np.arange(10, 0, -1)
        paths = [f"d/{'same' if i < 10 else 'other'}/x{i}.jpg" for i in range(50)]
        s = monosemanticity_score(acts, paths, 0, k=10, n_total_classes=2)
        assert s["purity_score"] == 1.0
        assert s["dominant_class_fraction"] == 1.0

    def test_null_baseline_collapses_real_minus_null_when_no_structure(self):
        # random activations -> real purity should sit near the shuffled-label null
        rng = np.random.default_rng(1)
        acts = rng.standard_normal((120, 8)).astype(np.float32)
        _, paths = _clustered_corpus()
        nc = n_distinct_classes(paths)
        feats = [0, 1, 2, 3]
        real = np.mean([monosemanticity_score(acts, paths, f, k=10, n_total_classes=nc)["purity_score"]
                        for f in feats])
        null = np.mean(shuffled_label_purity_baseline(acts, paths, feats, k=10, n_total_classes=nc, n_shuffles=4))
        assert abs(real - null) < 0.25


class TestFeatureSelection:
    def test_random_selection_excludes_dead_features(self):
        rng = np.random.default_rng(0)
        acts = rng.standard_normal((50, 20)).astype(np.float32)
        acts[:, 7] = 0.0  # dead
        sel = ev.select_eval_features(acts, 10, "random", seed=3)
        assert 7 not in sel
        assert len(sel) == 10
        assert len(set(sel)) == len(sel)

    def test_variance_selection_is_top_variance(self):
        acts = np.zeros((50, 5), dtype=np.float32)
        acts[:, 2] = np.linspace(0, 10, 50)   # highest variance
        acts[:, 4] = np.linspace(0, 1, 50)
        sel = ev.select_eval_features(acts, 2, "variance")
        assert sel[0] == 2


class TestGroundTruth:
    def test_in_sample_excludes_self(self):
        paths = ["d/a/1.jpg", "d/a/2.jpg", "d/b/3.jpg"]
        gt = ev.build_same_class_ground_truth(paths)
        assert gt == [[1], [0], []]

    def test_held_out_does_not_exclude_self(self):
        qp = ["d/a/q1.jpg", "d/b/q2.jpg"]
        cp = ["d/a/c1.jpg", "d/b/c2.jpg", "d/a/c3.jpg"]
        gt = ev.build_same_class_ground_truth(qp, cp)
        assert gt == [[0, 2], [1]]


class TestSteeringFaithfulnessBaseline:
    def test_zero_alpha_gives_no_effect(self):
        # alpha=0: steered == unsteered, so the ratio is ~1.0 (baseline-normalised)
        embs, _ = _clustered_corpus()
        index = build_index(embs)
        sae = SparseAutoencoder(input_dim=32, hidden_dim=16)
        sae.eval()
        with torch.no_grad():
            acts = sae.encode(torch.from_numpy(embs)).numpy()
        live = int(np.argmax(acts.var(0)))
        f = steering_faithfulness(sae, index, acts, embs[:10], live, alpha=0.0, k=5, n_queries=10)
        assert abs(f - 1.0) < 1e-3


class TestSteeringSelectivity:
    def test_on_target_fraction_in_unit_range(self):
        embs, _ = _clustered_corpus()
        index = build_index(embs)
        sae = SparseAutoencoder(input_dim=32, hidden_dim=16)
        sae.eval()
        with torch.no_grad():
            acts = sae.encode(torch.from_numpy(embs)).numpy()
        live = int(np.argmax(acts.var(0)))
        s = steering_selectivity(sae, index, acts, embs[:10], live, alpha=2.0, k=5, n_queries=10)
        assert np.isnan(s) or 0.0 <= s <= 1.0


class TestAblationLiftIsBounded:
    def test_advantage_is_finite_and_metric_is_shared(self):
        embs, _ = _clustered_corpus()
        index = build_index(embs)
        sae = SparseAutoencoder(input_dim=32, hidden_dim=16)
        sae.eval()
        with torch.no_grad():
            acts = sae.encode(torch.from_numpy(embs)).numpy()
        feats = [int(i) for i in np.argsort(acts.var(0))[::-1][:4]]
        q = embs[:8]
        ab = evaluate_sae_vs_pca(sae, index, embs, acts, q, feats,
                                 n_pca_components=4, alpha=2.0, k=5, n_queries=8)
        # lifts are cosine-scale, bounded in [-2, 2] (no ratio blow-up)
        for key in ("sae_mean_faithfulness", "pca_mean_faithfulness",
                    "steering_advantage", "steering_advantage_mean"):
            assert np.isfinite(ab[key])
            assert -2.0 <= ab[key] <= 2.0
        assert ab["steering_advantage"] == ab["sae_median_faithfulness"] - ab["pca_median_faithfulness"]
