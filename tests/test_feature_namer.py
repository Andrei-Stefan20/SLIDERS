"""Tests for feature naming utilities."""

import numpy as np

from src.naming.feature_namer import get_top_images, rank_features_by_variance


class TestGetTopImages:
    def test_returns_feature_images(self, fake_activations, fake_image_paths):
        fi = get_top_images(fake_activations, fake_image_paths, feature_id=0, k=5)
        assert fi.feature_id == 0
        assert len(fi.top_paths) == 5
        assert len(fi.bottom_paths) == 5

    def test_top_activations_descending(self, fake_activations, fake_image_paths):
        fi = get_top_images(fake_activations, fake_image_paths, feature_id=0, k=10)
        acts = fi.top_activations
        assert all(acts[i] >= acts[i + 1] for i in range(len(acts) - 1))

    def test_bottom_activations_ascending(self, fake_activations, fake_image_paths):
        fi = get_top_images(fake_activations, fake_image_paths, feature_id=0, k=5)
        acts = fi.bottom_activations
        assert all(acts[i] <= acts[i + 1] for i in range(len(acts) - 1))

    def test_k_clamps_to_dataset_size(self, fake_activations, fake_image_paths):
        fi = get_top_images(fake_activations, fake_image_paths, feature_id=0, k=1000)
        assert len(fi.top_paths) == len(fake_image_paths)

    def test_paths_are_path_objects(self, fake_activations, fake_image_paths):
        from pathlib import Path

        fi = get_top_images(fake_activations, fake_image_paths, feature_id=0, k=3)
        assert all(isinstance(p, Path) for p in fi.top_paths)


class TestRankFeaturesByVariance:
    def test_returns_list_of_ints(self, fake_activations):
        ranked = rank_features_by_variance(fake_activations)
        assert isinstance(ranked, list)
        assert all(isinstance(i, int) for i in ranked)

    def test_excludes_dead_features(self):
        acts = np.ones((50, 8), dtype=np.float32)
        acts[:, 0] = 0.0
        ranked = rank_features_by_variance(acts)
        assert 0 not in ranked

    def test_sorted_by_variance_descending(self):
        rng = np.random.default_rng(1)
        acts = rng.standard_normal((100, 10)).astype(np.float32)
        acts = np.abs(acts)
        for i in range(10):
            acts[:, i] *= i + 1
        ranked = rank_features_by_variance(acts)
        variances = acts.var(axis=0)
        ranked_variances = [variances[i] for i in ranked]
        assert all(
            ranked_variances[i] >= ranked_variances[i + 1]
            for i in range(len(ranked_variances) - 1)
        )

    def test_all_dead_returns_empty(self):
        acts = np.zeros((50, 8), dtype=np.float32)
        ranked = rank_features_by_variance(acts)
        assert ranked == []
