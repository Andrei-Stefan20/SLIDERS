"""Tests for serialisation utilities."""


import numpy as np

from src.utils.io import (
    load_embeddings,
    load_feature_names,
    load_image_paths,
    normalize_embeddings,
    save_embeddings,
    save_feature_names,
    save_image_paths,
)


class TestNormalizeEmbeddings:
    def test_unit_norm_rows(self):
        rng = np.random.default_rng(0)
        embs = rng.standard_normal((50, 32)).astype(np.float32)
        normed = normalize_embeddings(embs)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, np.ones(50), atol=1e-5)

    def test_zero_row_unchanged(self):
        embs = np.zeros((4, 8), dtype=np.float32)
        normed = normalize_embeddings(embs)
        np.testing.assert_array_equal(normed, embs)

    def test_shape_preserved(self):
        embs = np.random.randn(10, 16).astype(np.float32)
        assert normalize_embeddings(embs).shape == embs.shape

    def test_single_row(self):
        emb = np.array([[3.0, 4.0]], dtype=np.float32)
        normed = normalize_embeddings(emb)
        np.testing.assert_allclose(np.linalg.norm(normed), 1.0, atol=1e-6)

    def test_already_unit_norm(self):
        embs = np.eye(8, dtype=np.float32)
        normed = normalize_embeddings(embs)
        np.testing.assert_allclose(normed, embs, atol=1e-6)


class TestSaveLoadEmbeddings:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "embs.npy"
        embs = np.random.randn(20, 32).astype(np.float32)
        save_embeddings(embs, path)
        loaded = load_embeddings(path)
        np.testing.assert_allclose(loaded, embs, atol=1e-6)

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "embs.npy"
        embs = np.ones((4, 8), dtype=np.float32)
        save_embeddings(embs, path)
        assert path.exists()

    def test_loaded_dtype_is_float32(self, tmp_path):
        path = tmp_path / "embs.npy"
        embs = np.ones((4, 8), dtype=np.float64)
        save_embeddings(embs, path)
        loaded = load_embeddings(path)
        assert loaded.dtype == np.float32


class TestSaveLoadImagePaths:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "paths.json"
        paths = ["/data/a.jpg", "/data/b.png", "/data/c.jpeg"]
        save_image_paths(paths, path)
        loaded = load_image_paths(path)
        assert loaded == paths

    def test_empty_list(self, tmp_path):
        path = tmp_path / "paths.json"
        save_image_paths([], path)
        assert load_image_paths(path) == []


class TestSaveLoadFeatureNames:
    def test_roundtrip_string_keys(self, tmp_path):
        path = tmp_path / "names.json"
        names = {"0": "fur texture", "1": "bright background", "42": "leaf edge"}
        save_feature_names(names, path)
        loaded = load_feature_names(path)
        assert loaded == names

    def test_integer_keys_normalised_to_strings(self, tmp_path):
        path = tmp_path / "names.json"
        names = {0: "fur", 1: "leaf"}
        save_feature_names(names, path)
        loaded = load_feature_names(path)
        assert "0" in loaded and "1" in loaded

    def test_empty_dict(self, tmp_path):
        path = tmp_path / "names.json"
        save_feature_names({}, path)
        assert load_feature_names(path) == {}
