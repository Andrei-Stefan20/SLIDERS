"""Tests for query steering."""

import numpy as np
import pytest

from src.retrieval.steering import steer_query
from src.utils.exceptions import EmbeddingDimensionMismatch, InvalidSliderConfig


@pytest.fixture
def unit_query():
    rng = np.random.default_rng(7)
    q = rng.standard_normal(32).astype(np.float32)
    return q / np.linalg.norm(q)


@pytest.fixture
def directions():
    rng = np.random.default_rng(8)
    d = rng.standard_normal((4, 32)).astype(np.float32)
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    return d / norms


class TestSteerQuery:
    def test_output_is_unit_norm(self, unit_query, directions):
        alphas = [0.5, -0.5, 1.0, 0.0]
        result = steer_query(unit_query, directions, alphas)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    def test_output_shape(self, unit_query, directions):
        alphas = [1.0, 0.0, -1.0, 0.5]
        result = steer_query(unit_query, directions, alphas)
        assert result.shape == unit_query.shape

    def test_zero_alphas_returns_unit_query(self, unit_query, directions):
        alphas = [0.0, 0.0, 0.0, 0.0]
        result = steer_query(unit_query, directions, alphas)
        assert np.allclose(result, unit_query, atol=1e-5)

    def test_result_is_float32(self, unit_query, directions):
        result = steer_query(unit_query, directions, [1.0, 0.0, 0.0, 0.0])
        assert result.dtype == np.float32

    def test_single_direction(self):
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        d = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        result = steer_query(q, d, [1.0])
        assert result.shape == (3,)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5


class TestSteerQueryValidation:
    def test_2d_query_raises(self, directions):
        q = np.ones((1, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            steer_query(q, directions, [1.0, 0.0, 0.0, 0.0])

    def test_1d_directions_raises(self, unit_query):
        d = np.ones(32, dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            steer_query(unit_query, d, [1.0])

    def test_dim_mismatch_raises(self, unit_query):
        wrong_dirs = np.ones((4, 16), dtype=np.float32)
        with pytest.raises(EmbeddingDimensionMismatch):
            steer_query(unit_query, wrong_dirs, [1.0, 0.0, 0.0, 0.0])

    def test_alpha_count_mismatch_raises(self, unit_query, directions):
        with pytest.raises(InvalidSliderConfig):
            steer_query(unit_query, directions, [1.0, 0.0])  # 2 alphas but 4 directions


class TestSteerQueryEdgeCases:
    def test_near_cancellation_returns_original(self):
        """If directions cancel the query to near-zero, return original."""
        q = np.array([1.0, 0.0], dtype=np.float32)
        d = np.array([[1.0, 0.0]], dtype=np.float32)
        result = steer_query(q, d, [-1.0 - 1e-9])
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    @pytest.mark.parametrize("alpha", [0.1, 1.0, 3.0, -3.0])
    def test_parametrized_alpha_magnitudes(self, unit_query, directions, alpha):
        result = steer_query(unit_query, directions[:1], [alpha])
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5
