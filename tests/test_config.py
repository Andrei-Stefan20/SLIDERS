"""Tests for Pydantic configuration models."""

from pathlib import Path

import pytest

from src.config import (
    AppConfig,
    DatasetConfig,
    EncoderConfig,
    NamingConfig,
    RetrievalConfig,
    SAEConfig,
)


class TestSAEConfig:
    def test_defaults(self):
        cfg = SAEConfig()
        assert cfg.hidden_dim == 8192
        assert cfg.lambda_sparsity == pytest.approx(1e-3)
        assert cfg.topk == 0
        assert cfg.loss_type == "mse"
        assert cfg.lr == pytest.approx(3e-4)
        assert cfg.epochs == 50

    def test_valid_topk(self):
        cfg = SAEConfig(hidden_dim=1000, topk=50)
        assert cfg.topk == 50

    def test_topk_too_large_raises(self):
        with pytest.raises(Exception):
            SAEConfig(hidden_dim=100, topk=20)

    def test_topk_zero_always_valid(self):
        cfg = SAEConfig(topk=0)
        assert cfg.topk == 0

    def test_invalid_loss_type_raises(self):
        with pytest.raises(Exception):
            SAEConfig(loss_type="l1")

    def test_cosine_loss_type_valid(self):
        cfg = SAEConfig(loss_type="cosine")
        assert cfg.loss_type == "cosine"

    def test_negative_lr_raises(self):
        with pytest.raises(Exception):
            SAEConfig(lr=-1e-3)

    def test_hidden_dim_bounds(self):
        with pytest.raises(Exception):
            SAEConfig(hidden_dim=10)
        with pytest.raises(Exception):
            SAEConfig(hidden_dim=100000)

    def test_val_split_bounds(self):
        with pytest.raises(Exception):
            SAEConfig(val_split=0.0)
        with pytest.raises(Exception):
            SAEConfig(val_split=1.0)


class TestDatasetConfig:
    def test_basic(self, tmp_path):
        cfg = DatasetConfig(name="test", path=tmp_path)
        assert cfg.name == "test"
        assert cfg.adapter == "generic"

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(Exception):
            DatasetConfig(name="test", path=tmp_path / "does_not_exist")


class TestRetrievalConfig:
    def test_defaults(self):
        cfg = RetrievalConfig()
        assert cfg.n_sliders == 20

    def test_n_sliders_bounds(self):
        with pytest.raises(Exception):
            RetrievalConfig(n_sliders=0)
        with pytest.raises(Exception):
            RetrievalConfig(n_sliders=200)


class TestEncoderConfig:
    def test_defaults(self):
        cfg = EncoderConfig()
        assert cfg.use_patches is False


class TestNamingConfig:
    def test_defaults(self):
        cfg = NamingConfig()
        assert cfg.n_features == 20
        assert cfg.vlm_model == "Qwen/Qwen3-VL-4B-Instruct"


class TestAppConfig:
    def test_from_yaml(self, tmp_path):
        yaml_content = f"""
            dataset:
            name: plantvillage
            path: {tmp_path}
            adapter: plantvillage
            sae:
            hidden_dim: 4096
            epochs: 30
            retrieval:
            n_sliders: 12
            naming:
            n_features: 12
            """
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        cfg = AppConfig.from_yaml(yaml_file)
        assert cfg.dataset.name == "plantvillage"
        assert cfg.sae.hidden_dim == 4096
        assert cfg.sae.epochs == 30
        assert cfg.retrieval.n_sliders == 12
        assert cfg.naming.n_features == 12

    def test_from_yaml_minimal(self, tmp_path):
        yaml_content = f"""
                    dataset:
                    name: test
                    path: {tmp_path}
                    """
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        cfg = AppConfig.from_yaml(yaml_file)
        assert cfg.sae.hidden_dim == 8192
        assert cfg.retrieval.n_sliders == 20

    def test_from_yaml_unknown_key_raises(self, tmp_path):
        yaml_content = f"""
                    dataset:
                    name: test
                    path: {tmp_path}
                    naming:
                    llm_model: gpt-4o
                    """
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(Exception):
            AppConfig.from_yaml(yaml_file)

    def test_from_yaml_missing_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            AppConfig.from_yaml(tmp_path / "nonexistent.yaml")
