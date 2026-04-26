"""Pydantic configuration models for the SLIDERS project."""

import importlib
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SAEConfig(StrictModel):
    hidden_dim: int = Field(default=8192, ge=256, le=65536)
    lambda_sparsity: float = Field(default=1e-3, ge=0, le=1)
    topk: int = Field(default=0, ge=0)
    loss_type: str = Field(default="mse", pattern="^(mse|cosine)$")
    lr: float = Field(default=3e-4, gt=0)
    epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=512, ge=1)
    val_split: float = Field(default=0.1, gt=0, lt=1)
    patience: int = Field(default=10, ge=1)
    dead_threshold_steps: int = Field(default=1000, ge=1)

    @field_validator("topk")
    @classmethod
    def topk_reasonable(cls, v: int, info: ValidationInfo) -> int:
        hidden_dim = info.data.get("hidden_dim", 8192)
        if v > 0 and v > hidden_dim // 10:
            raise ValueError(f"topk={v} is too large for hidden_dim={hidden_dim}")
        return v


class DatasetConfig(StrictModel):
    name: str
    path: Path
    batch_size: int = 64
    adapter: str = "generic"


class EncoderConfig(StrictModel):
    use_patches: bool = False


class RetrievalConfig(StrictModel):
    n_sliders: int = Field(default=20, ge=1, le=100)


class NamingConfig(StrictModel):
    n_features: int = Field(default=20, ge=1)
    n_crops: int = Field(default=8, ge=1)
    crop_size: int = Field(default=96, ge=16)
    ranking: str = Field(default="diverse_mmr", pattern="^(variance|diverse_mmr|sparsity|selectivity)$")
    lambda_mmr: float = Field(default=0.5, ge=0, le=1)
    vlm_model: str = "Qwen/Qwen3-VL-4B-Instruct"


class AppConfig(StrictModel):
    dataset: DatasetConfig
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    sae: SAEConfig = Field(default_factory=SAEConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    naming: NamingConfig = Field(default_factory=NamingConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        yaml = cast(Any, importlib.import_module("yaml"))

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
