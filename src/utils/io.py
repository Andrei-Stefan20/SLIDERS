"""Serialisation helpers for embeddings, image paths, and feature names."""

import json
from pathlib import Path

import numpy as np


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalise each row. Rows with zero norm are left unchanged.

    Args:
        embeddings: Float32 array of shape ``(N, D)``.

    Returns:
        Row-normalised array of the same shape.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.where(norms > 0, norms, 1.0)


def save_embeddings(embeddings: np.ndarray, path: Path | str) -> None:
    """Save a float32 embedding matrix to a .npy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings.astype(np.float32))


def load_embeddings(path: Path | str) -> np.ndarray:
    """Load a .npy embedding matrix, cast to float32."""
    return np.load(path).astype(np.float32)


def save_image_paths(paths: list[str], path: Path | str) -> None:
    """Persist an ordered list of image path strings as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(paths, indent=2))


def load_image_paths(path: Path | str) -> list[str]:
    """Load an ordered list of image path strings from a JSON file."""
    return json.loads(Path(path).read_text())


def save_feature_names(
    info: dict[str | int, dict[str, str]], path: Path | str
) -> None:
    """Saves feature names and descriptions to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): v for k, v in info.items()}, indent=2))


def load_feature_names(
    path: Path | str,
) -> dict[str, dict[str, str]]:
    """Loads feature names and descriptions from a JSON file."""
    return json.loads(Path(path).read_text())
