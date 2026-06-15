import json
from pathlib import Path

import numpy as np


def dataset_stem(embeddings_path: Path | str) -> str:
    name = Path(embeddings_path).name
    if name.endswith("_embeddings.npy"):
        return name[: -len("_embeddings.npy")]
    return Path(embeddings_path).stem.replace("_embeddings", "")


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.where(norms > 0, norms, 1.0)


def save_embeddings(embeddings: np.ndarray, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings.astype(np.float32))


def load_embeddings(path: Path | str) -> np.ndarray:
    return np.load(path).astype(np.float32)


def save_image_paths(paths: list[str], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(paths, indent=2))


def load_image_paths(path: Path | str) -> list[str]:
    return json.loads(Path(path).read_text())


def save_feature_names(
    info: dict[str | int, dict[str, str]], path: Path | str
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): v for k, v in info.items()}, indent=2))


def load_feature_names(
    path: Path | str,
) -> dict[str, dict[str, str]]:
    return json.loads(Path(path).read_text())
