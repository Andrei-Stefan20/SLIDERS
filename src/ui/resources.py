import json
from pathlib import Path

import numpy as np
import torch

from src.datasets import get_adapter
from src.encoders.dino_encoder import DINOEncoder
from src.models.sae import SparseAutoencoder
from src.naming.feature_namer import rank_features_by_variance
from src.retrieval.index import load_index
from src.ui.state import AppState
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_INDEX_PATH = Path("data/processed/index.faiss")
DEFAULT_SAE_PATH = Path("models/sae_best.pt")
DEFAULT_EMBEDDINGS_PATH = Path("data/processed/embeddings.npy")
DEFAULT_IMAGE_PATHS_JSON = Path("data/processed/image_paths.json")

RAM_LOAD_MAX_BYTES = 4 * 1024 ** 3


def _load_array_smart(path: Path) -> np.ndarray:
    size = path.stat().st_size
    if size <= RAM_LOAD_MAX_BYTES:
        return np.load(path)
    return np.load(path, mmap_mode="r")


def _find_first(candidates):
    for npy, jp in candidates:
        if npy.exists() and jp.exists():
            return npy, jp
    return None


def _build_image_class_metadata(image_paths, adapter):
    image_classes = []

    for path in image_paths:
        category, subcategory, _ = adapter.parse_path(path)
        category_name = str(category or "").strip()
        subcategory_name = str(subcategory or "").strip()
        if category_name and subcategory_name and subcategory_name != category_name:
            class_name = f"{category_name} / {subcategory_name}"
        else:
            class_name = category_name or subcategory_name or "unknown"
        image_classes.append(class_name)

    return image_classes


def _previews_from_directions(embeddings, image_paths, directions):
    emb = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_n = emb / np.clip(norms, 1e-8, None)
    sims = emb_n @ directions.T
    top, bottom = [], []
    for d in range(directions.shape[0]):
        col = sims[:, d]
        top.append(image_paths[int(np.argmax(col))])
        bottom.append(image_paths[int(np.argmin(col))])
    return top, bottom


def _previews_from_activations(activations, image_paths, feature_ids):
    top, bottom = [], []
    for fid in feature_ids:
        col = activations[:, fid]
        top.append(image_paths[int(np.argmax(col))])
        bottom.append(image_paths[int(np.argmin(col))])
    return top, bottom


def _discover_processed(processed_dir, suffix):
    candidates = sorted(processed_dir.glob(f"*{suffix}"))
    if candidates:
        return candidates[0]
    return None


def load_resources(
    index_path,
    sae_path,
    embeddings_path,
    image_paths_json,
    dataset,
    adapter_name=None,
    n_sliders=20,
):
    if dataset is not None:
        embeddings_path = Path(f"data/processed/{dataset}_embeddings.npy")
        image_paths_json = Path(f"data/processed/{dataset}_image_paths.json")
    else:
        proc_dir = Path("data/processed")
        if not embeddings_path.exists():
            discovered = _discover_processed(proc_dir, "_embeddings.npy")
            if discovered:
                embeddings_path = discovered
                stem = discovered.stem.replace("_embeddings", "")
                image_paths_json = proc_dir / f"{stem}_image_paths.json"
                logger.warning(f"Auto-discovered embeddings: {embeddings_path}")

    adapter_name = adapter_name or dataset or "generic"
    adapter = get_adapter(adapter_name)

    dino = DINOEncoder(use_patches=False)

    sae_state = torch.load(sae_path, map_location="cpu", weights_only=True)
    input_dim = sae_state["encoder.weight"].shape[1]
    hidden_dim = sae_state["encoder.weight"].shape[0]
    sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    sae.load_state_dict(sae_state)
    sae.eval()

    index = load_index(index_path)
    sae_index = None
    sae_index_path = index_path.parent / "sae_index.faiss"
    if sae_index_path.exists():
        sae_index = load_index(sae_index_path)

    embeddings = _load_array_smart(embeddings_path)
    image_paths = json.loads(image_paths_json.read_text())
    image_classes = _build_image_class_metadata(image_paths, adapter)

    activations = None
    acts_path = embeddings_path.parent / "activations.npy"
    if acts_path.exists():
        activations = _load_array_smart(acts_path)

    processed = Path("data/processed")

    class_directions = class_names = None
    feature_ids = feature_names = feature_descriptions = []

    dirs_found = _find_first([
        (processed / "class_directions.npy", processed / "class_direction_names.json"),
    ])
    if dirs_found:
        dir_npy, dir_json = dirs_found
        class_directions = np.load(dir_npy).astype(np.float32)
        class_names = json.loads(dir_json.read_text())
        feature_ids = list(range(len(class_names)))
        feature_names = class_names
        feature_descriptions = [""] * len(class_names)
        logger.info(f"Class directions: {len(class_names)} sliders")
    else:
        names_path = sae_path.parent / "feature_names.json"
        if names_path.exists():
            all_names = json.loads(names_path.read_text())
            feature_ids = [int(k) for k in list(all_names.keys())[:n_sliders]]
            _vals = [all_names[str(fid)] for fid in feature_ids]
            feature_names = [v["name"] if isinstance(v, dict) else v for v in _vals]
            feature_descriptions = [v.get("description", "") if isinstance(v, dict) else "" for v in _vals]
        elif activations is not None:
            feature_ids = rank_features_by_variance(activations)[:n_sliders]
            feature_names = [f"Feature {fid}" for fid in feature_ids]
            feature_descriptions = [""] * len(feature_ids)

    logger.info("Computing feature previews...")
    preview_top = preview_bottom = None
    if class_directions is not None:
        preview_top, preview_bottom = _previews_from_directions(
            embeddings, image_paths, class_directions
        )
    elif activations is not None and feature_ids:
        preview_top, preview_bottom = _previews_from_activations(
            activations, image_paths, feature_ids
        )
    logger.info("Previews ready.")

    path_to_idx = {p: i for i, p in enumerate(image_paths)}

    feature_scales = None
    if activations is not None:
        stds = np.asarray(activations, dtype=np.float32).std(axis=0)
        feature_scales = np.where(stds > 1e-6, stds, 1.0).astype(np.float32)

    return AppState(
        dino=dino,
        sae=sae,
        index=index,
        sae_index=sae_index,
        embeddings=embeddings,
        activations=activations,
        image_paths=image_paths,
        feature_ids=feature_ids,
        feature_names=feature_names,
        feature_descriptions=feature_descriptions,
        image_classes=image_classes,
        class_directions=class_directions,
        class_names=class_names,
        preview_top=preview_top,
        preview_bottom=preview_bottom,
        path_to_idx=path_to_idx,
        feature_scales=feature_scales,
    )
