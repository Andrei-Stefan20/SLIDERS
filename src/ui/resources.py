import json
from pathlib import Path

import numpy as np

from src.datasets import get_adapter
from src.encoders.dino_encoder import DINOEncoder
from src.models.sae import SparseAutoencoder
from src.naming.feature_namer import rank_features_by_variance
from src.retrieval.index import load_index
from src.retrieval.patch_retrieval import PatchCorpus
from src.ui.state import AppState
from src.utils.io import dataset_stem, patch_sidecar_paths
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


def _feature_sliders_from_names(names_path: Path, n_sliders: int):
    """(feature_ids, names, descriptions) from a feature_names.json, or empty lists."""
    if not names_path.exists():
        return [], [], []
    all_names = json.loads(names_path.read_text())
    feature_ids = [int(k) for k in list(all_names.keys())[:n_sliders]]
    vals = [all_names[str(fid)] for fid in feature_ids]
    names = [v["name"] if isinstance(v, dict) else v for v in vals]
    descs = [v.get("description", "") if isinstance(v, dict) else "" for v in vals]
    return feature_ids, names, descs


def _load_patch_resources(embeddings_path, index_path, sae_path, image_paths_json,
                          stem, adapter, n_sliders) -> AppState:
    """Patch corpus: MaxSim retrieval over a patch index, queries are patch-token sets.
    No CLS embeddings/activations/previews (those are image-level only)."""
    dino = DINOEncoder(use_patches=True)
    sae = SparseAutoencoder.load(sae_path)
    index = load_index(index_path)
    patch_corpus = PatchCorpus(embeddings_path)
    image_paths = json.loads(image_paths_json.read_text())
    image_classes = _build_image_class_metadata(image_paths, adapter)
    feature_ids, feature_names, feature_descriptions = _feature_sliders_from_names(
        sae_path.parent / f"{stem}_feature_names.json", n_sliders
    )
    logger.info(f"Patch corpus: {patch_corpus.n_images} images, "
                f"{len(patch_corpus.data)} patches, {len(feature_ids)} sliders")
    return AppState(
        dino=dino, sae=sae, index=index, sae_index=None, embeddings=None, activations=None,
        image_paths=image_paths, feature_ids=feature_ids, feature_names=feature_names,
        feature_descriptions=feature_descriptions, image_classes=image_classes,
        path_to_idx={p: i for i, p in enumerate(image_paths)}, patch_corpus=patch_corpus,
    )


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
        proc = Path("data/processed")
        embeddings_path = proc / f"{dataset}_embeddings.npy"
        image_paths_json = proc / f"{dataset}_image_paths.json"
        index_path = proc / f"{dataset}_index.faiss"
        sae_path = Path("models") / f"{dataset}_sae_best.pt"
    elif not embeddings_path.exists():
        discovered = _discover_processed(Path("data/processed"), "_embeddings.npy")
        if discovered:
            embeddings_path = discovered
            image_paths_json = discovered.parent / f"{dataset_stem(discovered)}_image_paths.json"
            logger.warning(f"Auto-discovered embeddings: {embeddings_path}")

    # Per-dataset prefix used to locate the pipeline artifacts on disk.
    stem = dataset if dataset is not None else dataset_stem(embeddings_path)

    adapter_name = adapter_name or dataset or "generic"
    adapter = get_adapter(adapter_name)

    if patch_sidecar_paths(embeddings_path)[1].exists():
        logger.info("Patch corpus detected: loading MaxSim retrieval resources.")
        return _load_patch_resources(
            embeddings_path, index_path, sae_path, image_paths_json, stem, adapter, n_sliders
        )

    dino = DINOEncoder(use_patches=False)

    sae = SparseAutoencoder.load(sae_path)

    proc = embeddings_path.parent

    def _pick(directory: Path, prefixed: str, legacy: str) -> Path:
        # Prefer the prefixed name; use the legacy non-prefixed one otherwise.
        cand = directory / prefixed
        if cand.exists():
            return cand
        legacy_path = directory / legacy
        return legacy_path if legacy_path.exists() else cand

    sae_index_path = _pick(index_path.parent, f"{stem}_sae_index.faiss", "sae_index.faiss")
    acts_path = _pick(proc, f"{stem}_activations.npy", "activations.npy")
    names_path = _pick(sae_path.parent, f"{stem}_feature_names.json", "feature_names.json")

    index = load_index(index_path)
    sae_index = None
    if sae_index_path.exists():
        sae_index = load_index(sae_index_path)

    embeddings = _load_array_smart(embeddings_path)
    image_paths = json.loads(image_paths_json.read_text())
    image_classes = _build_image_class_metadata(image_paths, adapter)

    activations = None
    if acts_path.exists():
        activations = _load_array_smart(acts_path)

    class_directions = class_names = None
    feature_ids = feature_names = feature_descriptions = []

    dirs_found = _find_first([
        (proc / f"{stem}_class_directions.npy", proc / f"{stem}_class_direction_names.json"),
        (proc / "class_directions.npy", proc / "class_direction_names.json"),
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
