"""CLI script for extracting DINOv2 embeddings from a dataset.

Usage:
    python scripts/extract_embeddings.py --dataset plantvillage --output data/processed/
    python scripts/extract_embeddings.py --dataset plantvillage --val-split 0.2
"""

# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collections.abc import Iterable, Iterator

from src.config import AppConfig
from src.data.loader import ImageFolderFlat
from src.encoders.dino_encoder import DINOEncoder
from src.retrieval.patch_store import quantize_int8
from src.utils.io import (
    patch_scales_path,
    save_embeddings,
    save_image_paths,
    save_patch_sidecars,
)


def collate_fn(batch):
    images, paths = zip(*batch)
    return torch.stack(images), list(paths)


def _image_val_set(n_images: int, val_split: float) -> set[int]:
    """Image indices held out for validation (image-level split, seed 42)."""
    if val_split <= 0.0:
        return set()
    n_val = max(1, int(n_images * val_split))
    perm = np.random.default_rng(seed=42).permutation(n_images)
    return set(int(i) for i in perm[:n_val])


def _select_salient(block: np.ndarray, kept: int) -> np.ndarray:
    """Top-`kept` patches of one image by token L2 norm (high norm = distinctive content;
    the registers variant has already absorbed the artifact patches). Indices kept sorted
    so order is deterministic."""
    if kept >= block.shape[0]:
        return block
    top = np.sort(np.argsort(np.linalg.norm(block, axis=1))[::-1][:kept])
    return block[top]


def write_patch_memmaps(
    batches: Iterable[tuple[np.ndarray, list[str]]],
    n_images: int,
    output_dir: Path,
    dataset_name: str,
    val_split: float,
    dtype: np.dtype | type = np.float16,
    max_patches_per_image: int = 0,
) -> list[Path]:
    """Stream (patch_embeddings (B, P, D), paths) batches into per-split memmaps.

    Patches of one image are contiguous rows. Writes <ds>[_split]_patch_embeddings.npy
    plus the _image_ids.npy / _meta.json / _image_paths.json sidecars, one batch at a
    time. Returns the embeddings paths written.

    `dtype` (default float16) halves the footprint vs float32; `max_patches_per_image`>0
    keeps only that many salient patches per image (by token norm), cutting it further.
    """
    val_set = _image_val_set(n_images, val_split)
    if val_split > 0.0:
        sizes = {"train": n_images - len(val_set), "val": len(val_set)}
    else:
        sizes = {"": n_images}

    is_int8 = np.dtype(dtype) == np.int8
    writers: dict[str, dict] = {}
    grid = kept = None
    g = 0

    for emb, paths in batches:
        if grid is None:
            _, patches_per_image, dim = emb.shape
            grid = int(np.sqrt(patches_per_image))
            kept = min(max_patches_per_image, patches_per_image) if max_patches_per_image else patches_per_image
            for split, n_img in sizes.items():
                suffix = f"_{split}" if split else ""
                emb_path = output_dir / f"{dataset_name}{suffix}_patch_embeddings.npy"
                writers[split] = {
                    "path": emb_path,
                    "mm": np.lib.format.open_memmap(
                        emb_path, mode="w+", dtype=dtype, shape=(n_img * kept, dim),
                    ),
                    "ids": np.empty(n_img * kept, dtype=np.int32),
                    "scales": np.empty(n_img * kept, dtype=np.float32) if is_int8 else None,
                    "cursor": 0,
                    "paths": [],
                }

        for b in range(emb.shape[0]):
            split = "val" if g in val_set else ("train" if val_split > 0.0 else "")
            w = writers[split]
            r0 = w["cursor"]
            sel = _select_salient(emb[b], kept)
            if is_int8:
                codes, scales = quantize_int8(sel)
                w["mm"][r0 : r0 + kept] = codes
                w["scales"][r0 : r0 + kept] = scales
            else:
                w["mm"][r0 : r0 + kept] = sel.astype(dtype)
            w["ids"][r0 : r0 + kept] = len(w["paths"])
            w["cursor"] += kept
            w["paths"].append(paths[b])
            g += 1

    written: list[Path] = []
    for split, w in writers.items():
        w["mm"].flush()
        suffix = f"_{split}" if split else ""
        save_image_paths(
            w["paths"], output_dir / f"{dataset_name}{suffix}_patch_image_paths.json"
        )
        if is_int8:
            np.save(patch_scales_path(w["path"]), w["scales"])
        save_patch_sidecars(
            w["path"], w["ids"],
            {"grid_size": grid, "patches_per_image": kept, "n_images": len(w["paths"]),
             "patch_dtype": str(np.dtype(dtype))},
        )
        print(f"Saved {split or 'all'} patches {w['mm'].shape} ({np.dtype(dtype)}) -> {w['path']}")
        written.append(w["path"])
    return written


def _encoded_batches(loader, encoder) -> Iterator[tuple[np.ndarray, list[str]]]:
    for images, paths in tqdm(loader, desc="Extracting patch embeddings"):
        yield encoder.encode(images).numpy(), list(paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 embeddings from an image dataset."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name (used for output filenames, e.g. plantvillage).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config file. CLI arguments override matching config values.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to raw dataset root (defaults to data/raw/<dataset>).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Directory where embeddings and image paths are saved.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Inference batch size."
    )
    parser.add_argument(
        "--use-patches",
        action="store_true",
        help="Extract patch tokens instead of CLS token.",
    )
    parser.add_argument(
        "--patch-dtype", choices=["float16", "float32", "int8"], default="float16",
        help="Storage dtype for the patch memmap: float16 (~22 GB) halves vs float32; "
             "int8 (~11 GB) quantizes per-row (a scales sidecar is written, dequantized on read).",
    )
    parser.add_argument(
        "--max-patches-per-image", type=int, default=0,
        help="Keep only the N most salient patches per image (0 = all 256). "
             "E.g. 64 cuts the footprint 4x at some retrieval-recall cost.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.0,
        help="Fraction of images held out as validation set (e.g. 0.2). "
             "When set, saves <dataset>_train_embeddings.npy and <dataset>_val_embeddings.npy "
             "instead of a single <dataset>_embeddings.npy.",
    )
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config) if args.config is not None else None
    dataset_name = args.dataset or (cfg.dataset.name if cfg else None)
    if dataset_name is None:
        parser.error("--dataset is required when --config is not provided")

    batch_size = args.batch_size or (cfg.dataset.batch_size if cfg else 64)
    input_dir = args.input or (cfg.dataset.path if cfg else Path("data/raw") / dataset_name)
    use_patches = args.use_patches or (cfg.encoder.use_patches if cfg else False)

    if not torch.cuda.is_available():
        if batch_size == 64:  # only override if still at default
            batch_size = 32
            print("Non-CUDA device detected - batch size reduced to 32")

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading images from: {input_dir}")
    dataset = ImageFolderFlat(input_dir)
    print(f"  Found {len(dataset)} images.")
    if len(dataset) == 0:
        print(f"No images found in {input_dir}. Download the dataset first.")
        return

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    encoder = DINOEncoder(use_patches=use_patches)

    if use_patches:
        write_patch_memmaps(
            _encoded_batches(loader, encoder),
            n_images=len(dataset),
            output_dir=output_dir,
            dataset_name=dataset_name,
            val_split=args.val_split,
            dtype={"float16": np.float16, "float32": np.float32, "int8": np.int8}[args.patch_dtype],
            max_patches_per_image=args.max_patches_per_image,
        )
        return

    all_embeddings: list[np.ndarray] = []
    all_paths: list[str] = []

    for images, paths in tqdm(loader, desc="Extracting embeddings"):
        emb = encoder.encode(images).numpy()
        all_embeddings.append(emb)
        all_paths.extend(paths)

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    if args.val_split > 0.0:
        n_val = max(1, int(len(embeddings) * args.val_split))
        rng = np.random.default_rng(seed=42)
        indices = rng.permutation(len(embeddings))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        splits = {"train": train_idx, "val": val_idx}
        for split, idx in splits.items():
            emb_path = output_dir / f"{dataset_name}_{split}_embeddings.npy"
            paths_path = output_dir / f"{dataset_name}_{split}_image_paths.json"
            save_embeddings(embeddings[idx], emb_path)
            save_image_paths([all_paths[i] for i in idx], paths_path)
            print(f"Saved {split} embeddings ({embeddings[idx].shape}) -> {emb_path}")
            print(f"Saved {split} image paths ({len(idx)})  -> {paths_path}")
    else:
        emb_path = output_dir / f"{dataset_name}_embeddings.npy"
        paths_path = output_dir / f"{dataset_name}_image_paths.json"
        save_embeddings(embeddings, emb_path)
        save_image_paths(all_paths, paths_path)
        print(f"Saved embeddings ({embeddings.shape}) -> {emb_path}")
        print(f"Saved image paths ({len(all_paths)})  -> {paths_path}")


if __name__ == "__main__":
    main()
