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

from src.config import AppConfig
from src.data.loader import ImageFolderFlat
from src.encoders.dino_encoder import DINOEncoder
from src.utils.io import save_embeddings, save_image_paths


def collate_fn(batch):
    images, paths = zip(*batch)
    return torch.stack(images), list(paths)


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
