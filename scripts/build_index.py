# ruff: noqa: E402
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.index import build_index, build_sae_index, save_index
from src.utils.io import dataset_stem, normalize_embeddings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Primary index path. Default: <dir>/<dataset>_index.faiss derived from --embeddings.",
    )
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument(
        "--sae-model", type=Path, default=None,
        help="If provided, also builds a SAE-space index and saves <dataset>_activations.npy.",
    )
    parser.add_argument("--sae-index", type=Path, default=None)
    args = parser.parse_args()

    stem = dataset_stem(args.embeddings)
    proc = args.embeddings.parent
    args.output = args.output or proc / f"{stem}_index.faiss"

    embeddings = np.load(args.embeddings).astype(np.float32)
    logger.info(f"Loaded {embeddings.shape} from {args.embeddings}")

    if embeddings.shape[0] == 0:
        raise ValueError("Embeddings array is empty.")

    if not args.no_normalize:
        embeddings = normalize_embeddings(embeddings)

    index = build_index(embeddings)
    save_index(index, args.output)
    logger.info(f"Primary index → {args.output}  ({index.ntotal} vectors, dim={index.d})")

    if args.sae_model is not None:
        from src.models.sae import SparseAutoencoder

        sae = SparseAutoencoder.load(args.sae_model)

        logger.info(f"Computing SAE activations for {len(embeddings)} embeddings...")
        all_acts = []
        with torch.no_grad():
            for start in range(0, len(embeddings), 1024):
                batch = torch.from_numpy(embeddings[start : start + 1024])
                all_acts.append(sae.encode(batch).numpy())
        activations = np.concatenate(all_acts, axis=0)

        sae_index = build_sae_index(activations)
        sae_out = args.sae_index or proc / f"{stem}_sae_index.faiss"
        save_index(sae_index, sae_out)
        logger.info(f"SAE-space index → {sae_out}  ({sae_index.ntotal} vectors, dim={sae_index.d})")

        acts_path = proc / f"{stem}_activations.npy"
        np.save(acts_path, activations.astype(np.float32))
        logger.info(f"Activations → {acts_path}  shape={activations.shape}")


if __name__ == "__main__":
    main()
