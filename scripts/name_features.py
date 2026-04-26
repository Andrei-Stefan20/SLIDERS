"""CLI script for naming SAE features using VLM contrastive analysis.

Usage:
    python scripts/name_features.py \\
        --embeddings data/processed/plantvillage_embeddings.npy \\
        --image-paths data/processed/plantvillage_image_paths.json \\
        --sae-model models/sae_best.pt \\
        --output models/feature_names.json \\
        --n-features 20
"""

# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import AppConfig
from src.models.sae import SparseAutoencoder
from src.naming.feature_namer import get_top_images, rank_features_by_variance


def main() -> None:
    parser = argparse.ArgumentParser(description="Name SAE features with VLM contrastive analysis.")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--image-paths", type=Path, required=True)
    parser.add_argument("--sae-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/feature_names.json"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--n-features", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None, help="TopK used during SAE training (0 = ReLU).")
    parser.add_argument(
        "--ranking",
        type=str,
        default=None,
        choices=["variance", "diverse_mmr", "sparsity", "selectivity"],
        help="Feature selection strategy.",
    )
    parser.add_argument("--lambda-mmr", type=float, default=None)
    parser.add_argument("--vlm-model", type=str, default=None)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--n-crops", type=int, default=None)
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config) if args.config is not None else None
    naming_cfg = cfg.naming if cfg else None
    sae_cfg = cfg.sae if cfg else None
    n_features = args.n_features or (naming_cfg.n_features if naming_cfg else 20)
    topk = args.topk if args.topk is not None else (sae_cfg.topk if sae_cfg else 0)
    ranking = args.ranking or (naming_cfg.ranking if naming_cfg else "diverse_mmr")
    lambda_mmr = args.lambda_mmr if args.lambda_mmr is not None else (
        naming_cfg.lambda_mmr if naming_cfg else 0.5
    )
    vlm_model = args.vlm_model or (
        naming_cfg.vlm_model if naming_cfg else "Qwen/Qwen3-VL-4B-Instruct"
    )
    crop_size = args.crop_size or (naming_cfg.crop_size if naming_cfg else 96)
    n_crops = args.n_crops or (naming_cfg.n_crops if naming_cfg else 8)

    embeddings = np.load(args.embeddings).astype(np.float32)
    image_paths = json.loads(args.image_paths.read_text())

    state = torch.load(args.sae_model, map_location="cpu", weights_only=True)
    input_dim = embeddings.shape[1]
    hidden_dim = state["encoder.weight"].shape[0]
    sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, topk=topk)
    sae.load_state_dict(state)
    sae.eval()

    batch_size = 1024
    all_acts = []
    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            batch = torch.from_numpy(embeddings[start : start + batch_size])
            all_acts.append(sae.encode(batch).numpy())
    activations = np.concatenate(all_acts, axis=0)

    if ranking == "diverse_mmr":
        from src.naming.feature_ranking import rank_diverse_mmr
        ranked_features = rank_diverse_mmr(
            activations, n_features=n_features, lambda_mmr=lambda_mmr
        )
    elif ranking == "selectivity":
        from pathlib import PurePath
        from src.naming.feature_ranking import rank_by_selectivity_mmr
        class_labels = np.array([PurePath(p).parent.name for p in image_paths])
        ranked_features = rank_by_selectivity_mmr(
            activations, class_labels, n_features=n_features, lambda_mmr=lambda_mmr
        )
    elif ranking == "sparsity":
        from src.naming.feature_ranking import compute_sparsity
        sparsity = compute_sparsity(activations)
        valid = np.where((sparsity >= 0.90) & (sparsity <= 0.995))[0]
        max_acts = activations[:, valid].max(axis=0)
        top_idx = np.argsort(max_acts)[::-1][:n_features]
        ranked_features = list(valid[top_idx])
    else:
        ranked_features = rank_features_by_variance(activations)[:n_features]

    print(f"Naming {len(ranked_features)} features (ranking={ranking})...")

    from src.encoders.dino_encoder import DINOEncoder
    from src.naming.spatial_localization import localize_feature_batch
    from src.naming.vlm_namer import VLMFeatureNamer

    dino = DINOEncoder(use_patches=True)
    vlm = VLMFeatureNamer(model=vlm_model)

    feature_info: dict[str, dict] = {}
    for fid in ranked_features:
        fi = get_top_images(activations, image_paths, fid, k=n_crops)
        top_crops = localize_feature_batch(fi.top_paths, dino, sae, fid, crop_size)
        bot_crops = localize_feature_batch(fi.bottom_paths, dino, sae, fid, crop_size)
        name, desc = vlm.name_feature(top_crops, bot_crops)
        feature_info[str(fid)] = {"name": name, "description": desc}
        print(f"  feature {fid:5d} -> {name!r}")
        print(f"             {desc}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(feature_info, indent=2))
    print(f"\nSaved feature names -> {args.output}")


if __name__ == "__main__":
    main()
