"""CLI script for evaluating retrieval quality.

Metrics computed:
  - Recall@K      — same-class ground truth (measures backbone quality)
  - CLIP alignment — how well feature names describe top-activating images
  - Steering faithfulness — whether sliders actually shift feature activations
  - Direction faithfulness — same check for class-direction sliders
  - Monosemanticity — class purity of each SAE feature's top images
  - SAE vs PCA ablation — whether SAE adds value over linear decomposition
  - Cross-model alignment — name validation with a different CLIP model
  - Reverse retrieval — use name as text query, measure image overlap

Usage:
    python scripts/evaluate.py \\
        --embeddings data/processed/plantvillage_embeddings.npy \\
        --image-paths data/processed/plantvillage_image_paths.json \\
        --index data/processed/index.faiss \\
        --sae-model models/sae_best.pt \\
        --feature-names models/feature_names.json
"""

# ruff: noqa: E402

import argparse
import json
import sys
from pathlib import Path, PurePath

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.encoders.clip_encoder import CLIPEncoder
from src.evaluation.ablation import evaluate_sae_vs_pca
from src.evaluation.clip_alignment import batch_clip_alignment
from src.evaluation.cross_model_alignment import batch_cross_validation
from src.evaluation.monosemanticity import batch_monosemanticity, mean_purity
from src.evaluation.recall_at_k import mean_recall_at_k
from src.evaluation.steering_faithfulness import (
    batch_steering_faithfulness,
    direction_steering_faithfulness,
)
from src.models.sae import SparseAutoencoder
from src.naming.feature_namer import get_top_images, rank_features_by_variance
from src.retrieval.index import load_index
from src.retrieval.query import search
from src.utils.io import normalize_embeddings
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# k-values used exclusively for Recall@K — independent of --top-k
RECALL_KS = [1, 5, 10]


def build_same_class_ground_truth(image_paths: list[str]) -> list[list[int]]:
    """For each image, return the indices of all other images in the same class directory."""
    labels = [PurePath(p).parent.name for p in image_paths]
    label_to_indices: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(i)

    return [
        [j for j in label_to_indices[labels[i]] if j != i]
        for i in range(len(image_paths))
    ]


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Evaluate SLIDERS retrieval.")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--image-paths", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--sae-model", type=Path, required=True)
    parser.add_argument("--feature-names", type=Path, default=None)
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="k for monosemanticity, faithfulness, and alignment metrics. "
             "Does NOT affect Recall@K cutoffs (always [1, 5, 10]).",
    )
    parser.add_argument("--n-align-features", type=int, default=20)
    parser.add_argument(
        "--faithfulness-alpha",
        type=float,
        default=2.0,
        help="Steering alpha for faithfulness evaluation.",
    )
    parser.add_argument(
        "--faithfulness-queries",
        type=int,
        default=100,
        help="Number of queries to average for faithfulness.",
    )
    parser.add_argument(
        "--n-pca-components",
        type=int,
        default=20,
        help="PCA components for ablation baseline.",
    )
    parser.add_argument(
        "--class-directions",
        type=Path,
        default=None,
        help="Path to .npy class directions (from compute_class_directions.py) "
             "for direction steering faithfulness evaluation.",
    )
    parser.add_argument(
        "--skip-faithfulness",
        action="store_true",
        help="Skip SAE steering faithfulness (slow for large corpora).",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip SAE vs PCA ablation (slow).",
    )
    parser.add_argument(
        "--cross-model-validator",
        type=str,
        default="ViT-B-32",
        help="CLIP model for cross-model name validation.",
    )
    args = parser.parse_args()

    embeddings = np.load(args.embeddings).astype(np.float32)
    image_paths = json.loads(args.image_paths.read_text())
    index = load_index(args.index)

    state = torch.load(args.sae_model, map_location="cpu", weights_only=True)
    input_dim = embeddings.shape[1]
    hidden_dim = state["encoder.weight"].shape[0]
    sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    sae.load_state_dict(state)
    sae.eval()

    norm_embs = normalize_embeddings(embeddings)

    # --- Recall@K -------------------------------------------------------
    # Search depth is always max(RECALL_KS) regardless of --top-k.
    print("\n=== Recall@K ===")
    ground_truth = build_same_class_ground_truth(image_paths)
    recall_depth = max(RECALL_KS)
    results = []
    for i in range(len(norm_embs)):
        _, retrieved = search(index, norm_embs[i], k=recall_depth + 1)
        mask = retrieved != i
        retrieved_filtered = retrieved[mask][:recall_depth].tolist()
        results.append({"retrieved": retrieved_filtered, "relevant": ground_truth[i]})
    recall = mean_recall_at_k(results, k_values=RECALL_KS)
    for k, v in recall.items():
        print(f"  {k}: {v:.4f}")

    # --- SAE activations ------------------------------------------------
    print("\nComputing SAE activations...")
    all_acts = []
    with torch.no_grad():
        for start in range(0, len(embeddings), 1024):
            batch = torch.from_numpy(embeddings[start : start + 1024])
            all_acts.append(sae.encode(batch).numpy())
    activations = np.concatenate(all_acts, axis=0)

    ranked = rank_features_by_variance(activations)[: args.n_align_features]

    # --- Monosemanticity ------------------------------------------------
    print("\n=== Monosemanticity ===")
    mono_scores = batch_monosemanticity(activations, image_paths, ranked, k=args.top_k)
    for m in sorted(mono_scores, key=lambda x: -x["purity_score"])[:10]:
        print(f"  [{m['feature_id']:5d}]  purity={m['purity_score']:.3f}  "
              f"dominant={m['dominant_class'][:30]:30s}  ({m['dominant_class_fraction']:.0%})")
    print(f"  Mean purity: {mean_purity(mono_scores):.4f}")

    # Shared query sample — needed by faithfulness, ablation, and direction faithfulness.
    need_queries = (
        not args.skip_faithfulness
        or not args.skip_ablation
        or args.class_directions is not None
    )
    query_sample: np.ndarray | None = None
    if need_queries:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(
            len(norm_embs),
            size=min(args.faithfulness_queries, len(norm_embs)),
            replace=False,
        )
        query_sample = norm_embs[sample_idx]

    # --- Steering Faithfulness ------------------------------------------
    if not args.skip_faithfulness:
        print("\n=== Steering Faithfulness ===")
        faith_scores = batch_steering_faithfulness(
            sae, index, activations, query_sample,
            ranked, alpha=args.faithfulness_alpha,
            k=args.top_k, n_queries=args.faithfulness_queries,
        )
        sorted_faith = sorted(faith_scores.items(), key=lambda x: -x[1])
        for fid, score in sorted_faith[:10]:
            print(f"  [{fid:5d}]  faithfulness={score:.3f}")
        print(f"  Mean faithfulness: {np.mean(list(faith_scores.values())):.4f}  "
              f"(>1.0 = steering works, ≈1.0 = no effect)")

    # --- SAE vs PCA Ablation --------------------------------------------
    # Independent of --skip-faithfulness: ablation can run on its own.
    if not args.skip_ablation:
        print("\n=== SAE vs PCA Ablation ===")
        ablation = evaluate_sae_vs_pca(
            sae, index, norm_embs, activations, query_sample,
            ranked, n_pca_components=args.n_pca_components,
            alpha=args.faithfulness_alpha, k=args.top_k,
            n_queries=args.faithfulness_queries,
        )
        print(f"  SAE mean faithfulness : {ablation['sae_mean_faithfulness']:.4f}")
        print(f"  PCA mean faithfulness : {ablation['pca_mean_faithfulness']:.4f}")
        print(f"  Improvement ratio     : {ablation['improvement_ratio']:.3f}x")
        if ablation["improvement_ratio"] < 1.1:
            print("  WARNING: SAE barely outperforms PCA — consider longer training.")

    # --- Direction Steering Faithfulness --------------------------------
    if args.class_directions is not None:
        print("\n=== Direction Steering Faithfulness ===")
        directions = np.load(args.class_directions).astype(np.float32)
        dir_scores: list[float] = []
        for i in range(len(directions)):
            score = direction_steering_faithfulness(
                index, norm_embs, directions, i, query_sample,
                alpha=args.faithfulness_alpha,
                k=args.top_k,
                n_queries=args.faithfulness_queries,
            )
            dir_scores.append(score)
            print(f"  [direction {i:3d}]  faithfulness={score:.3f}")
        print(f"  Mean direction faithfulness: {float(np.mean(dir_scores)):.4f}  "
              f"(>1.0 = directions steer retrieval)")

    # --- CLIP Alignment / Cross-Model / Reverse Retrieval ---------------
    if args.feature_names is not None:
        print("\n=== CLIP Alignment ===")
        feature_names_map: dict = json.loads(args.feature_names.read_text())
        clip_enc = CLIPEncoder()

        named_features = []
        for fid in ranked:
            name = feature_names_map.get(str(fid), f"Feature {fid}")
            fi = get_top_images(activations, image_paths, fid, k=args.top_k)
            named_features.append({"feature_id": fid, "name": name, "top_paths": fi.top_paths})

        alignment_scores = batch_clip_alignment(named_features, clip_enc)
        mean_align = float(np.mean(list(alignment_scores.values())))

        for fid, score in sorted(alignment_scores.items(), key=lambda x: -x[1]):
            name = feature_names_map.get(str(fid), f"Feature {fid}")
            print(f"  [{fid:5d}] {name:40s}  {score:.4f}")
        print(f"\n  Mean alignment: {mean_align:.4f}")

        print(f"\n=== Cross-Model Alignment ({args.cross_model_validator}) ===")
        cross_results = batch_cross_validation(
            named_features, image_paths, clip_enc,
            validator_model=args.cross_model_validator, k=args.top_k,
        )
        for r in sorted(cross_results, key=lambda x: -x["cross_model_score"])[:10]:
            print(f"  [{r['feature_id']:5d}] {r['name']:40s}  "
                  f"cross={r['cross_model_score']:.4f}  "
                  f"reverse={r['reverse_retrieval_score']:.4f}")
        mean_cross = float(np.mean([r["cross_model_score"] for r in cross_results]))
        mean_reverse = float(np.mean([r["reverse_retrieval_score"] for r in cross_results]))
        print(f"  Mean cross-model: {mean_cross:.4f}  "
              f"Mean reverse-retrieval: {mean_reverse:.4f}")


if __name__ == "__main__":
    main()
