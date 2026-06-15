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
from src.evaluation.recall_at_k import mean_average_precision, mean_precision_at_k, mean_recall_at_k
from src.evaluation.retrieval_comparison import compare_retrieval_methods, print_comparison_table
from src.evaluation.steering_faithfulness import (
    batch_steering_faithfulness,
    direction_steering_faithfulness,
)
from src.evaluation.steering_isotonicity import batch_steering_isotonicity
from src.evaluation.targeted_recall import batch_targeted_recall
from src.models.sae import SparseAutoencoder
from src.naming.feature_namer import get_top_images, rank_features_by_variance
from src.retrieval.index import load_index
from src.retrieval.query import search
from src.utils.io import normalize_embeddings
from src.utils.logging import setup_logging

RECALL_KS = [1, 5, 10]


def build_same_class_ground_truth(image_paths: list[str]) -> list[list[int]]:
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
        "--skip-isotonicity",
        action="store_true",
        help="Skip steering isotonicity (tests monotone controllability of sliders).",
    )
    parser.add_argument(
        "--skip-targeted-recall",
        action="store_true",
        help="Skip targeted class delta-Recall@K (tests whether sliders improve class retrieval).",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip retrieval method comparison (Unsteered vs PCA vs SAE, with P@K and mAP).",
    )
    parser.add_argument(
        "--comparison-queries",
        type=int,
        default=200,
        help="Number of queries for the retrieval comparison.",
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

    print("\n=== Recall@K / Precision@K / mAP (unsteered baseline) ===")
    ground_truth = build_same_class_ground_truth(image_paths)
    recall_depth = max(RECALL_KS)
    results = []
    for i in range(len(norm_embs)):
        _, retrieved = search(index, norm_embs[i], k=recall_depth + 1)
        mask = retrieved != i
        retrieved_filtered = retrieved[mask][:recall_depth].tolist()
        results.append({"retrieved": retrieved_filtered, "relevant": ground_truth[i]})
    recall = mean_recall_at_k(results, k_values=RECALL_KS)
    precision = mean_precision_at_k(results, k_values=RECALL_KS)
    map_score = mean_average_precision(results, k=recall_depth)
    for k in RECALL_KS:
        print(f"  recall@{k}: {recall[f'recall@{k}']:.4f}   precision@{k}: {precision[f'precision@{k}']:.4f}")
    print(f"  mAP@{recall_depth}: {map_score:.4f}")

    print("\nComputing SAE activations...")
    all_acts = []
    with torch.no_grad():
        for start in range(0, len(embeddings), 1024):
            batch = torch.from_numpy(embeddings[start : start + 1024])
            all_acts.append(sae.encode(batch).numpy())
    activations = np.concatenate(all_acts, axis=0)

    ranked = rank_features_by_variance(activations)[: args.n_align_features]

    print("\n=== Monosemanticity ===")
    mono_scores = batch_monosemanticity(activations, image_paths, ranked, k=args.top_k)
    for m in sorted(mono_scores, key=lambda x: -x["purity_score"])[:10]:
        print(f"  [{m['feature_id']:5d}]  purity={m['purity_score']:.3f}  "
              f"dominant={m['dominant_class'][:30]:30s}  ({m['dominant_class_fraction']:.0%})")
    print(f"  Mean purity: {mean_purity(mono_scores):.4f}")

    if not args.skip_comparison:
        print("\n=== Retrieval Method Comparison ===")
        rng_cmp = np.random.default_rng(1)
        cmp_indices = rng_cmp.choice(
            len(norm_embs),
            size=min(args.comparison_queries, len(norm_embs)),
            replace=False,
        ).tolist()
        comparison = compare_retrieval_methods(
            index, norm_embs, image_paths, sae, activations,
            query_indices=cmp_indices,
            k_values=(5, 10),
            steer_alpha=args.faithfulness_alpha,
        )
        print_comparison_table(comparison)

    if not args.skip_targeted_recall:
        print("\n=== Targeted Class Delta-Recall@K ===")
        targeted_results = batch_targeted_recall(
            sae, index, activations, norm_embs, image_paths,
            mono_scores, alpha=args.faithfulness_alpha,
            k=args.top_k, n_queries=args.faithfulness_queries,
        )
        for r in sorted(targeted_results, key=lambda x: -x["delta_recall"]):
            sign = "+" if r["delta_recall"] >= 0 else ""
            print(f"  [{r['feature_id']:5d}]  class={r['dominant_class'][:30]:30s}  "
                  f"base={r['baseline_recall']:.3f}  "
                  f"steered={r['steered_recall']:.3f}  "
                  f"delta={sign}{r['delta_recall']:.3f}")
        mean_delta = float(np.mean([r["delta_recall"] for r in targeted_results]))
        print(f"  Mean delta-Recall@{args.top_k}: {'+' if mean_delta >= 0 else ''}{mean_delta:.4f}  "
              f"(>0 = slider helps find the right class)")

    need_queries = (
        not args.skip_faithfulness
        or not args.skip_isotonicity
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
              f"(>1.0 = steering works, ~1.0 = no effect)")

    if not args.skip_isotonicity:
        print("\n=== Steering Isotonicity ===")
        iso_results = batch_steering_isotonicity(
            sae, index, activations, query_sample,
            ranked, k=args.top_k, n_queries=min(50, args.faithfulness_queries),
        )
        for r in sorted(iso_results, key=lambda x: x["spearman_rho"]):
            bar = "OK" if r["spearman_rho"] >= 0.7 else "!!"
            print(f"  [{r['feature_id']:5d}]  isotonicity_rho={r['spearman_rho']:.3f}  [{bar}]")
        mean_rho = float(np.mean([r["spearman_rho"] for r in iso_results]))
        print(f"  Mean isotonicity rho: {mean_rho:.4f}  "
              f"(1.0 = perfectly monotone, <0.7 = unreliable slider)")

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
            print("  WARNING: SAE barely outperforms PCA - consider longer training.")

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
