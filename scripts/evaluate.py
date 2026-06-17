# ruff: noqa: E402
"""Run the full retrieval + SAE interpretability metric battery. See docs/EVALUATION.md."""

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
from src.evaluation.monosemanticity import (
    batch_monosemanticity,
    n_distinct_classes,
    shuffled_label_purity_baseline,
)
from src.evaluation.recall_at_k import mean_average_precision, mean_precision_at_k, mean_recall_at_k
from src.evaluation.retrieval_comparison import compare_retrieval_methods, print_comparison_table
from src.evaluation.stats import format_summary, summarize
from src.evaluation.steering_faithfulness import (
    batch_steering_faithfulness,
    direction_steering_faithfulness,
)
from src.evaluation.steering_isotonicity import batch_steering_isotonicity
from src.evaluation.steering_selectivity import batch_steering_selectivity
from src.evaluation.targeted_recall import batch_targeted_recall
from src.models.sae import SparseAutoencoder
from src.naming.feature_namer import get_top_images
from src.retrieval.index import load_index
from src.retrieval.query import search
from src.utils.io import normalize_embeddings
from src.utils.logging import setup_logging

RECALL_KS = [1, 5, 10]


def build_same_class_ground_truth(
    query_paths: list[str],
    corpus_paths: list[str] | None = None,
) -> list[list[int]]:
    """Same-label corpus indices for each query. corpus_paths=None means in-sample
    (drop the query's own index)."""
    if corpus_paths is None:
        corpus_paths = query_paths
        self_query = True
    else:
        self_query = False

    corpus_labels = [PurePath(p).parent.name for p in corpus_paths]
    label_to_indices: dict[str, list[int]] = {}
    for j, label in enumerate(corpus_labels):
        label_to_indices.setdefault(label, []).append(j)

    query_labels = [PurePath(p).parent.name for p in query_paths]
    out: list[list[int]] = []
    for i, label in enumerate(query_labels):
        rel = label_to_indices.get(label, [])
        out.append([j for j in rel if not (self_query and j == i)])
    return out


def select_eval_features(
    activations: np.ndarray,
    n_features: int,
    selection: str,
    seed: int = 0,
) -> list[int]:
    """Features for the per-feature metrics. 'random' samples live features (a fair
    sample of the dictionary); 'variance' is the legacy top-variance head."""
    live = [int(i) for i in np.flatnonzero(activations.var(axis=0) > 0)]
    if not live:
        return []
    if selection == "variance":
        from src.naming.feature_namer import rank_features_by_variance
        return rank_features_by_variance(activations)[:n_features]
    rng = np.random.default_rng(seed)
    n = min(n_features, len(live))
    return sorted(int(i) for i in rng.choice(live, size=n, replace=False))


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Evaluate SLIDERS retrieval.")
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="Corpus embeddings (the indexed set the UI searches).")
    parser.add_argument("--image-paths", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--sae-model", type=Path, required=True)
    parser.add_argument("--feature-names", type=Path, default=None)
    parser.add_argument(
        "--query-embeddings", type=Path, default=None,
        help="Held-out query embeddings (e.g. a validation split). If given, all "
             "query-based metrics use these instead of the indexed corpus, "
             "avoiding in-sample (train==test) optimism. Strongly recommended.",
    )
    parser.add_argument(
        "--query-image-paths", type=Path, default=None,
        help="Image paths for --query-embeddings (required if that is set).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Legacy fallback for --n-top-images / --retrieval-k when those are "
             "unset. Does NOT affect Recall@K cutoffs (always [1, 5, 10]).",
    )
    parser.add_argument(
        "--n-top-images", type=int, default=None,
        help="Top-activating images examined per feature for monosemanticity and "
             "CLIP alignment (defaults to --top-k).",
    )
    parser.add_argument(
        "--retrieval-k", type=int, default=None,
        help="Retrieval depth k for faithfulness / isotonicity / targeted recall "
             "(defaults to --top-k). Distinct from --n-top-images.",
    )
    parser.add_argument(
        "--n-eval-features", type=int, default=40,
        help="How many features the per-feature metrics are evaluated on.",
    )
    parser.add_argument(
        "--feature-selection", choices=["random", "variance"], default="random",
        help="random: representative sample of the dictionary (default). "
             "variance: legacy top-variance head (optimistic).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for feature/query sampling and bootstrap CIs.",
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
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write the full metrics as JSON here (consumed by scripts/make_report.py).",
    )
    args = parser.parse_args()
    report: dict = {}

    n_top_images = args.n_top_images if args.n_top_images is not None else args.top_k
    retrieval_k = args.retrieval_k if args.retrieval_k is not None else args.top_k

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
    n_classes = n_distinct_classes(image_paths)

    # Query set: held-out if provided, otherwise in-sample self-retrieval.
    if args.query_embeddings is not None:
        if args.query_image_paths is None:
            parser.error("--query-image-paths is required when --query-embeddings is set")
        query_norm = normalize_embeddings(np.load(args.query_embeddings).astype(np.float32))
        query_paths = json.loads(args.query_image_paths.read_text())
        held_out = True
        print(f"\nHeld-out evaluation: {len(query_norm)} queries vs "
              f"{len(norm_embs)}-image corpus ({n_classes} classes).")
    else:
        query_norm = norm_embs
        query_paths = image_paths
        held_out = False
        print("\n" + "!" * 72)
        print("WARNING: no --query-embeddings given -> IN-SAMPLE (train==test) evaluation.")
        print("Recall and faithfulness are measured on the indexed corpus itself and")
        print("are OPTIMISTIC. Pass a held-out split via --query-embeddings for honest")
        print("generalization numbers.")
        print("!" * 72)

    print("\nNOTE: relevance ground truth is the parent-folder label (single-label "
          "proxy). Visually similar images in other folders count as non-relevant, "
          "so recall/precision are conservative lower bounds on semantic quality.")

    print("\n=== Recall@K / Precision@K / mAP (unsteered baseline) ===")
    ground_truth = build_same_class_ground_truth(
        query_paths, None if not held_out else image_paths
    )
    recall_depth = max(RECALL_KS)
    # in-sample: fetch one extra to drop the query's own hit
    fetch = recall_depth + (1 if not held_out else 0)
    results = []
    for i in range(len(query_norm)):
        _, retrieved = search(index, query_norm[i], k=fetch)
        if not held_out:
            retrieved = retrieved[retrieved != i]
        retrieved_filtered = retrieved[:recall_depth].tolist()
        results.append({"retrieved": retrieved_filtered, "relevant": ground_truth[i]})
    recall = mean_recall_at_k(results, k_values=RECALL_KS)
    precision = mean_precision_at_k(results, k_values=RECALL_KS)
    map_score = mean_average_precision(results, k=recall_depth)
    for k in RECALL_KS:
        print(f"  recall@{k}: {recall[f'recall@{k}']:.4f}   precision@{k}: {precision[f'precision@{k}']:.4f}")
    print(f"  mAP@{recall_depth}: {map_score:.4f}")

    report["meta"] = {
        "held_out": held_out, "n_classes": n_classes, "n_corpus": len(norm_embs),
        "n_queries": len(query_norm), "hidden_dim": hidden_dim,
        "feature_selection": args.feature_selection, "seed": args.seed,
        "alpha": args.faithfulness_alpha, "retrieval_k": retrieval_k,
        "n_top_images": n_top_images, "recall_ks": RECALL_KS,
    }
    report["retrieval"] = {"recall": recall, "precision": precision, f"map@{recall_depth}": map_score}

    print("\nComputing SAE activations...")
    all_acts = []
    with torch.no_grad():
        for start in range(0, len(embeddings), 1024):
            batch = torch.from_numpy(embeddings[start : start + 1024])
            all_acts.append(sae.encode(batch).numpy())
    activations = np.concatenate(all_acts, axis=0)

    ranked = select_eval_features(
        activations, args.n_eval_features, args.feature_selection, seed=args.seed
    )
    print(f"\nEvaluating {len(ranked)} features "
          f"({args.feature_selection} selection, seed={args.seed}) out of "
          f"{hidden_dim} ({len(ranked) / hidden_dim:.1%} of the dictionary).")

    print("\n=== Monosemanticity ===")
    mono_scores = batch_monosemanticity(
        activations, image_paths, ranked, k=n_top_images, n_total_classes=n_classes
    )
    for m in sorted(mono_scores, key=lambda x: -x["purity_score"])[:10]:
        print(f"  [{m['feature_id']:5d}]  purity={m['purity_score']:.3f}  "
              f"dominant={m['dominant_class'][:30]:30s}  ({m['dominant_class_fraction']:.0%})")
    purity_summary = summarize([m["purity_score"] for m in mono_scores], seed=args.seed)
    print(format_summary("Purity", purity_summary))
    null_purities = shuffled_label_purity_baseline(
        activations, image_paths, ranked, k=n_top_images,
        n_total_classes=n_classes, n_shuffles=5, seed=args.seed,
    )
    null_mean = float(np.mean(null_purities)) if null_purities else float("nan")
    print(f"  Null baseline (shuffled labels): mean purity={null_mean:.4f}  "
          f"-> real-minus-null gap={purity_summary['mean'] - null_mean:+.4f}  "
          f"(gap near 0 = features are not class-selective)")
    report["monosemanticity"] = {
        "per_feature": mono_scores, "summary": purity_summary,
        "null_mean": null_mean, "null_purities": list(null_purities),
    }

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
        report["retrieval_comparison"] = comparison

    if not args.skip_targeted_recall:
        print("\n=== Targeted Class Delta-Recall@K ===")
        targeted_results = batch_targeted_recall(
            sae, index, activations, norm_embs, image_paths,
            mono_scores, alpha=args.faithfulness_alpha,
            k=retrieval_k, n_queries=args.faithfulness_queries,
        )
        for r in sorted(targeted_results, key=lambda x: -x["delta_recall"]):
            sign = "+" if r["delta_recall"] >= 0 else ""
            print(f"  [{r['feature_id']:5d}]  class={r['dominant_class'][:30]:30s}  "
                  f"base={r['baseline_recall']:.3f}  "
                  f"steered={r['steered_recall']:.3f}  "
                  f"delta={sign}{r['delta_recall']:.3f}")
        delta_summary = summarize(
            [r["delta_recall"] for r in targeted_results], gt_threshold=0.0, seed=args.seed
        )
        print(format_summary(f"delta-Recall@{retrieval_k}", delta_summary, gt_threshold=0.0))
        print("  (>0 = slider helps find the right class; frac>0 = features that help)")
        report["targeted_recall"] = {"per_feature": targeted_results, "summary": delta_summary}

    need_queries = (
        not args.skip_faithfulness
        or not args.skip_isotonicity
        or not args.skip_ablation
        or args.class_directions is not None
    )
    query_sample: np.ndarray | None = None
    if need_queries:
        rng = np.random.default_rng(args.seed)
        sample_idx = rng.choice(
            len(query_norm),
            size=min(args.faithfulness_queries, len(query_norm)),
            replace=False,
        )
        query_sample = query_norm[sample_idx]

    if not args.skip_faithfulness:
        print("\n=== Steering Faithfulness ===")
        faith_scores = batch_steering_faithfulness(
            sae, index, activations, query_sample,
            ranked, alpha=args.faithfulness_alpha,
            k=retrieval_k, n_queries=args.faithfulness_queries,
        )
        sorted_faith = sorted(faith_scores.items(), key=lambda x: -x[1])
        for fid, score in sorted_faith[:10]:
            print(f"  [{fid:5d}]  faithfulness={score:.3f}")
        faith_summary = summarize(
            list(faith_scores.values()), gt_threshold=1.0, seed=args.seed
        )
        print(format_summary("Faithfulness", faith_summary, gt_threshold=1.0))
        print("  (>1.0 = steering pulls retrieval toward the concept vs unsteered "
              "baseline; ~1.0 = no effect)")
        report["faithfulness"] = {
            "per_feature": {int(k): float(v) for k, v in faith_scores.items()},
            "summary": faith_summary,
        }

        print("\n=== Steering Selectivity ===")
        sel_scores = batch_steering_selectivity(
            sae, index, activations, query_sample,
            ranked, alpha=args.faithfulness_alpha,
            k=retrieval_k, n_queries=args.faithfulness_queries,
        )
        sel_vals = [v for v in sel_scores.values() if not np.isnan(v)]
        sel_summary = summarize(sel_vals, gt_threshold=0.5, seed=args.seed)
        print(format_summary("On-target fraction", sel_summary, gt_threshold=0.5))
        print("  (1.0 = the slider moves only its own feature; low = it drags unrelated "
              "features along)")
        report["steering_selectivity"] = {
            "per_feature": {int(k): (None if np.isnan(v) else float(v))
                            for k, v in sel_scores.items()},
            "summary": sel_summary,
        }

    if not args.skip_isotonicity:
        print("\n=== Steering Isotonicity ===")
        iso_results = batch_steering_isotonicity(
            sae, index, activations, query_sample,
            ranked, k=retrieval_k, n_queries=min(50, args.faithfulness_queries),
        )
        for r in sorted(iso_results, key=lambda x: x["spearman_rho"]):
            bar = "OK" if r["spearman_rho"] >= 0.7 else "!!"
            print(f"  [{r['feature_id']:5d}]  isotonicity_rho={r['spearman_rho']:.3f}  [{bar}]")
        iso_summary = summarize(
            [r["spearman_rho"] for r in iso_results], gt_threshold=0.7, seed=args.seed
        )
        print(format_summary("Isotonicity rho", iso_summary, gt_threshold=0.7))
        print("  (1.0 = perfectly monotone; frac>0.7 = reliably controllable sliders)")
        report["isotonicity"] = {"per_feature": iso_results, "summary": iso_summary}

    if not args.skip_ablation:
        print("\n=== SAE vs PCA Ablation ===")
        ablation = evaluate_sae_vs_pca(
            sae, index, norm_embs, activations, query_sample,
            ranked, n_pca_components=args.n_pca_components,
            alpha=args.faithfulness_alpha, k=retrieval_k,
            n_queries=args.faithfulness_queries,
        )
        print("  Same metric for both: additive cosine lift toward the steering "
              "direction (steered - unsteered). >0 = steering works.")
        print(f"  SAE steering lift : mean={ablation['sae_mean_faithfulness']:+.4f}  "
              f"median={ablation['sae_median_faithfulness']:+.4f}")
        print(f"  PCA steering lift : mean={ablation['pca_mean_faithfulness']:+.4f}  "
              f"median={ablation['pca_median_faithfulness']:+.4f}")
        print(f"  SAE advantage (median lift diff): {ablation['steering_advantage']:+.4f}  "
              f"(mean: {ablation['steering_advantage_mean']:+.4f})")
        print("  NOTE: cosine lift rewards high-variance directions, so PCA scores higher "
              "here by moving the query farther. It does NOT mean PCA steering is better - "
              "see the Retrieval Method Comparison: PCA steering collapses precision while "
              "SAE preserves it. Read that table, not this lift, for the SAE-vs-PCA verdict.")
        report["ablation"] = ablation

    if args.class_directions is not None:
        print("\n=== Direction Steering Faithfulness ===")
        directions = np.load(args.class_directions).astype(np.float32)
        dir_scores: list[float] = []
        for i in range(len(directions)):
            score = direction_steering_faithfulness(
                index, norm_embs, directions, i, query_sample,
                alpha=args.faithfulness_alpha,
                k=retrieval_k,
                n_queries=args.faithfulness_queries,
            )
            dir_scores.append(score)
            print(f"  [direction {i:3d}]  steering lift={score:+.3f}")
        dir_summary = summarize(dir_scores, gt_threshold=0.0, seed=args.seed)
        print(format_summary("Direction steering lift", dir_summary, gt_threshold=0.0))
        print("  (>0 = directions pull retrieval toward the class; cosine lift)")
        report["direction_steering"] = {"per_direction": dir_scores, "summary": dir_summary}

    if args.feature_names is not None:
        print("\n=== CLIP Alignment ===")
        feature_names_map: dict = json.loads(args.feature_names.read_text())
        clip_enc = CLIPEncoder()

        named_features = []
        for fid in ranked:
            name = feature_names_map.get(str(fid), f"Feature {fid}")
            fi = get_top_images(activations, image_paths, fid, k=n_top_images)
            named_features.append({"feature_id": fid, "name": name, "top_paths": fi.top_paths})

        alignment_scores = batch_clip_alignment(named_features, clip_enc)

        for fid, score in sorted(alignment_scores.items(), key=lambda x: -x[1]):
            name = feature_names_map.get(str(fid), f"Feature {fid}")
            print(f"  [{fid:5d}] {name:40s}  {score:.4f}")
        align_summary = summarize(list(alignment_scores.values()), seed=args.seed)
        print(format_summary("Alignment", align_summary))
        report["clip_alignment"] = {
            "per_feature": {int(k): float(v) for k, v in alignment_scores.items()},
            "summary": align_summary,
            "names": {f["feature_id"]: f["name"] for f in named_features},
        }

        print(f"\n=== Cross-Model Alignment ({args.cross_model_validator}) ===")
        cross_results = batch_cross_validation(
            named_features, image_paths, clip_enc,
            validator_model=args.cross_model_validator, k=n_top_images,
        )
        for r in sorted(cross_results, key=lambda x: -x["cross_model_score"])[:10]:
            print(f"  [{r['feature_id']:5d}] {r['name']:40s}  "
                  f"cross={r['cross_model_score']:.4f}  "
                  f"reverse={r['reverse_retrieval_score']:.4f}")
        mean_cross = float(np.mean([r["cross_model_score"] for r in cross_results]))
        mean_reverse = float(np.mean([r["reverse_retrieval_score"] for r in cross_results]))
        print(f"  Mean cross-model: {mean_cross:.4f}  "
              f"Mean reverse-retrieval: {mean_reverse:.4f}")
        report["cross_model"] = {
            "per_feature": cross_results,
            "mean_cross": mean_cross, "mean_reverse": mean_reverse,
        }

    if args.output is not None:
        def _json_default(o):
            if isinstance(o, np.generic):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(f"not JSON-serializable: {type(o)}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=_json_default))
        print(f"\nWrote metrics JSON to {args.output}")


if __name__ == "__main__":
    main()
