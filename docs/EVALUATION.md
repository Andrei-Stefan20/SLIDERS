# Evaluation

## Command

```bash
python scripts/evaluate.py \
  --embeddings data/processed/plantvillage_train_embeddings.npy \
  --image-paths data/processed/plantvillage_train_image_paths.json \
  --index data/processed/plantvillage_train_index.faiss \
  --sae-model models/plantvillage_train_sae_best.pt \
  --query-embeddings data/processed/plantvillage_val_embeddings.npy \
  --query-image-paths data/processed/plantvillage_val_image_paths.json \
  --feature-names models/plantvillage_train_feature_names.json \
  --class-directions data/processed/plantvillage_train_class_directions.npy
```

`--embeddings`, `--image-paths`, and `--index` describe the **corpus** (the indexed
set the UI searches — use the training set). `--sae-model` is required too.
`--feature-names` and `--class-directions` are optional and enable the CLIP-naming
and class-direction-steering metrics respectively.

### Held-out queries (strongly recommended)

`--query-embeddings` / `--query-image-paths` supply a **held-out** query split
(e.g. a validation set). When given, all query-based metrics (recall and steering
faithfulness) use those queries against the training corpus, so the numbers
reflect generalization. **Without them the script falls back to in-sample
(train==test) self-retrieval and prints a loud warning** — those numbers are
optimistic and should not be reported as generalization.

## Methodology notes

- **Ground truth is the parent-folder label** (single-label proxy). Visually
  similar images in other folders count as non-relevant, so recall/precision are
  conservative lower bounds on semantic quality.
- **Feature sampling.** Per-feature metrics run on a sample of the dictionary
  chosen by `--feature-selection` (`random`, default — representative; or
  `variance`, the legacy top-variance head, which is optimistic). `--n-eval-features`
  sets the count (default 40). Results are reported as a distribution (mean,
  median, 95% bootstrap CI, p10/p90) via `src/evaluation/stats.py`, not a bare mean.
- **Two `k` knobs.** `--n-top-images` is how many top-activating images are
  examined per feature (monosemanticity, CLIP alignment); `--retrieval-k` is the
  retrieval depth for faithfulness / isotonicity / targeted recall. Both default to
  the legacy `--top-k`. Recall@K cutoffs are always [1, 5, 10].
- `--seed` controls feature/query sampling and the bootstrap CIs.

## Metrics

**recall@k / precision@k / mAP** (`src/evaluation/recall_at_k.py`) measure
same-folder retrieval for each query against the corpus. In-sample, the query's
own image is excluded; held-out, queries are not in the corpus.

**monosemanticity** (`src/evaluation/monosemanticity.py`) checks class purity of
the images that activate each feature most strongly. Entropy is normalised by
`log(n_total_classes)` (the whole dataset), so purity is comparable across
features. A **shuffled-label null baseline** is reported alongside: real purity is
only meaningful as the gap above this null.

**steering faithfulness** (`src/evaluation/steering_faithfulness.py`) steers
held-out queries along each feature's decoder direction (pure embedding-space
steering — no activation rerank, which would be circular) and reports the ratio of
the feature's activation in steered vs **unsteered** retrieval. >1.0 = steering
pulls retrieval toward the concept; ~1.0 = no effect. `--faithfulness-alpha`
(default 2.0), `--faithfulness-queries` (default 100). Skip with `--skip-faithfulness`.

**steering isotonicity** (`src/evaluation/steering_isotonicity.py`) checks that
increasing the slider value monotonically increases the feature's activation in
results (Spearman ρ over several alphas). frac>0.7 = reliably controllable sliders.
Skip with `--skip-isotonicity`.

**targeted class delta-recall** (`src/evaluation/targeted_recall.py`) measures
whether steering a feature improves retrieval of its dominant class (delta of
same-class recall, steered minus unsteered). Skip with `--skip-targeted-recall`.

**SAE vs PCA** (`src/evaluation/ablation.py`) compares SAE steering against a PCA
baseline using the **same metric for both**: the additive cosine *lift* of
retrieved items toward the steering direction (steered minus unsteered). A ratio
is avoided because cosine alignment is signed and baselines can be ~0. The headline
is `steering_advantage` = SAE median lift − PCA median lift (>0 = SAE directions
steer better than raw principal components). `--n-pca-components` (default 20). Skip
with `--skip-ablation`.

**retrieval method comparison** (`src/evaluation/retrieval_comparison.py`) prints a
P@K / R@K / mAP table for Unsteered vs PCA vs SAE steering. Skip with
`--skip-comparison`.

**CLIP alignment** (`src/evaluation/clip_alignment.py`) embeds each feature name
with CLIP and measures cosine similarity to the top-activating images. Note this is
partly circular (names are generated from those images); see below.

**cross-model alignment** and **reverse retrieval** (`src/evaluation/cross_model_alignment.py`)
break the CLIP feedback loop: cross-model repeats the name check with a different
CLIP model (`--cross-model-validator`, default `ViT-B-32`); reverse retrieval uses
the name as a text query and measures Jaccard overlap with the feature's top images
(the positives are always included in the search pool so the overlap is not biased
by sampling).

## When to run

Run after changing SAE training settings, feature ranking, VLM naming, retrieval
steering, dataset adapters, or index build behavior. Prefer a held-out query split
for any number you intend to report.
