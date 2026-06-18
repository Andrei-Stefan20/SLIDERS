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

**What "success" means here.** SLIDERS is a tool for *per-image visual attribute
manipulation*, not a classifier. The headline steering metrics are therefore
**faithfulness** and **isotonicity** (does pushing a slider monotonically increase its
own attribute in the results?), corroborated by **steering selectivity** (does it move
*only* that attribute?) and the qualitative montages + cross-model name checks. Metrics
framed around class retrieval (recall@k, targeted delta-recall) are diagnostic, not the
objective — see their entries below.

**recall@k / precision@k / mAP** (`src/evaluation/recall_at_k.py`) measure
same-folder retrieval for each query against the corpus. In-sample, the query's
own image is excluded; held-out, queries are not in the corpus. On datasets with large
per-class pools, recall@k is mechanically tiny (k ≪ |relevant|) — **read precision and
mAP**, not recall@k.

**monosemanticity** (`src/evaluation/monosemanticity.py`) checks class purity of
the images that activate each feature most strongly. Entropy is normalised by
`log(n_total_classes)` (the whole dataset), so purity is comparable across
features. A **shuffled-label null baseline** is reported alongside: real purity is
only meaningful as the gap above this null.

**steering faithfulness** *(headline)* (`src/evaluation/steering_faithfulness.py`) steers
held-out queries along each feature's decoder direction (pure embedding-space
steering — no activation rerank, which would be circular) and reports the ratio of
the feature's activation in steered vs **unsteered** retrieval. >1.0 = steering
pulls retrieval toward the concept; ~1.0 = no effect. This is the operational definition
of "the slider does what it says". `--faithfulness-alpha` (default 2.0),
`--faithfulness-queries` (default 100). Skip with `--skip-faithfulness`.

**steering isotonicity** *(headline)* (`src/evaluation/steering_isotonicity.py`) checks
that increasing the slider value monotonically increases the feature's activation in
results (Spearman ρ over several alphas). frac>0.7 = reliably controllable sliders.
Skip with `--skip-isotonicity`.

**steering selectivity** *(headline)* (`src/evaluation/steering_selectivity.py`) asks
whether a slider moves *only* its own attribute. For each feature it reports the
**on-target fraction** = the feature's mean activation increase in the steered top-k
divided by the total positive increase across all features. ~1.0 = clean, disentangled
slider; low = pushing it drags unrelated features along. Faithfulness/isotonicity are
partly self-referential (they measure the steered feature itself); selectivity and the
cross-model name checks are the independent corroboration.

**targeted class delta-recall** *(diagnostic, not a success metric)*
(`src/evaluation/targeted_recall.py`) measures the change in same-class recall when
steering a feature. Steering an *attribute* is not meant to change which class dominates,
so a value **near 0 is the expected, healthy result**: it means steering moved the
attribute while staying on the data manifold. Read it as an on-manifold sanity check
(contrast PCA, which collapses precision); a large negative value flags a feature whose
steering throws retrieval off-manifold. Skip with `--skip-targeted-recall`.

**SAE vs PCA** (`src/evaluation/ablation.py`) reports the additive cosine *lift* toward
the steering direction for SAE features vs PCA components. **Caveat: cosine lift rewards
high-variance directions, so PCA scores higher simply by moving the query farther — it is
not a verdict that PCA steers better.** For the real comparison use the *retrieval method
comparison* below: PCA steering collapses precision while SAE steering preserves it.
`--n-pca-components` (default 20). Skip with `--skip-ablation`.

**retrieval method comparison** *(the SAE-vs-PCA verdict)*
(`src/evaluation/retrieval_comparison.py`) prints a P@K / R@K / mAP table for Unsteered
vs PCA vs SAE steering. This is the honest SAE-vs-PCA comparison: it shows whether
steering keeps results relevant. Typically PCA steering tanks precision (it shoves the
query off-manifold) while SAE steering preserves it. Skip with `--skip-comparison`.

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

## Region-level features (patch-token SAE)

A CLS-trained SAE captures global concepts (leaf edge, veins, shadow): on PlantVillage the
sliders are **directionally distinct** (decoder-cosine heatmap near-zero off-diagonal) yet
several get the **same VLM name** — more distinct directions exist than coarse nameable
concepts on a narrow dataset. The structural fix is to train on **DINOv2 patch tokens**
instead of the CLS token, so features become local and region-specific.

Implemented:
- `extract_embeddings --use-patches` writes the patch tokens (from the `dinov2_vitl14_reg`
  registers variant, which avoids the high-norm artifact patches) as a memory-mapped
  `(N_images*256, 1024)` array plus the `_patch_image_ids` / `_patch_meta` sidecars.
- `train_sae --mmap` trains the SAE directly on that memmap (auto-enabled above 2 GB).
- `name_features` detects patch embeddings, aggregates per image by max over patches, and
  shows the VLM a montage of each image's top-N patches (`--n-patches`), each as a context
  crop with the active patch outlined (activation location is correlational, not causal —
  arXiv:2509.00749 — so context is kept rather than cropped away).

Patch retrieval and evaluation (late-interaction MaxSim):
- `build_index` on a patch embeddings file builds a **patch index** (IVF-PQ above ~200k
  patches, ~1-2 GB; exact flat below) for candidate generation.
- `src/retrieval/patch_retrieval.py` scores a query image against the corpus by **MaxSim**
  (sum over query patches of the best-matching corpus patch), computed exactly from the
  patch memmap for the candidate images. Steering adds the SAE feature direction to the
  query patches first (`steer_patches`) — the patch-space analog of slider steering.
- `evaluate` detects a patch corpus and runs the patch path (`src/evaluation/patch_eval.py`):
  **recall@k / precision@k / mAP** (parent-folder ground truth), **monosemanticity** (class
  purity of each feature's top *patches*, labelled by their parent image), **steering
  faithfulness** (steering a feature raises its activation in the MaxSim results, >1),
  **selectivity** (on-target fraction), and **isotonicity** (Spearman ρ over alphas).
- The patch retrieval is also wired into the live app: launch the API with the patch stem
  (`--dataset <name>_train_patch`) and `RetrievalService` uses MaxSim; sliders steer the
  query patches.

Still CLS-only: the **SAE-vs-PCA ablation** and the **CLIP / cross-model naming** checks
(the latter are CLIP-circular by construction). Everything else has a patch equivalent.
