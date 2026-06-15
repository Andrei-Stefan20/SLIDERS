# Evaluation

## Command

```bash
python scripts/evaluate.py \
  --embeddings data/processed/plantvillage_train_embeddings.npy \
  --image-paths data/processed/plantvillage_train_image_paths.json \
  --index data/processed/plantvillage_train_index.faiss \
  --sae-model models/plantvillage_train_sae_best.pt \
  --feature-names models/plantvillage_train_feature_names.json \
  --class-directions data/processed/plantvillage_train_class_directions.npy
```

`evaluate.py` does self-retrieval over the indexed corpus, so `--embeddings`, `--image-paths`, and `--index` must all refer to the **same set** — use the training set, which is also what the UI searches. `--feature-names` and `--class-directions` are optional and enable the CLIP-naming and class-direction-steering metrics respectively.

`--embeddings`, `--image-paths`, `--index`, and `--sae-model` are required. `--feature-names` is optional, pass it to enable the CLIP-based naming metrics.

## Metrics

**recall@k** (`src/evaluation/recall_at_k.py`) measures same-folder retrieval: for each query image, how many of the top-k results share its folder label. Requires embeddings, image paths, and index. k values are hardcoded to [1, 5, 10].

**monosemanticity** (`src/evaluation/monosemanticity.py`) checks class purity of the images that activate each feature most strongly. A monosemantic feature fires predominantly on one class. Requires embeddings, image paths, SAE checkpoint, and precomputed activations. `--top-k` controls how many top images are checked per feature (default 10).

**steering faithfulness** (`src/evaluation/steering_faithfulness.py`) steers a sampled set of query embeddings along each feature direction and measures the activation lift in retrieved results. Requires embeddings, index, SAE checkpoint, and precomputed activations. `--faithfulness-alpha` sets the steering weight (default 2.0), `--faithfulness-queries` the sample size (default 100). Skip with `--skip-faithfulness` if you want a faster run.

**SAE vs PCA** (`src/evaluation/ablation.py`) compares SAE steering quality against a PCA baseline of the same dimensionality. Requires embeddings, index, SAE checkpoint, and precomputed activations. `--n-pca-components` sets the baseline size (default 20). Skip with `--skip-ablation`.

**CLIP alignment** (`src/evaluation/clip_alignment.py`) embeds each feature name with CLIP and measures cosine similarity to the top-activating images for that feature. Requires feature names, image paths, SAE checkpoint, and precomputed activations. `--n-align-features` controls how many features are evaluated (default 20).

**cross-model alignment** and **reverse retrieval** (`src/evaluation/cross_model_alignment.py`) repeat the name check with a second CLIP model to catch alignment that is specific to one model's vocabulary. Requires feature names, image paths, SAE checkpoint, and precomputed activations. `--cross-model-validator` sets the second model (default `ViT-B-32`). Reverse retrieval checks whether the text name alone retrieves the feature's top images.

## When to run

Run after changing SAE training settings, feature ranking, VLM naming, retrieval steering, dataset adapters, or index build behavior.
