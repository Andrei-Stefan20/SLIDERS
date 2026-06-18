# Tensor shapes and data flow

This document traces every tensor through the pipeline in order, from raw images to retrieved results. Shapes use the convention `(dim0, dim1, ...)` where `N` is the corpus size and `B` is a batch size.

---

## Phase 1: embedding extraction

**Script:** `scripts/extract_embeddings.py`

```
raw image on disk  (H, W, 3)  PIL, any resolution

  Resize(256)         shorter side → 256 px, aspect ratio kept
  CenterCrop(224)     (224, 224, 3)
  ToTensor()          (3, 224, 224)   float32, values in [0, 1]
  Normalize()         (3, 224, 224)   float32, ImageNet mean/std subtracted

batched by DataLoader → (B, 3, 224, 224)
```

DINOv2 ViT-L/14 processes each batch. Patch size is 14 px, so a 224×224 image produces a 16×16 = 256 patch grid.

```
(B, 3, 224, 224)
  → DINOv2 forward pass
  → CLS token extracted                  (B, 1024)   float32, NOT yet normalized
```

All batches are concatenated and saved.

```
saved to disk:
  {dataset}_embeddings.npy               (N, 1024)   float32, raw CLS tokens
  {dataset}_image_paths.json             list of N path strings, same order
```

The embeddings at this point are **not** L2-normalized. Normalization happens in the next phase.

### Patch variant (`--use-patches`)

The encoder loads `dinov2_vitl14_reg` (registers variant) and returns patch tokens:

```
(B, 3, 224, 224)
  → model.forward_features(images)["x_norm_patchtokens"]   (B, 256, 1024)
  → flatten to rows, written into a preallocated memmap one batch at a time

saved to disk:
  {dataset}_patch_embeddings.npy          (N_images*256, 1024)  float32, memmap
  {dataset}_patch_image_ids.npy           (N_images*256,)       int32, patch→image
  {dataset}_patch_meta.json               {grid_size, patches_per_image, n_images}
  {dataset}_patch_image_paths.json        list of N_images path strings
```

Patches of one image are contiguous rows. The SAE then trains on this 2-D memmap exactly as below (`EmbeddingDataset(mmap=True)`); the module is unchanged since rows are still 1024-d.

---

## Phase 2: SAE training

**Script:** `scripts/train_sae.py`

`EmbeddingDataset` loads the `.npy` file and serves one row per item. By default each row is **L2-normalized** before being served (`normalize=True`), so the SAE trains on unit vectors. Pass `normalize=False` to train on raw CLS tokens. When `mmap=True` the array stays memory-mapped and rows are normalized lazily in `__getitem__`.

```
batch from EmbeddingDataset             (B, 1024)   float32, L2-normalized

encoder linear + bias:
  W_enc shape                           (8192, 1024)
  b_enc shape                           (8192,)
  W_enc · x + b_enc  →                 (B, 8192)   pre-activations

ReLU (or TopK):
  h  →                                  (B, 8192)   sparse, most values zero

decoder linear + bias:
  W_dec shape                           (1024, 8192)
  b_dec shape                           (1024,)
  W_dec · h + b_dec  →  x̂              (B, 1024)   reconstruction

loss terms:
  MSE(x, x̂)          →  scalar
  L1(h) = mean(|h|)  →  scalar
  L = MSE + λ · L1   →  scalar
```

After each optimizer step (untied weights only), each column of `W_dec` is re-normalized to unit length. `W_dec` remains `(1024, 8192)` in shape; only the values change.

Dead neuron tracking uses a counter `steps_since_active` of shape `(8192,)`. When a unit exceeds the effective dead threshold without firing, its encoder row `W_enc[i]` is replaced with `normalize(x − x̂)` from the current batch, `b_enc[i]` is set to zero, and the Adam moments for the revived rows are reset. The effective threshold is `min(dead_threshold_steps, 2 × steps_per_epoch)`; when the configured `dead_threshold_steps` (default 1000) is capped, a warning is logged.

The best checkpoint is saved as `models/{dataset}_sae_best.pt` (selected on the lowest validation **score**: reconstruction loss for TopK, or `recon + λ · sparsity` for ReLU). A sidecar `*.meta.json` stores `topk`/`tied_weights` so `SparseAutoencoder.load` can rebuild the model.

---

## Phase 3: index and activation pre-computation

**Script:** `scripts/build_index.py`

```
load {dataset}_embeddings.npy           (N, 1024)   float32, raw

L2-normalize each row:
  norms                                 (N, 1)
  embeddings / norms  →                 (N, 1024)   float32, unit vectors

faiss.IndexFlatIP(1024)
  index.add(embeddings)
  stored vectors inside index           (N, 1024)

saved:
  data/processed/{dataset}_index.faiss
```

If a SAE checkpoint is provided, corpus activations and a second index are also computed from the **normalized** embeddings.

```
normalized embeddings                   (N, 1024)

batched through sae.encode():
  h = ReLU(W_enc · x + b_enc)  →       (B, 8192)   per batch

concatenated:
  activations                           (N, 8192)   float32, sparse

saved:
  data/processed/{dataset}_activations.npy   (N, 8192)

per-vector normalize activations:
  norms                                 (N, 1)
  acts / norms  →                       (N, 8192)   unit vectors in activation space

faiss.IndexFlatIP(8192)
  sae_index.add(acts_normed)

saved:
  data/processed/{dataset}_sae_index.faiss
```

Note: both training and activation pre-computation now operate on L2-normalized embeddings, so the SAE sees a consistent input distribution offline and at query time.

---

## Phase 4: query encoding

**API:** `POST /api/encode` → `RetrievalService.encode_image`

```
user uploads image                      (H, W, 3)   numpy array

convert to PIL, apply DINO transform:
  (3, 224, 224)   float32

DINOEncoder.encode (use_patches=False):
  unsqueeze  →                          (1, 3, 224, 224)
  DINOv2 forward  →                     (1, 1024)   raw CLS token
  squeeze  →                            (1024,)

L2-normalize:
  norm = ‖emb‖                          scalar
  emb / norm  →                         (1024,)     unit vector   ← this is q
```

The normalized `(1024,)` vector is what gets sent to the retrieval step.

---

## Phase 5: slider steering and retrieval

**`search_with_sliders` in `src/retrieval/query.py`**

Assume `n_active` sliders are set (non-zero alpha values). Feature ids are indices into the 8192-d hidden space.

```
slider_config = {fid_1: α_1, fid_2: α_2, ...}   n_active entries

extract encoder directions:
  W_enc                                 (8192, 1024)
  W_enc[[fid_1, fid_2, ...]]  →         (n_active, 1024)
  L2-normalize each row  →  directions  (n_active, 1024)   each row is a unit vector

alphas array                            (n_active,)

steer_query:
  (alphas[:, None] * directions).sum(axis=0)  →  delta   (1024,)
  q + delta  →  steered_raw                               (1024,)
  steered_raw / ‖steered_raw‖  →  q'                      (1024,)   unit vector
```

FAISS search on the primary (DINO-space) index.

```
q' reshaped  →                          (1, 1024)
index.search(q', fetch_k)  →
  distances                             (fetch_k,)   cosine similarities in [−1, 1]
  indices                               (fetch_k,)   int64, positions in [0, N)
```

`fetch_k = max(k × 3, 60)` when reranking is active, otherwise `k`.

---

## Phase 6: optional SAE-space index merge

```
encode the UNsteered query through SAE:
  sae.encode(q.reshape(1, -1))  →       (1, 8192)
  squeeze  →                            (8192,)

bump the active features directly:
  acts[fid] = max(0, acts[fid] + alpha)   for each active slider

normalize  →                            (8192,)     unit vector in activation space

sae_index.search(acts_normed, fetch_k)  →
  sae_distances                         (fetch_k,)
  sae_indices                           (fetch_k,)

merge with DINO results:
  if corpus_activations is available:
    union the two candidate pools (deduped, DINO first); the activation
    rerank in Phase 7 (same guard) recomputes the real ordering, so dists
    here are only a descending placeholder.
  else (RRF fallback):
    score[i] = (1 − 0.3) / (60 + rank_dino[i]) + 0.3 / (60 + rank_sae[i])
    sorted desc  →  top fetch_k

  idxs                                  (fetch_k,)
  dists                                 (fetch_k,)
```

---

## Phase 7: optional activation rerank

```
corpus_activations[idxs]                (fetch_k, 8192)
[:, active_feature_ids]  →              (fetch_k, n_active)   only active columns

alphas_arr                              (n_active,)

sae_scores = (activations_slice * alphas_arr).sum(axis=1)
                                        (fetch_k,)   one score per candidate

argsort descending  →  reorder idxs by sae_scores
                        dists are replaced by sae_scores (relevance, not
                        cosine sim); the MMR pass uses them as its relevance term
```

Images with high activation on positively-weighted features (and low activation on negatively-weighted ones) rise to the top.

---

## Phase 8: optional MMR diversity rerank

```
corpus_embeddings[idxs]                 (fetch_k, 1024)   raw embeddings from AppState, not unit-normalized
sim_matrix = embs @ embs.T              (fetch_k, fetch_k) pairwise dot products

iterative greedy selection:
  start with highest-distance candidate
  each step: argmax [ 0.7 · distance[i] − 0.3 · max_sim_to_already_selected[i] ]
  repeat until all fetch_k candidates are ordered

final idxs                              (fetch_k,)   diversity-reranked
```

Return `idxs[:k]` and `dists[:k]`.

---

## Phase 9: feature naming localization

**`src/naming/spatial_localization.py`**

This path uses patch tokens instead of the CLS token. The geometry transform
(`Resize(256) → CenterCrop(224)`) matches corpus extraction, so crop coordinates align
with the pixels DINOv2 saw.

```
raw image  (H, W, 3)   PIL, any resolution
  Resize(256) → CenterCrop(224)         (224, 224, 3)   same as corpus extraction

DINOEncoder (use_patches=True, dinov2_vitl14_reg):
  (1, 3, 224, 224)
  model.forward_features(images)["x_norm_patchtokens"]   (1, 256, 1024)
  squeeze  →                            (256, 1024)

SAE encoder applied patch by patch:
  sae.encode(patch_tokens)              (256, 8192)   sparse activations per patch
  patch_acts[:, feature_id]  →          (256,)        one value per patch

top-N patches (default 4)  →  for each: 14px patch cell + 96px context box (grid=16)
  crop the 96px context window, outline the patch cell (red), montage the N crops
  → one PIL montage per image
```

For ranking and top-image selection (patch-trained SAE), `name_features` first reduces the
patch memmap to a per-image `(N_images, 8192)` matrix by taking the **max over each image's
patches**. The montage (HIGH and LOW examples) is then passed to the VLM. Showing the patch
in context, not a tight crop, is deliberate: activation location is correlational, not
causal (arXiv:2509.00749).

---

## AppState at runtime

All offline artifacts are held in `AppState` after `load_resources` runs.

| Field | Shape | Source |
| --- | --- | --- |
| `embeddings` | `(N, 1024)` | raw `.npy`, memory-mapped |
| `activations` | `(N, 8192)` or `None` | `activations.npy`, memory-mapped |
| `index` | FAISS `IndexFlatIP(1024)` | `index.faiss` |
| `sae_index` | FAISS `IndexFlatIP(8192)` or `None` | `sae_index.faiss` |
| `feature_ids` | list of `n_sliders` ints | from `feature_names.json` or variance rank |
| `class_directions` | `(n_classes, 1024)` or `None` | `class_directions.npy` |