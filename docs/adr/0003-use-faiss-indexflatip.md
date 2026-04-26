# Use FAISS IndexFlatIP over normalized vectors

## Context

Slider steering moves the query embedding before search. The steered query is `q' = normalize(q + Σ αᵢ·dᵢ)`. Approximate index structures (IVF, HNSW) are trained on the distribution of corpus embeddings and assume queries come from roughly the same distribution. After aggressive steering, the query can land far from any training centroid, causing IVF to mis-assign it and HNSW to traverse the wrong neighborhood. The approximation error compounds with steering magnitude.

At the dataset sizes this project targets, exact search is fast enough that there's no reason to accept approximation.

## Decision

Use `faiss.IndexFlatIP` over L2-normalized vectors. Inner product over normalized vectors is cosine similarity. Two indices are built.

- `data/processed/index.faiss`, DINOv2 CLS embeddings, normalized, dim=1024
- `data/processed/sae_index.faiss` (optional), SAE activation vectors, normalized per-vector, dim=8192

```python
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)   # embeddings already L2-normalized upstream
```

For the SAE-space index, activations can have very different norms across images (many zeros), so normalization happens at index-build time:

```python
norms = np.linalg.norm(acts, axis=1, keepdims=True)
acts_normed = acts / np.where(norms > 1e-8, norms, 1.0)
```

## Why not IndexFlatL2

On normalized vectors, L2 and cosine distance give identical rankings (`‖a−b‖² = 2 − 2·cos(a,b)` for unit vectors). The difference is that `IndexFlatIP` returns similarity scores (higher = more similar), which is easier to work with when merging results from two indices. `IndexFlatL2` would return distances and need sign-flipping before merging.

## Query pipeline

When sliders are active, `search_with_sliders` in `src/retrieval/query.py` runs up to five sequential steps.

```
1. Steer:   q' = normalize(q + Σ αᵢ · W_dec[:, fid_i])

2. Fetch:   FAISS search, fetch_k = max(k×3, 60) candidates
            (3× over-fetch to give reranking room to work)

3. Merge    (if sae_index provided):
            SAE-space search on normalized encode(q')
            score[i] = (1 - 0.3) · dino_sim[i]/max_dino
                     +       0.3  · sae_sim[i]/max_sae

4. Rerank   (if corpus_activations provided):
            score[i] = Σ_j α_j · h[i, fid_j]
            promotes images that actually fire on the active features,
            fixing the mismatch between DINO-space similarity and SAE-space relevance

5. MMR      (if corpus_embeddings provided):
            argmax [ 0.7·relevance − 0.3·max_sim_to_selected ]
            adds result diversity (Carbonell & Goldstein, 1998)
```

Steps 3–5 are all optional and additive. A minimal deployment uses only steps 1–2.

The 3× over-fetch in step 2 exists because the steered DINO query finds globally similar images, some of which may have low activation on the requested features. Without over-fetching, reranking has too few candidates to surface the genuinely relevant ones.

## Consequences

`IndexFlatIP` scans every vector at search time. Adding new images requires only `index.add(new_embeddings)`, no retraining needed.

## Owner files

| File | Role |
| --- | --- |
| `src/retrieval/index.py` | `build_index`, `build_sae_index`, save/load |
| `scripts/build_index.py` | offline artifact creation |
| `src/retrieval/query.py` | full query pipeline with optional reranking stages |
| `src/retrieval/steering.py` | `steer_query` |
