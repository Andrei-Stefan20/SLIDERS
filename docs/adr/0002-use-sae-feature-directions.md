# Use SAE feature directions as visual sliders

## Context

The UI needs sliders that map to visual properties. The obvious alternatives do not work well here. Raw DINOv2 dimensions have no individual meaning; each of the 1024 values in a CLS embedding is the result of the network combining information from across the entire image in ways that are not tied to any single visual concept. PCA finds directions of maximum variance but tends to pick up global scene properties like brightness and scale rather than narrow visual attributes (more on this below). Class-based directions, computed as the mean embedding of one class minus another, only cover the visual axes that happen to separate labelled categories, and most target datasets in this project have no labels.

The requirement is a set of directions in the 1024-d DINO space such that adding a scaled direction to a query embedding predictably shifts retrieval toward images that share a specific visual property.

## Why SAE over PCA

PCA finds the directions of maximum variance in a set of vectors. On DINOv2 embeddings, the first few components almost always correspond to global scene properties like brightness, background color, or scale, because those are the properties that vary most across a large corpus. A narrow property like "yellow chlorotic halo on a leaf" accounts for very little total variance, ends up in late components, and is typically mixed with unrelated properties.

PCA also has a hard limit at `input_dim` components. At 1024 dimensions there are at most 1024 directions, and every image has a non-zero projection on all of them. Activating a PCA component shifts the query along a direction that blends many visual properties at once rather than isolating one.

A SAE with 8192 hidden units has far more directions available than the input has dimensions, which is intentional. When there are only as many units as input dimensions, every unit has to respond to many different images just to reconstruct the data well, and there is not enough room for each unit to stay focused on one thing. With 8192 units on a 1024-d input there is enough slack that a unit can afford to stay silent on most inputs and only fire when its specific property is present. The L1 sparsity penalty actively reinforces this. A unit that weakly activates on many unrelated inputs gets penalized, so it learns to either commit to a clear role or stay at zero. In practice most units are inactive on any given input, and the ones that do fire tend to each respond to a single narrow visual property. That is what makes them usable as named sliders.

## Decision

Train a single-layer Sparse Autoencoder on DINOv2 CLS embeddings. Use decoder columns as steering directions. A slider applies a weighted decoder column to the query embedding, then the result is L2-normalized before FAISS search.

```
input:   x ∈ R^1024

encode:  h = ReLU(W_enc · x + b_enc)     W_enc ∈ R^{8192×1024}
decode:  x̂ = W_dec · h + b_dec           W_dec ∈ R^{1024×8192}

steering:  q' = normalize(q + Σ αᵢ · W_dec[:, fid_i])
```

Default `hidden_dim = 8192`, an 8× expansion over the 1024-d input. The expansion gives each unit enough room to focus on one thing rather than having to cover many unrelated properties at once.

## Loss function

The training loss has two terms.

```
L = MSE(x, x̂) + λ · L1(h)
  = mean((x - x̂)²) + λ · mean(|h|)
```

The reconstruction term (`MSE`) pushes `x̂` toward `x`. The sparsity term (`L1`) penalizes the sum of activations. The goal is to keep most units at exactly zero for any given input, but counting active units directly is not something gradient descent can optimize. L1 is a good substitute. Unlike L2, which shrinks all activations proportionally but never brings any of them to exactly zero, L1 applies a flat cost per unit regardless of its value. Small activations get pushed all the way to zero; large activations pay the same per-unit cost but stay active because their contribution to reconstruction is worth it. The result is that most of the 8192 units are zero on any given input, and a few fire strongly.

With `λ=1e-3` the reconstruction loss dominates, so L1 adds gentle pressure rather than collapsing features. Increasing λ gives sparser codes at the cost of reconstruction quality; decreasing it recovers reconstruction quality but produces units that respond to many unrelated properties.

**TopK mode** (`topk > 0`) removes the L1 term and enforces a hard activity budget instead. After the linear encoder step, only the K largest values are kept and passed through ReLU; everything else is set to zero.

```
h_i = ReLU(h_pre_i)  if h_pre_i in top-K(h_pre)
    = 0               otherwise

L = recon(x, x̂)   (no sparsity term; recon can be MSE or cosine)
```

This guarantees exactly K active units per sample (or fewer if some of the top-K values are negative and get zeroed by ReLU). There is no λ to tune, and K has a direct meaning as the average number of active features per image. The choice of reconstruction loss (MSE or cosine) is independent of whether TopK is used. This is the formulation from Gao et al. (2024).

**Cosine loss** (`loss_type="cosine"`) replaces MSE with an angular term.

```
L = mean(1 - cos(x, x̂)) + 0.1 · MSE(x, x̂)
```

Since FAISS uses inner product on normalized vectors, the steering directions only need to point in the right direction; their magnitude does not matter. The cosine term optimizes for that directly. The 0.1·MSE term is there to stop the model from gaming the loss by outputting very large `x̂` vectors. A large-magnitude vector always has cosine similarity close to 1 with anything, so without the MSE guard the model could minimize cosine loss without actually learning to reconstruct the input. In practice on normalized DINO embeddings the difference from pure MSE is small.

**Why val reconstruction loss for checkpoint selection.** When using ReLU+L1, the training loss includes the sparsity term, whose magnitude changes as the codes become sparser over training. Two models with identical reconstruction quality but different sparsity levels would have different training losses. Validation loss measures only reconstruction, giving a clean comparison across configurations and training stages.

## Training loop details

Optimizer is Adam at `lr=3e-4` with CosineAnnealingLR, batch size 512, 10% validation split, early stopping at patience=10.

The SAE supports an optional tied-weights mode where the decoder weight matrix is forced to equal `W_enc^T` rather than being learned as a separate matrix. When weights are untied (the default), decoder columns are re-normalized to unit length after each optimizer step:

```python
F.normalize(sae.decoder.weight.data, dim=0, out=sae.decoder.weight.data)
```

This keeps all feature directions the same length, so two sliders set to the same α value produce the same size shift regardless of how the individual features were learned.

**Dead neuron reset.** A unit is considered dead if it has not fired on any sample in the last `dead_threshold_steps=1000` gradient steps. Dead units are recycled by replacing the encoder row with a direction taken from the current batch's reconstruction error, specifically `normalize(x − x̂)`, and zeroing its bias. The reconstruction error represents what the SAE is currently failing to capture, so pointing a dead unit at it gives the unit a useful starting point. A persistently high dead count is a signal to re-run with a lower λ or a smaller `hidden_dim`.

## How steering works and what slider values mean

Each slider corresponds to a decoder column, `W_dec[:, fid]`, a unit vector in R^1024. The decoder learns these columns so that images with a high activation on unit `fid` are reconstructed by adding a large amount of that column to the output. The column therefore points in the direction in embedding space that represents "more of this feature." The feature id `fid` is just an integer index into the 8192 hidden units.

When the user sets a slider to value α, that column is added to the query scaled by α, and the result is renormalized.

```
q' = normalize(q + α₁·d₁ + α₂·d₂ + ...)
```

All embeddings are unit vectors, so they sit on a sphere in R^1024. Adding `α·d` and renormalizing rotates the query toward the direction `d`. After this rotation, the nearest neighbors found by FAISS tend to be images whose embeddings sit close to `d`, which are the images that activate that feature strongly. Positive α pushes the query toward that region of the sphere; negative α pushes it away, suppressing that property in results.

The slider value has no absolute unit. Small values nudge results gently; large values can rotate the query so far from its original direction that results strongly exhibit the feature but look nothing like the original query image. Because decoder columns are kept at unit length during training (for untied weights, the default), two features set to the same α produce the same size rotation regardless of how strongly each was learned.

## SAE use at query time

The SAE artifact (`models/sae_best.pt`) participates in three places during retrieval.

**Steering** is the operation described above. Decoder columns are added to the query embedding with slider alphas and then normalized.

**Activation rerank** addresses a gap that comes up in practice. For every image in the corpus the SAE encoder pre-computes a sparse activation vector `h[i]` and stores it. At query time, the candidates returned by FAISS are re-scored by how much they activate the features currently set in the slider config (`Σ αⱼ · h[i, fidⱼ]`). The reason this matters is that the steered DINO query finds images that are globally similar in embedding space, but some of those may not actually have the requested visual property. Re-ranking by the stored activations surfaces the ones that do.

**SAE-space index** (`sae_index.faiss`) is an optional second index built from the same pre-computed activation vectors, normalized to unit length. Searching it finds images whose overall activation patterns are close to the steered query's activation pattern, independently of their similarity in DINO space. Results from both indices are merged before reranking.

## Consequences

Features are dataset-specific. An SAE checkpoint trained on PlantVillage does not transfer to ceramics; the decoder columns encode visual properties present in the training corpus. On small or visually narrow datasets the SAE may learn weak or duplicate features; the naming pipeline will return `"undifferentiated"` for those, which is the signal to either add more images or reduce `hidden_dim`.

## Owner files

| File | Role |
| --- | --- |
| `src/models/sae.py` | SAE module |
| `src/models/train_sae.py` | training loop, dead reset, checkpoint selection |
| `src/models/losses.py` | MSE, cosine, L1 sparsity |
| `src/retrieval/steering.py` | `steer_query` |
| `src/retrieval/query.py` | decoder directions, activation rerank, SAE-space merge |
