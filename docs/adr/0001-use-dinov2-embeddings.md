# Use DINOv2 ViT-L/14 embeddings

## Context

Retrieval, SAE training, and class directions all share the same embedding space. If any of those components uses a different backbone, direction arithmetic breaks. A decoder column from an SAE trained on DINO embeddings is meaningless in a CLIP space. The embedding model is therefore a fixed global dependency for the whole pipeline, chosen once per deployment.

## Decision

Use DINOv2 ViT-L/14 loaded via `torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")`. Store one CLS token per image as a 1024-d float32 vector in `data/processed/<dataset>_embeddings.npy`. Patch tokens are used in two cases: the feature-naming step (spatial crop localization), and the optional patch-level SAE (`--use-patches`), where the SAE trains on the patch tokens themselves to learn local, region-level features instead of global CLS concepts. Patch extraction loads the registers variant `dinov2_vitl14_reg`, which absorbs the high-norm artifact patches that would otherwise become spurious SAE features; the CLS retrieval path keeps the plain `dinov2_vitl14` so existing CLS artifacts are unaffected.

The preprocessing pipeline is `Resize(256) → CenterCrop(224) → ToTensor() → ImageNet normalize`.

Corpus extraction path
```
raw image → 224×224 tensor → DINOv2 CLS token → (B, 1024) → .npy
```

Localization path (naming)
```
224×224 tensor → patch tokens (B, 256, 1024)  [16×16 grid, 14px patches]
              → SAE encoder per patch → (256, 8192)
              → top-N patches on feature_id → 96px context crops, active patch outlined
              → montage per image
```

`DINOEncoder` in `src/encoders/dino_encoder.py` wraps both modes (and both model variants via `model_name`). When `use_patches=False` it calls `model(images)` and returns the CLS token; when `True` it calls `model.forward_features(images)["x_norm_patchtokens"]`.

On Apple MPS the constructor forces `device = cpu`. `forward_features` produces NaN activations on MPS in float16, a known issue with the attention kernel on that backend. Extraction is slower on CPU but runs once offline.

## Why DINOv2 ViT-L/14

DINOv2 is trained with DINO self-distillation and masked image modelling (iBOT) on a curated 142M-image corpus, entirely without class labels. This matters here because the pipeline must generalize to arbitrary datasets (PlantVillage, ceramics collections, custom image folders) and we can't assume labels exist.

The ViT-L/14 variant specifically. The 14-pixel patch size divides 224 evenly, giving a 16×16 = 256-patch grid that's fine-grained enough to localize small visual details (leaf lesions, glaze defects). The 1024-d CLS output gives the SAE sufficient input dimensionality for an 8× expansion to 8192 hidden units without underdetermining the encoder; ViT-S/14 (384-d) and ViT-B/14 (768-d) are more compressed and leave less room for independent sparse directions.

CLIP exists in `src/encoders/clip_encoder.py` and is used for evaluation and cross-model alignment checks, not retrieval. Its contrastive training objective introduces a language prior that pulls image embeddings toward text-compatible representations, which is useful for text-queried retrieval but adds noise to purely visual similarity.

The CLS token, not mean-of-patches, is used for corpus embeddings. CLS attends to all patches and aggregates global semantics; mean pooling weights every spatial position equally, which dilutes the signal when the discriminative content is localized.

## Consequences

The main consequence of this choice is that the entire pipeline is coupled to a single 1024-d space. Switching backbone means re-extracting embeddings, retraining the SAE, and rebuilding both FAISS indices. That's acceptable because extraction and SAE training are offline steps, but it should be explicit.

## Owner files

| File | Role |
| --- | --- |
| `src/encoders/dino_encoder.py` | DINOv2 wrapper, CLS and patch modes |
| `scripts/extract_embeddings.py` | offline corpus extraction |
| `src/naming/spatial_localization.py` | patch-token crop localization |
