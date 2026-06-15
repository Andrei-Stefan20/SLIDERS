# Architecture

## Runtime flow

```
query image
  -> DINOv2 encoder
  -> normalized query embedding
  -> optional slider steering
  -> FAISS search
  -> optional reranking
  -> image results
```

## Packages

`src/data` loads raw images and precomputed embeddings. `src/encoders` wraps DINOv2 (main backbone) and CLIP (evaluation and naming checks only). `src/models` contains the SAE module, loss functions, and training loop. `src/naming` handles feature ranking, patch-level crop localization, and VLM calls. `src/retrieval` builds and queries FAISS indexes, applies slider steering, and reranks. `src/ui` loads all runtime artifacts at startup and wires the retrieval service to the API. `src/evaluation` holds the metric suite. `src/utils` covers IO, logging, device selection, and exceptions.

## Artifacts

Each script produces files that the next step consumes:

```
extract_embeddings  ->  <dataset>_embeddings.npy, <dataset>_image_paths.json
train_sae           ->  <dataset>_sae_best.pt (+ .meta.json)
name_features       ->  <dataset>_feature_names.json
build_index         ->  <dataset>_index.faiss, <dataset>_activations.npy, (<dataset>_sae_index.faiss)
compute_class_dir.  ->  <dataset>_class_directions.npy, <dataset>_class_direction_names.json
```

All artifacts share the `<dataset>` prefix (the `dataset.name` in the config), derived automatically from the embeddings filename so multiple datasets coexist on disk.

## DINOv2

The extractor uses `dinov2_vitl14` loaded via `torch.hub`. In extraction mode (`use_patches: false`) it saves the CLS token, one float32 vector of shape `(1024,)` per image. In naming mode the same model returns patch tokens so the SAE can score each patch individually and find the spatial region that most activates a feature.

Input preprocessing is fixed: resize to 256, center-crop to 224, ImageNet normalization. MPS (Apple Silicon) is forced to CPU because DINOv2 produces incorrect results on that backend.

CLIP is not used as a retrieval backbone. It appears only in `src/evaluation` for cross-model name checks and in `src/naming/vlm_namer.py` for alignment scoring.

## SAE

The SAE maps `1024 -> 8192 -> 1024`. The 8× expansion gives the model enough room to learn a sparse, overcomplete representation. The goal is not compression but disentanglement.

```
h = ReLU(W_enc x + b_enc)          # (B, 8192)
x_hat = W_dec h + b_dec            # (B, 1024)
```

When `topk > 0`, only the largest K activations survive; the rest are zeroed before decoding. With `tied_weights` off (default), decoder columns are L2-normalized after every optimizer step so that all steering directions have comparable magnitude at retrieval time.

Encoder rows that stay inactive for `dead_threshold_steps` steps are reset by reinitializing from normalized residual vectors. This prevents the model from wasting hidden units.

## Steering at retrieval time

Sliders move the query embedding along SAE decoder columns before the FAISS search. If the user sets feature 17 to +1.5 and feature 203 to −0.8, `src/retrieval/query.py`:

1. pulls decoder columns for ids 17 and 203
2. adds the weighted directions to the query embedding
3. normalizes the steered vector
4. searches `index.faiss`

If `<dataset>_activations.npy` exists the retrieved images are then reranked by their precomputed SAE activations on the active features, compensating for the fact that FAISS operates in DINO embedding space while slider meaning lives in SAE feature space.

If `<dataset>_sae_index.faiss` also exists the code additionally encodes the steered query into SAE activations, searches that index, merges the two hit lists, and reranks again.

## Slider sources

At API startup `src/ui/resources.py` decides which sliders to show:

1. If `<dataset>_class_directions.npy` exists → class direction sliders, ignoring SAE features.
2. Else if `<dataset>_feature_names.json` exists → named SAE feature sliders.
3. Else if `<dataset>_activations.npy` exists → SAE features ranked by activation variance, labeled `Feature <id>`.

## API and frontend

`src/api.py` exposes FastAPI routes and serves the static frontend. `src/ui/retrieval_service.py` encodes query images, applies class filtering, and calls the retrieval stack. `app/app.js` manages browser state: upload flow, slider values, and result rendering. `app/index.html` and `app/style.css` provide the DOM shell and layout.
