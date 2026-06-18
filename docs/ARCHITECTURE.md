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
   (--use-patches)  ->  <dataset>_patch_embeddings.npy (memmap; float16 default, or int8),
                        <dataset>_patch_image_ids.npy, <dataset>_patch_meta.json,
                        <dataset>_patch_image_paths.json, (<dataset>_patch_scales.npy if int8)
train_sae           ->  <dataset>_sae_best.pt (+ .meta.json)
name_features       ->  <dataset>_feature_names.json
build_index         ->  <dataset>_index.faiss, <dataset>_activations.npy, (<dataset>_sae_index.faiss)
   (patch corpus)   ->  <dataset>_patch_index.faiss (late-interaction patch index)
compute_class_dir.  ->  <dataset>_class_directions.npy, <dataset>_class_direction_names.json
```

All artifacts share the `<dataset>` prefix (the `dataset.name` in the config), derived automatically from the embeddings filename so multiple datasets coexist on disk.

## DINOv2

The CLS retrieval path uses `dinov2_vitl14` loaded via `torch.hub` and saves the CLS token, one float32 vector of shape `(1024,)` per image. With `--use-patches` the encoder switches to the **registers** variant `dinov2_vitl14_reg` (registers absorb the high-norm artifact patches that would otherwise become spurious SAE features) and saves the patch tokens (16×16 = 256 per image at 224px, each `(1024,)`), flattened into a memory-mapped `(N_images*256, 1024)` array written one batch at a time so the full set never lives in RAM. A sidecar `_patch_image_ids.npy` maps each patch row back to its image (patches of one image are contiguous), and `_patch_meta.json` records the grid size.

For a patch-trained SAE, `name_features` ranks images by each feature's max activation over their patches and shows the VLM a montage of an image's top-N patches (`--n-patches`). Each patch is shown as a **context crop with the active patch outlined** rather than a tight crop: an SAE activation correlates with a patch but attention mixes in context, so the cause may be nearby (arXiv:2509.00749) — keeping context and marking the patch lets the VLM use both.

Input preprocessing is fixed: resize to 256, center-crop to 224, ImageNet normalization. MPS (Apple Silicon) is forced to CPU because DINOv2 produces incorrect results on that backend.

CLIP is not used as a retrieval backbone. It appears only in `src/evaluation` for cross-model name checks and in `src/naming/vlm_namer.py` for alignment scoring.

## SAE

The SAE maps `1024 -> 8192 -> 1024`. The 8× expansion gives the model enough room to learn a sparse, overcomplete representation. The goal is not compression but disentanglement. The input is either the CLS token (whole-image concepts) or DINOv2 patch tokens when trained on `<dataset>_patch_embeddings.npy` (`train_sae --mmap`, auto-enabled above 2 GB); patch training yields local, region-level features. The module is identical either way since both are 1024-dim.

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

## Patch-level retrieval (MaxSim)

For a patch-trained SAE the corpus is indexed at the patch level (`build_index` on a
`_patch_embeddings.npy` file → `<dataset>_patch_index.faiss`, IVF-PQ above ~200k patches).
Retrieval is **late interaction** (ColBERT-style), in `src/retrieval/patch_retrieval.py`:

1. a query image is its set of 256 patch vectors
2. the patch index fetches candidate patches per query patch → candidate images
3. each candidate is scored exactly by **MaxSim** (sum over query patches of the best
   matching corpus patch), reconstructing its patches from the memmap

Steering adds the SAE feature direction to the query patches (`steer_patches`) before the
MaxSim search — the patch-space analog of the CLS slider. `evaluate` runs this path for
patch corpora (recall/precision/mAP + monosemanticity + faithfulness/selectivity/isotonicity);
see `docs/EVALUATION.md`. The live app uses it too: launch with the patch stem
(`--dataset <name>_train_patch`); `load_resources` detects the patch sidecars and builds a
patch `AppState` (`patch_corpus` set, no CLS embeddings), and `RetrievalService.retrieve`
takes the MaxSim branch. The query image is sent as its flattened patch tokens, so the
API's 1-D embedding schema and the frontend are unchanged.

## Slider sources

At API startup `src/ui/resources.py` decides which sliders to show:

1. If `<dataset>_class_directions.npy` exists → class direction sliders, ignoring SAE features.
2. Else if `<dataset>_feature_names.json` exists → named SAE feature sliders.
3. Else if `<dataset>_activations.npy` exists → SAE features ranked by activation variance, labeled `Feature <id>`.

## API and frontend

`src/api.py` exposes FastAPI routes and serves the static frontend. `src/ui/retrieval_service.py` encodes query images, applies class filtering, and calls the retrieval stack. `app/app.js` manages browser state: upload flow, slider values, and result rendering. `app/index.html` and `app/style.css` provide the DOM shell and layout.
