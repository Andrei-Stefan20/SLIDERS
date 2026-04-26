# Configuration

The default config is `configs/plantvillage.yaml`. Pass it to every script with `--config`.

## Full schema

```yaml
dataset:
  name: plantvillage
  path: data/raw/plantvillage
  batch_size: 64
  adapter: plantvillage

encoder:
  use_patches: false

sae:
  hidden_dim: 8192
  lambda_sparsity: 0.001
  lr: 0.0003
  epochs: 50
  batch_size: 512

retrieval:
  n_sliders: 20

naming:
  n_features: 20
  n_crops: 8
  crop_size: 96
  ranking: diverse_mmr
  lambda_mmr: 0.5
  vlm_model: Qwen/Qwen3-VL-4B-Instruct
```

## dataset

`name` drives output filenames (`plantvillage_train_embeddings.npy`, etc.) and the API artifact lookup at startup. `path` is the raw image root. `adapter` selects the dataset-specific class logic via `get_adapter()`. Use `plantvillage` for PlantVillage, `generic` for anything without folder-level class labels. `batch_size` controls DINOv2 extraction throughput.

## encoder

`use_patches: false` means the extractor saves one CLS embedding per image. The naming script overrides this internally to extract patch tokens for spatial localization, you do not need to change this field.

## sae

`hidden_dim` is the SAE feature count. The default of `8192` is an 8× expansion over the 1024-d DINOv2 CLS embedding, giving the model enough room to learn sparse, disentangled features without becoming too large.

Two training modes:

- **ReLU + L1** (default, `topk: 0`): loss is `mse(x, x_hat) + lambda_sparsity * mean(|h|)`. `lambda_sparsity` controls how hard the model is pushed toward sparse activations.
- **TopK** (`topk > 0`): only the top K activations per sample are kept; reconstruction loss only, no sparsity term. `topk` must be at most `hidden_dim // 10`.

`loss_type: cosine` replaces MSE with `(1 − cosine_similarity(x, x_hat)) + 0.1 * mse(x, x_hat)`, which aligns reconstruction with the cosine geometry used by FAISS at retrieval time.

Training uses Adam with cosine LR annealing. The best checkpoint is selected by lowest validation reconstruction loss. Early stopping triggers after `patience` epochs without improvement. If a feature stays inactive for `dead_threshold_steps` steps, its encoder row is reinitialized from normalized residuals.

`--tied-weights` (CLI only, not in YAML) makes the decoder use `encoder.weight.T`. Untied weights are the default.

## retrieval

`n_sliders` sets how many SAE feature axes are shown in the UI.

## naming

`n_features` is the number of features to name. `n_crops` controls how many high- and low-activation images are cropped per feature. `crop_size` is the pixel size of each crop sent to the VLM.

`ranking` selects how candidate feature ids are chosen:

- `variance`: highest activation variance, fast baseline
- `diverse_mmr`: variance + MMR diversity, default
- `sparsity`: high max activation with sparse candidates
- `selectivity`: class-selective features with MMR, best when folder labels are meaningful

`lambda_mmr` balances relevance and diversity in MMR ranking (0 = pure diversity, 1 = pure relevance).

`vlm_model` is any HuggingFace model id. The default is `Qwen/Qwen3-VL-4B-Instruct`.
