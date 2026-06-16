# Run guide

## Prerequisites

| Requirement | Notes |
| --- | --- |
| Python 3.11 | Available at python.org |
| Kaggle account | Needed to download PlantVillage |
| Config file | Use `configs/plantvillage.yaml` by default |

## One command

After the environment is set up (step 0) and the raw data is in place, the whole
pipeline runs in one go via the orchestrator. It retrains the SAE, names features,
builds the index, computes class directions, evaluates on the held-out split, and
generates the report — stopping at the first failure.

```powershell
.\scripts\run_pipeline.ps1
```

Useful switches: `-ExtractEmbeddings` to also re-extract embeddings first,
`-StartUI` to launch the UI at the end, `-Topk 40` to set sparsity,
`-SkipClassDirections`, and `-DryRun` to print the commands without running them.
The steps below document each stage individually.

## Run order

0. Install environment.

Allow script execution (once per terminal session):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Create the virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

The default install pulls the CPU-only build of PyTorch. For GPU acceleration, reinstall torch after the step above with the right CUDA build for your card.

RTX 30xx / 40xx (Ampere / Ada, CUDA 12.1):

```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

RTX 50xx (Blackwell, CUDA 12.8, requires driver 570+):

```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Check that the GPU is visible before continuing:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Both lines should print `True` and your GPU name. If `is_available()` returns `False`, update your NVIDIA drivers from nvidia.com and verify that `nvidia-smi` shows CUDA Version 12.1 or 12.8 respectively.

1. Download raw images.

Put `kaggle.json` in `C:\Users\<you>\.kaggle\kaggle.json`.

```powershell
kaggle datasets download -d mohitsingh1804/plantvillage -p data\raw\plantvillage
Expand-Archive -LiteralPath data\raw\plantvillage\plantvillage.zip -DestinationPath data\raw\plantvillage -Force
```

After extraction the structure will be:

```text
data\raw\plantvillage\PlantVillage\
  train\
    Apple___Apple_scab\
    Apple___healthy\
    ...
  val\
    Apple___Apple_scab\
    Apple___healthy\
    ...
```

2. Extract embeddings.

Run once for train, once for val.

```powershell
python scripts/extract_embeddings.py --config configs/plantvillage.yaml --dataset plantvillage_train
```

```powershell
python scripts/extract_embeddings.py `
  --config configs/plantvillage.yaml `
  --dataset plantvillage_val `
  --input data/raw/plantvillage/PlantVillage/val
```

Together, the two commands produce:

```text
data/processed/plantvillage_train_embeddings.npy
data/processed/plantvillage_train_image_paths.json
data/processed/plantvillage_val_embeddings.npy
data/processed/plantvillage_val_image_paths.json
```

3. Train the SAE.

Train on the training set only.

```powershell
python scripts/train_sae.py `
  --embeddings data/processed/plantvillage_train_embeddings.npy `
  --output models/ `
  --config configs/plantvillage.yaml
```

**What to watch during training.**

The key metric is `val_L0` — the average number of SAE features active per sample. Features are interpretable when `val_L0` is in the range 20–80. Above ~200 the SAE is not truly sparse and the features will not be monosemantic.

**Sparsity: TopK vs ReLU + L1.**

TopK (`--topk K`) forces exactly K features active per sample. It is the easiest way to control sparsity because the L0 is guaranteed regardless of the other hyperparameters.

```powershell
# 40 features active per sample 
python scripts/train_sae.py `
  --embeddings data/processed/plantvillage_train_embeddings.npy `
  --output models/ --config configs/plantvillage.yaml `
  --topk 40
```

Without `--topk` the model uses ReLU + L1. The default `lambda_sparsity: 0.001` in the config is too weak and will leave `val_L0` in the hundreds. Increase it until `val_L0` lands in range.

```powershell
# ReLU + L1 with  sparsity penalty
python scripts/train_sae.py `
  --embeddings data/processed/plantvillage_train_embeddings.npy `
  --output models/ --config configs/plantvillage.yaml `
  --lambda-sparsity 0.05
```

**Loss type.**

The default MSE loss minimises reconstruction error in Euclidean space. The cosine loss additionally aligns directions, which matches the inner-product geometry used by the FAISS index. Use it when retrieval quality matters more than pixel-accurate reconstruction.

```powershell
--loss-type cosine
```

**Feature dictionary size.**

`--hidden-dim` controls how many SAE features are learned (default 8192, 8× the DINOv2 embedding size). Larger dictionaries can capture more fine-grained concepts but take longer to train and require stronger sparsity pressure.

```powershell
--hidden-dim 4096   # faster, fewer concepts
--hidden-dim 16384  # more concepts, needs --topk or stronger lambda
```

**Early stopping and training length.**

`--patience` stops training after N epochs without val loss improvement (default 10). `--epochs` sets the maximum (default 50). For large datasets or a high `--hidden-dim`, both can be increased.

```powershell
--epochs 100 --patience 20
```

4. Name SAE features.

```powershell
python scripts/name_features.py `
  --config configs/plantvillage.yaml `
  --embeddings data/processed/plantvillage_train_embeddings.npy `
  --image-paths data/processed/plantvillage_train_image_paths.json `
  --sae-model models/plantvillage_train_sae_best.pt
```

The output path is derived from the embeddings name (`models/plantvillage_train_feature_names.json`); pass `--output` only to override it.

5. Build indexes and activations.

Build on the training set. The UI will search only over training images.

```powershell
python scripts/build_index.py `
  --embeddings data/processed/plantvillage_train_embeddings.npy `
  --sae-model models/plantvillage_train_sae_best.pt
```

Index and activation paths are derived from the embeddings name (`plantvillage_train_index.faiss`, `plantvillage_train_sae_index.faiss`, `plantvillage_train_activations.npy`).

6. Optional: compute class direction sliders.

```powershell
python scripts/compute_class_directions.py `
  --embeddings data/processed/plantvillage_train_embeddings.npy `
  --image-paths data/processed/plantvillage_train_image_paths.json `
  --adapter plantvillage `
  --output data/processed/
```

7. Evaluate before reporting any number.

Run the full metric battery. Pass the **held-out val split** as queries so the
numbers reflect generalization rather than in-sample self-retrieval.

```powershell
python scripts/evaluate.py `
  --embeddings data/processed/plantvillage_train_embeddings.npy `
  --image-paths data/processed/plantvillage_train_image_paths.json `
  --index data/processed/plantvillage_train_index.faiss `
  --sae-model models/plantvillage_train_sae_best.pt `
  --query-embeddings data/processed/plantvillage_val_embeddings.npy `
  --query-image-paths data/processed/plantvillage_val_image_paths.json `
  --feature-names models/plantvillage_train_feature_names.json `
  --class-directions data/processed/plantvillage_train_class_directions.npy `
  --output reports/plantvillage_train_eval.json
```

The corpus flags `--embeddings`, `--image-paths` and `--index` are the **training**
set the UI searches; the `--query-*` flags are the held-out queries. Omitting the
query flags falls back to in-sample retrieval and prints a loud warning — those
numbers are optimistic. `--feature-names` and `--class-directions` are optional and
enable the CLIP-naming and class-direction-steering metrics. `--output` dumps every
metric as JSON for the report in the next step. See
[docs/EVALUATION.md](EVALUATION.md) for every metric, the k knobs, feature
sampling, and the null baselines.

8. Generate report figures.

Produces a `reports/plantvillage_train/` folder with one subfolder per stage,
each holding PNG figures plus the underlying CSV/JSON: embedding projection and
class distribution, training curves and sparsity, top-activating montages per
feature, the class-direction similarity heatmap, and the evaluation charts. Stages
whose inputs are missing are skipped with a note, so it also runs mid-pipeline.

```powershell
python scripts/make_report.py `
  --dataset plantvillage_train `
  --eval-json reports/plantvillage_train_eval.json
```

Training curves need the `<dataset>_sae_history.json` written by `train_sae.py`, so
they appear only after a fresh train; the evaluation charts need the `--output` JSON
from step 7.

9. Start the UI.

```powershell
python -m src.api --config configs/plantvillage.yaml --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## Output files

All artifacts are prefixed with the dataset name (`dataset.name` in the config, here `plantvillage_train`), derived automatically from the embeddings filename. This lets multiple datasets coexist in `data/processed/` and `models/` without overwriting each other.

| File | Created by | Used by |
| --- | --- | --- |
| `data/processed/plantvillage_train_embeddings.npy` | `extract_embeddings.py` | SAE training, index build, UI |
| `data/processed/plantvillage_train_image_paths.json` | `extract_embeddings.py` | feature naming, class directions, UI |
| `data/processed/plantvillage_val_embeddings.npy` | `extract_embeddings.py` | evaluation |
| `data/processed/plantvillage_val_image_paths.json` | `extract_embeddings.py` | evaluation |
| `models/plantvillage_train_sae_best.pt` (+ `.meta.json`) | `train_sae.py` | feature naming, index build, UI |
| `models/plantvillage_train_sae_history.json` | `train_sae.py` | report training curves |
| `models/plantvillage_train_feature_names.json` | `name_features.py` | UI slider labels |
| `data/processed/plantvillage_train_index.faiss` | `build_index.py` | FAISS search |
| `data/processed/plantvillage_train_sae_index.faiss` | `build_index.py --sae-model` | optional SAE-space result merge |
| `data/processed/plantvillage_train_activations.npy` | `build_index.py --sae-model` | slider reranking, previews, automatic labels |
| `data/processed/plantvillage_train_class_directions.npy` | `compute_class_directions.py` | optional class sliders |
| `data/processed/plantvillage_train_class_direction_names.json` | `compute_class_directions.py` | optional class slider labels |
| `reports/plantvillage_train_eval.json` | `evaluate.py --output` | report evaluation charts |
| `reports/plantvillage_train/` | `make_report.py` | figures + data tables for the report |
