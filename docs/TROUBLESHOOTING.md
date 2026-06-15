# Troubleshooting

## UI startup

| Symptom | Cause | Fix | File to inspect |
| --- | --- | --- | --- |
| API returns `503` | resources did not load | check artifact paths and startup logs | `src/api.py`, `src/ui/resources.py` |
| missing `<dataset>_sae_best.pt` | SAE training did not run or output path differs | run `scripts/train_sae.py` or copy checkpoint to `models/` | `scripts/train_sae.py` |
| missing `<dataset>_index.faiss` | index build did not run | run `scripts/build_index.py` | `scripts/build_index.py` |
| config validation error | unknown field or invalid value | compare YAML with `src/config.py` | `src/config.py`, `configs/*.yaml` |

## Sliders

| Symptom | Cause | Fix | File to inspect |
| --- | --- | --- | --- |
| no sliders | no feature names, class directions, or activations | run naming or rebuild index with `--sae-model` | `src/ui/resources.py` |
| class sliders appear instead of SAE features | class direction files exist | remove class direction files if SAE sliders are wanted | `data/processed/` |
| sliders do not change results | `<dataset>_activations.npy` or `<dataset>_sae_index.faiss` missing | rebuild index with `--sae-model` | `scripts/build_index.py` |
| generic `Feature <id>` labels | `<dataset>_feature_names.json` missing | run `scripts/name_features.py` | `models/<dataset>_feature_names.json` |

## Retrieval results

| Symptom | Cause | Fix | File to inspect |
| --- | --- | --- | --- |
| wrong dataset appears | artifacts from different datasets are mixed | rebuild all artifacts for one dataset | `data/processed/`, `models/` |
| results stay in one class | majority-class filter is active | inspect adapter labels and nearest neighbors | `src/ui/retrieval_service.py` |
| few or no class directions | classes have fewer than five samples | add images or change adapter grouping | `scripts/compute_class_directions.py` |
| FAISS search fails | index dimension differs from embeddings | rebuild index from the same embeddings used by UI | `src/retrieval/index.py` |

## Feature naming

| Symptom | Cause | Fix | File to inspect |
| --- | --- | --- | --- |
| VLM labels are weak | crops do not isolate the feature | increase `n_crops` or inspect crops manually | `src/naming/spatial_localization.py` |
| many repeated names | selected features are redundant | use `diverse_mmr` or tune `lambda_mmr` | `src/naming/feature_ranking.py` |
| naming is slow | local VLM and DINO patch encoding are expensive | reduce `n_features` or `n_crops` | `scripts/name_features.py` |

## Typing and checks

| Symptom | Cause | Fix | File to inspect |
| --- | --- | --- | --- |
| Pylance warning on dynamic model loader | dependency returns loose types | use local `Protocol` and `cast` | `src/encoders/dino_encoder.py`, `src/encoders/clip_encoder.py` |
| Pylance warning on preprocessing output | torchvision stubs are broad | cast result to `torch.Tensor` | `src/naming/spatial_localization.py` |
| FAISS `.add()` typing warning | FAISS stubs do not expose all runtime methods | cast test index to `Any` | `tests/test_retrieval_service.py` |

## Pytest temp permission errors on Windows

If pytest fails with `WinError 5` under `AppData/Local/Temp/pytest-of-*`, use targeted tests that do not need `tmp_path`:

```bash
python -m pytest -q tests/test_config.py tests/test_datasets.py tests/test_query.py tests/test_sae.py tests/test_losses.py tests/test_steering.py -p no:cacheprovider
```
