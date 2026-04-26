# Docs

<video src="media/esempio.mp4" controls muted playsinline width="960"></video>

[media/esempio.mp4](media/esempio.mp4)

## Pages

| Page | Type | Role |
| --- | --- | --- |
| [GUIDE.md](GUIDE.md) | How-to | Run the project from artifacts to UI |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Troubleshooting | Map failures to fixes and files |

## Reference

| Page | Role |
| --- | --- |
| [CONFIGURATION.md](CONFIGURATION.md) | YAML sections, defaults, validation |
| [RETRIEVAL_UI.md](RETRIEVAL_UI.md) | API routes, browser flow, slider sources |
| [EVALUATION.md](EVALUATION.md) | Metrics, evaluator files, CLI arguments |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Checks, tests, typing notes |

## Explanation

| Page | Role |
| --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System flow, modules, artifacts |
| [FEATURE_NAMING.md](FEATURE_NAMING.md) | SAE feature ranking, crop localization, VLM naming |
| [adr/README.md](adr/README.md) | Architecture decisions |

## Runtime map

```text
raw images
  -> DINOv2 embeddings
  -> SAE checkpoint
  -> feature names and activations
  -> FAISS indexes
  -> FastAPI backend
  -> browser UI
```

## Files

| File | Purpose | Used by |
| --- | --- | --- |
| `src/api.py` | FastAPI routes and static app server | browser UI |
| `src/ui/resources.py` | Loads models, embeddings, indexes, names | API startup |
| `src/ui/retrieval_service.py` | Encodes query images and retrieves results | `/api/encode`, `/api/retrieve` |
| `src/retrieval/query.py` | Search, slider steering, reranking | UI service, evaluation |
| `src/models/sae.py` | Sparse autoencoder module | training, naming, retrieval |
| `src/naming/vlm_namer.py` | Local VLM wrapper | `scripts/name_features.py` |
| `src/config.py` | Strict config models | scripts, API |
