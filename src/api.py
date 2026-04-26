import argparse
import base64
import io
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage
from pydantic import BaseModel

from src.config import AppConfig
from src.ui.resources import (
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_IMAGE_PATHS_JSON,
    DEFAULT_INDEX_PATH,
    DEFAULT_SAE_PATH,
    load_resources,
)
from src.ui.retrieval_service import RetrievalService
from src.ui.state import AppState
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

_state: AppState | None = None
_service: RetrievalService | None = None


def _require_state() -> AppState:
    if _state is None:
        raise HTTPException(status_code=503, detail="API resources are not loaded")
    return _state


def _require_service() -> RetrievalService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Retrieval service is not loaded")
    return _service


def _to_b64(img: PILImage.Image, size: int = 0, quality: int = 82) -> str:
    img = img.convert("RGB")
    if size:
        img.thumbnail((size, size), PILImage.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _path_to_b64(path: str, size: int = 120) -> str | None:
    try:
        return _to_b64(PILImage.open(path), size=size, quality=75)
    except Exception:
        return None


def _compute_scores(emb: np.ndarray) -> list[float]:
    state = _require_state()
    if state.class_directions is not None:
        return (state.class_directions @ emb).tolist()
    with torch.no_grad():
        t = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
        acts = state.sae.encode(t).squeeze(0).numpy()
    return [float(acts[fid]) for fid in state.feature_ids]


def _feature_values(ui_feature_id: int) -> np.ndarray:
    state = _require_state()
    if ui_feature_id < 0 or ui_feature_id >= len(state.feature_names):
        raise HTTPException(status_code=404, detail="Feature not found")

    if state.class_directions is not None:
        emb = np.asarray(state.embeddings, dtype=np.float32)
        emb_norm = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
        return emb_norm @ state.class_directions[ui_feature_id]

    if state.activations is None:
        raise HTTPException(status_code=400, detail="Feature activations are not available")

    feature_id = state.feature_ids[ui_feature_id]
    return np.asarray(state.activations[:, feature_id], dtype=np.float32)


def _serialize_examples(values: np.ndarray, descending: bool, limit: int = 8) -> list[dict]:
    state = _require_state()
    order = np.argsort(values)
    if descending:
        order = order[::-1]
    selected = order[:limit].tolist()

    examples = []
    for idx in selected:
        img_b64 = _path_to_b64(state.image_paths[idx], size=220)
        if img_b64 is None:
            continue
        examples.append({
            "image": img_b64,
            "value": float(values[idx]),
            "path": state.image_paths[idx],
        })
    return examples


def _build_feature_payload(feature_id: int) -> dict:
    state = _require_state()
    values = _feature_values(feature_id)
    return {
        "id": feature_id,
        "name": state.feature_names[feature_id],
        "description": (state.feature_descriptions[feature_id] if state.feature_descriptions else ""),
        "preview_absent": _path_to_b64(state.preview_bottom[feature_id]) if state.preview_bottom else None,
        "preview_present": _path_to_b64(state.preview_top[feature_id]) if state.preview_top else None,
        "high_examples": _serialize_examples(values, descending=True, limit=8),
        "low_examples": _serialize_examples(values, descending=False, limit=8),
    }


app = FastAPI(title="SLIDERS API")


@app.get("/api/features")
async def get_features():
    state = _require_state()
    features = [_build_feature_payload(i) for i in range(len(state.feature_names))]
    return {"features": features}


@app.get("/api/features/{feature_id}")
async def get_feature_detail(feature_id: int):
    return _build_feature_payload(feature_id)


@app.post("/api/encode")
async def encode(file: UploadFile = File(...)):
    service = _require_service()
    data = await file.read()
    pil = PILImage.open(io.BytesIO(data)).convert("RGB")
    img_np = np.array(pil)
    raw_emb = service.encode_image_raw(img_np)
    emb = service.encode_image(img_np)
    scores = _compute_scores(raw_emb)
    return {"embedding": emb.tolist(), "scores": scores}


class RetrieveRequest(BaseModel):
    embedding: list[float]
    sliders:   list[float]
    k: int = 15


@app.post("/api/retrieve")
async def retrieve(req: RetrieveRequest):
    state = _require_state()
    service = _require_service()
    emb  = np.array(req.embedding, dtype=np.float32)
    imgs = service.retrieve(emb, req.sliders, k=req.k)
    items = []
    for img in imgs:
        path = getattr(img, "filename", "") or ""
        class_name = ""
        if path:
            try:
                idx = state.image_paths.index(path)
                class_name = state.image_classes[idx]
            except ValueError:
                class_name = ""
        items.append({
            "image": _to_b64(img, size=240, quality=82),
            "download": _to_b64(img, size=0, quality=92),
            "path": path,
            "class_name": class_name,
        })
    return {"images": items, "majority_class": service.last_majority_class}


# Serve static frontend, registered last so API routes take precedence
_APP_DIR = Path(__file__).parent.parent / "app"
if _APP_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_APP_DIR), html=True), name="static")


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config) if args.config is not None else None
    dataset = args.dataset or (cfg.dataset.name if cfg else None)
    n_sliders = cfg.retrieval.n_sliders if cfg else 20

    _state = load_resources(
        index_path=DEFAULT_INDEX_PATH,
        sae_path=DEFAULT_SAE_PATH,
        embeddings_path=DEFAULT_EMBEDDINGS_PATH,
        image_paths_json=DEFAULT_IMAGE_PATHS_JSON,
        dataset=dataset,
        adapter_name=(cfg.dataset.adapter if cfg else dataset or "generic"),
        n_sliders=n_sliders,
    )
    _service = RetrievalService(_state)
    logger.info(f"API ready at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
