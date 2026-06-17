"""Cross-model and reverse-retrieval validation for feature names.

Breaks the CLIP feedback loop by:
  1. Validating names with a different CLIP model than was used to generate them.
  2. Using the name as a text query and checking how much it retrieves the feature's images.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from src.encoders.clip_encoder import CLIPEncoder


def cross_model_alignment(
    feature_name: str,
    top_image_paths: list[Path | str],
    validator_model: str = "ViT-B-32",
    validator_pretrained: str = "openai",
    validator: "CLIPEncoder | None" = None,
) -> float:
    """Validate a feature name using a different CLIP model.

    If the name scores high on both the generator model (ViT-L-14) AND this
    validator, it is genuinely descriptive rather than an artefact of one model's
    vocabulary.

    Args:
        feature_name: Name to validate.
        top_image_paths: Top-activating images for this feature.
        validator_model: open-clip model identifier (default ViT-B-32).
        validator_pretrained: Pretrained weights tag.
        validator: Pre-built encoder to reuse across calls; built from
            validator_model/pretrained if None (loading CLIP is slow, so batch
            callers should pass one).

    Returns:
        Mean cosine similarity in [-1, 1]. Higher is better.
    """
    if validator is None:
        from src.encoders.clip_encoder import CLIPEncoder
        validator = CLIPEncoder(model_name=validator_model, pretrained=validator_pretrained)
    text_emb = validator.encode_text([feature_name]).squeeze(0)

    similarities: list[float] = []
    for path in top_image_paths:
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = validator.preprocess(img).unsqueeze(0)
            img_emb = validator.encode_images(img_tensor).squeeze(0)
            similarities.append(validator.similarity(img_emb, text_emb))
        except Exception:
            pass

    return float(sum(similarities) / len(similarities)) if similarities else 0.0


def reverse_retrieval_score(
    feature_name: str,
    top_image_paths: list[Path | str],
    all_image_paths: list[Path | str],
    clip_encoder: "CLIPEncoder",
    k: int = 10,
    sample_size: int = 500,
) -> float:
    """Use the feature name as a text query and measure retrieval overlap.

    Encodes the name as text, searches the corpus by cosine similarity, and
    computes Jaccard overlap with the known top-activating images.

    If the name is genuinely descriptive, the text retrieval should recover
    many of the same images the feature activates on.

    Args:
        feature_name: Name to test.
        top_image_paths: Ground-truth feature images (the positive set).
        all_image_paths: Full corpus paths.
        clip_encoder: Pre-instantiated encoder (same or different from generator).
        k: Number of text-retrieval results to consider.
        sample_size: Randomly sample this many corpus images if corpus is large.

    Returns:
        Jaccard overlap in [0, 1]. 0 = no overlap, 1 = perfect overlap.
    """
    rng = np.random.default_rng(42)
    positive_set = {str(p) for p in top_image_paths}
    if len(all_image_paths) > sample_size:
        idxs = rng.choice(len(all_image_paths), size=sample_size, replace=False).tolist()
        sampled_paths = [all_image_paths[i] for i in idxs]
        # keep the positives in the pool, else Jaccard is biased toward 0
        present = {str(p) for p in sampled_paths}
        for p in all_image_paths:
            if str(p) in positive_set and str(p) not in present:
                sampled_paths.append(p)
                present.add(str(p))
    else:
        sampled_paths = list(all_image_paths)
        idxs = list(range(len(all_image_paths)))

    text_emb = clip_encoder.encode_text([feature_name]).squeeze(0)

    img_embs: list[torch.Tensor] = []
    for path in sampled_paths:
        try:
            img = Image.open(path).convert("RGB")
            t = clip_encoder.preprocess(img).unsqueeze(0)
            img_embs.append(clip_encoder.encode_images(t).squeeze(0))
        except Exception:
            img_embs.append(torch.zeros(text_emb.shape[0]))

    if not img_embs:
        return 0.0

    emb_matrix = torch.stack(img_embs)
    sims = (emb_matrix @ text_emb).numpy()
    top_k_local = np.argsort(sims)[::-1][:k].tolist()
    retrieved_paths = {str(sampled_paths[i]) for i in top_k_local}
    positive_paths = {str(p) for p in top_image_paths}

    intersection = len(retrieved_paths & positive_paths)
    union = len(retrieved_paths | positive_paths)
    return float(intersection / union) if union > 0 else 0.0


def batch_cross_validation(
    named_features: list[dict],
    all_image_paths: list[Path | str],
    generator_clip: "CLIPEncoder",
    validator_model: str = "ViT-B-32",
    k: int = 10,
) -> list[dict]:
    """Run both cross-model and reverse-retrieval validation for a list of features.

    Args:
        named_features: List of dicts with keys feature_id, name, top_paths.
        all_image_paths: Full corpus paths for reverse retrieval.
        generator_clip: The CLIP encoder used to generate names (for reverse retrieval).
        validator_model: Different CLIP model for cross-model check.
        k: Retrieval cut-off.

    Returns:
        List of result dicts with keys: feature_id, name, cross_model_score,
        reverse_retrieval_score.
    """
    from src.encoders.clip_encoder import CLIPEncoder
    validator = CLIPEncoder(model_name=validator_model, pretrained="openai")

    results = []
    for nf in named_features:
        cross = cross_model_alignment(nf["name"], nf["top_paths"], validator_model,
                                      validator=validator)
        reverse = reverse_retrieval_score(
            nf["name"], nf["top_paths"], all_image_paths, generator_clip, k=k
        )
        results.append({
            "feature_id": nf["feature_id"],
            "name": nf["name"],
            "cross_model_score": cross,
            "reverse_retrieval_score": reverse,
        })
    return results
