"""Measures how well a generated feature name aligns with its top images."""

from pathlib import Path

from PIL import Image

from src.encoders.clip_encoder import CLIPEncoder


def clip_alignment_score(
    feature_name: str,
    top_image_paths: list[Path | str],
    clip_encoder: CLIPEncoder,
) -> float:
    """Compute the average CLIP cosine similarity between a feature name and
    its most-activating images.

    A higher score indicates that the generated name is a good textual
    description of what the images have in common.

    Args:
        feature_name: Short human-readable label produced by VLMFeatureNamer.
        top_image_paths: Paths to the K images that most strongly activate
            the feature.
        clip_encoder: Pre-instantiated CLIPEncoder.

    Returns:
        Mean cosine similarity in ``[-1, 1]``.  Higher is better.
    """
    text_emb = clip_encoder.encode_text([feature_name]).squeeze(0)

    similarities: list[float] = []
    for path in top_image_paths:
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = clip_encoder.preprocess(img).unsqueeze(0)
            img_emb = clip_encoder.encode_images(img_tensor).squeeze(0)
            similarities.append(clip_encoder.similarity(img_emb, text_emb))
        except Exception:
            pass

    return sum(similarities) / len(similarities) if similarities else 0.0


def batch_clip_alignment(
    named_features: list[dict],
    clip_encoder: CLIPEncoder,
) -> dict[int, float]:
    """Compute CLIP alignment scores for multiple named features.

    Args:
        named_features: List of dicts, each with keys:
            - ``"feature_id"`` (int)
            - ``"name"`` (str)
            - ``"top_paths"`` (list of Path)
        clip_encoder: Pre-instantiated CLIPEncoder.

    Returns:
        Dict mapping feature_id to its alignment score.
    """
    return {
        nf["feature_id"]: clip_alignment_score(
            nf["name"], nf["top_paths"], clip_encoder
        )
        for nf in named_features
    }
