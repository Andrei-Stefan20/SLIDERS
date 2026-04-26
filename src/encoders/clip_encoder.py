"""CLIP ViT-L/14 encoder for image and text feature extraction."""

from collections.abc import Callable
from typing import Protocol, cast

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image as PILImage

from src.utils.device import get_device


class _CLIPModel(Protocol):
    def eval(self) -> "_CLIPModel": ...

    def to(self, device: torch.device) -> "_CLIPModel": ...

    def encode_image(self, images: torch.Tensor) -> torch.Tensor: ...

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor: ...


class CLIPEncoder:
    """Wraps open-clip CLIP ViT-L/14 for image and text encoding.

    This encoder is used exclusively for feature naming and evaluation,
    not for retrieval (DINOv2 is used for that).
    """

    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai") -> None:
        """Load the CLIP model and its preprocessing transform.

        Args:
            model_name: open-clip model identifier.
            pretrained: Pretrained weights tag recognised by open-clip.
        """
        self.device = get_device()

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = cast(_CLIPModel, model)
        self.preprocess = cast(Callable[[PILImage.Image], torch.Tensor], preprocess)
        self.tokenizer = cast(
            Callable[[list[str]], torch.Tensor],
            open_clip.get_tokenizer(model_name),
        )
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into L2-normalised feature vectors.

        Args:
            images: Float32 tensor of shape ``(B, 3, H, W)``.

        Returns:
            L2-normalised float32 tensor of shape ``(B, 768)``.
        """
        images = images.to(self.device, dtype=torch.float32)
        features = self.model.encode_image(images)
        return F.normalize(features, dim=-1).cpu()

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of text strings into L2-normalised feature vectors.

        Args:
            texts: Strings to encode.

        Returns:
            L2-normalised float32 tensor of shape ``(N, 768)``.
        """
        tokens = self.tokenizer(texts).to(self.device)
        features = self.model.encode_text(tokens)
        return F.normalize(features, dim=-1).cpu()

    def similarity(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> float:
        """Compute cosine similarity between a single image and text embedding.

        Both inputs should already be L2-normalised (as returned by
        ``encode_images`` / ``encode_text``).

        Args:
            image_emb: Float32 tensor of shape ``(768,)`` or ``(1, 768)``.
            text_emb: Float32 tensor of shape ``(768,)`` or ``(1, 768)``.

        Returns:
            Scalar cosine similarity in ``[-1, 1]``.
        """
        image_emb = image_emb.flatten()
        text_emb = text_emb.flatten()
        return float(torch.dot(image_emb, text_emb).item())
