"""DINOv2 ViT-L/14 encoder for extracting image features."""

from typing import Protocol, cast

import torch
import torchvision.transforms as T

DINO_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

from src.utils.device import get_device


class _DINOModel(Protocol):
    def eval(self) -> "_DINOModel": ...

    def to(self, device: torch.device) -> "_DINOModel": ...

    def __call__(self, images: torch.Tensor) -> torch.Tensor: ...

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]: ...


class DINOEncoder:
    """Wraps DINOv2 ViT-L/14 for CLS-token and patch-token extraction.

    Loads the model from ``facebookresearch/dinov2`` via torch.hub.
    All inference is performed with ``torch.no_grad()`` in float32.
    """

    def __init__(self, use_patches: bool = False, model_name: str | None = None) -> None:
        """Initialise and load the DINOv2 ViT-L/14 model.

        Args:
            use_patches: If True, return patch tokens instead of the CLS token.
                CLS token has shape ``(B, 1024)``; patch tokens ``(B, N_patches, 1024)``.
            model_name: torch.hub entrypoint. Defaults to the registers variant
                ``dinov2_vitl14_reg`` for patches (it removes the high-norm artifact
                patches that would otherwise become spurious SAE features), and the
                plain ``dinov2_vitl14`` for the CLS retrieval path.
        """
        device = get_device()
        if device.type == "mps":
            device = torch.device("cpu")
        self.device = device
        self.use_patches = use_patches
        self.model_name = model_name or (
            "dinov2_vitl14_reg" if use_patches else "dinov2_vitl14"
        )

        model = cast(
            _DINOModel,
            torch.hub.load("facebookresearch/dinov2", self.model_name),
        )
        model.eval()
        model.to(self.device)
        self.model = model

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images.

        Args:
            images: Float32 tensor of shape ``(B, 3, H, W)``, values in
                ``[0, 1]`` or pre-normalised.

        Returns:
            Float32 tensor.  Shape ``(B, 1024)`` when ``use_patches=False``,
            or ``(B, N_patches, 1024)`` when ``use_patches=True``.
        """
        images = images.to(self.device, dtype=torch.float32)

        if self.use_patches:
            out = self.model.forward_features(images)
            return out["x_norm_patchtokens"].cpu()
        else:
            return self.model(images).cpu()
