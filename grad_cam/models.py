"""Building the trained CNNs and getting the layer Grad-CAM needs.

The four training notebooks in Alex_Shim/ all use torchvision models with the head
swapped for 500 classes, saved as a plain state_dict. I just rebuild the same thing and
point Grad-CAM at the last conv layer of each. The tricky one is Swin - it's a
transformer, so its feature maps come out token-shaped and need a permute (see below).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torchvision import models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Same as the 'val' transform in the training notebooks - keep it identical or the
# CAMs won't line up with the reported predictions.
EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def swin_reshape(act: torch.Tensor) -> torch.Tensor:
    # Swin gives activations as (B, H, W, C), not (B, C, H, W) like the CNNs. Grad-CAM
    # was coming out blank until I added this permute.
    return act.permute(0, 3, 1, 2).contiguous()


@dataclass(frozen=True)
class ArchSpec:
    build: Callable[[int], nn.Module]          # make the model with a 500-class head
    target_layer: Callable[[nn.Module], nn.Module]   # layer to hook for Grad-CAM
    reshape: Callable[[torch.Tensor], torch.Tensor] | None = None


def _build_resnet(depth: int) -> Callable[[int], nn.Module]:
    builder = {18: models.resnet18, 50: models.resnet50}[depth]

    def build(num_classes: int) -> nn.Module:
        m = builder(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    return build


def _build_convnext_tiny(num_classes: int) -> nn.Module:
    m = models.convnext_tiny(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    return m


def _build_swin_t(num_classes: int) -> nn.Module:
    m = models.swin_t(weights=None)
    m.head = nn.Linear(m.head.in_features, num_classes)
    return m


# The last conv block is the standard Grad-CAM target: deepest layer that still keeps
# spatial info. For ResNet that's layer4, for ConvNeXt/Swin it's features[-1].
MODEL_REGISTRY: dict[str, ArchSpec] = {
    "resnet18": ArchSpec(_build_resnet(18), lambda m: m.layer4[-1]),
    "resnet50": ArchSpec(_build_resnet(50), lambda m: m.layer4[-1]),
    "convnext_tiny": ArchSpec(_build_convnext_tiny, lambda m: m.features[-1]),
    "swin_t": ArchSpec(_build_swin_t, lambda m: m.features[-1], reshape=swin_reshape),
}


def load_model(arch, num_classes=500, weights_path=None, device="cpu"):
    """Build the model and load a saved state_dict.

    Pass weights_path=None to get a random model - only handy for a quick smoke test
    before the trained .pth exists.
    """
    if arch not in MODEL_REGISTRY:
        raise KeyError(f"Unknown arch '{arch}'. Options: {sorted(MODEL_REGISTRY)}")

    device = torch.device(device)
    model = MODEL_REGISTRY[arch].build(num_classes)

    if weights_path is not None:
        state = torch.load(weights_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:   # some checkpoints wrap it
            state = state["state_dict"]
        model.load_state_dict(state)

    return model.to(device).eval()


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    # undo the ImageNet normalisation so we can actually show the image
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(-1, 1, 1)
    if tensor.dim() == 4:
        mean, std = mean.unsqueeze(0), std.unsqueeze(0)
    return (tensor * std + mean).clamp(0, 1)


def genus_of(class_name: str) -> str:
    # iNat folders look like id_Kingdom_..._Family_Genus_species, so genus is the
    # second-to-last chunk. Used to find same-genus confusions.
    parts = class_name.split("_")
    return parts[-2] if len(parts) >= 2 else class_name
