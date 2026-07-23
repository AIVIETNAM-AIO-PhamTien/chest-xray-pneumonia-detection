"""ResNet-18 transfer-learning classifier."""

import torch.nn as nn
from torchvision import models

from src.models.registry import register_model


@register_model("resnet18")
def build_resnet18(
    num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False
) -> nn.Module:
    """Build a ResNet-18 classifier.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet-pretrained weights.
        freeze_backbone: Whether to freeze all layers except the final head.

    Returns:
        The constructed model.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    return model
