"""Model construction for the pneumonia classification baseline."""

import torch.nn as nn
from torchvision import models


def build_model(
    backbone: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a transfer-learning classifier on top of a torchvision backbone.

    Args:
        backbone: Name of the torchvision backbone ("resnet18" or "resnet34").
        num_classes: Number of output classes (2 for NORMAL vs PNEUMONIA).
        pretrained: Whether to load ImageNet-pretrained weights.
        freeze_backbone: Whether to freeze all layers except the final head.

    Returns:
        A PyTorch model ready for training.

    Raises:
        ValueError: If an unsupported backbone name is given.
    """
    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif backbone == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    return model
