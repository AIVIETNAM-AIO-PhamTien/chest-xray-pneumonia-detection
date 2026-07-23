"""Example from-scratch CNN architecture.

Use this file as a template: write your own ``nn.Module``, add a builder
function with the exact keywords it needs, and decorate it with
``@register_model("your_name")``. It will be auto-discovered — no other
file needs to change.
"""

import torch
import torch.nn as nn

from src.models.registry import register_model


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Build one Conv-BatchNorm-ReLU-MaxPool block.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.

    Returns:
        A sequential block halving spatial resolution via max pooling.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class SimpleCNN(nn.Module):
    """Small CNN classifier: 4 conv blocks + global average pooling + linear head.

    Attributes:
        features: Convolutional feature extractor.
        classifier: Global-pooling + linear classification head.
    """

    def __init__(self, num_classes: int = 2) -> None:
        """Initialize the network layers.

        Args:
            num_classes: Number of output classes.
        """
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input batch of shape ``(N, 3, H, W)``.

        Returns:
            Class logits of shape ``(N, num_classes)``.
        """
        x = self.features(x)
        return self.classifier(x)


@register_model("simple_cnn")
def build_simple_cnn(
    num_classes: int = 2, pretrained: bool = False, freeze_backbone: bool = False
) -> nn.Module:
    """Build a SimpleCNN classifier.

    ``pretrained`` and ``freeze_backbone`` are accepted (build_model always
    passes them) but ignored — this model is trained from scratch.

    Args:
        num_classes: Number of output classes.
        pretrained: Ignored.
        freeze_backbone: Ignored.

    Returns:
        The constructed model.
    """
    return SimpleCNN(num_classes=num_classes)
