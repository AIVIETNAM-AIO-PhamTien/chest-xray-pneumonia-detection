"""Shared registry that every model architecture registers itself into."""

from typing import Callable, Dict

import torch.nn as nn

ModelBuilder = Callable[..., nn.Module]

MODEL_REGISTRY: Dict[str, ModelBuilder] = {}


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """Register a model builder function under a unique name.

    Args:
        name: Key usable as ``model.backbone`` in a YAML config.

    Returns:
        A decorator that registers the builder and returns it unchanged.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def decorator(builder: ModelBuilder) -> ModelBuilder:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model name already registered: {name}")
        MODEL_REGISTRY[name] = builder
        return builder

    return decorator


def build_model(
    backbone: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    **params: object,
) -> nn.Module:
    """Build a model by name from the registry.

    Args:
        backbone: Registered model name (e.g. "resnet18", "simple_cnn").
        num_classes: Number of output classes.
        pretrained: Whether to load pretrained weights.
        freeze_backbone: Whether to freeze all layers except the final head.
        **params: Architecture-specific keyword arguments (from
            ``model.params`` in the config).

    Returns:
        A PyTorch model ready for training.

    Raises:
        ValueError: If ``backbone`` is not a registered model name.
    """
    if backbone not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported backbone: {backbone!r}. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[backbone](
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        **params,
    )
