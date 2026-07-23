"""Model architectures package.

Each architecture file registers itself via ``@register_model(...)`` when
imported. Add a new architecture file, then import it below to activate it.
"""

from src.models import resnet18, simple_cnn  # noqa: F401
from src.models.registry import MODEL_REGISTRY, build_model, register_model

__all__ = ["MODEL_REGISTRY", "build_model", "register_model"]
