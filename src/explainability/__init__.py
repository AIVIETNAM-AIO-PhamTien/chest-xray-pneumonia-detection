"""Post-hoc explainability utilities for chest X-ray classifiers."""

from src.explainability.grad_cam import GradCAM, resolve_target_layer

__all__ = ["GradCAM", "resolve_target_layer"]
