"""Tests for the dependency-light Grad-CAM implementation."""

import torch
import torch.nn as nn

from src.explainability import GradCAM


class TinyClassifier(nn.Module):
    """Small deterministic CNN used to test hook and shape behaviour."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return binary logits for a batch.

        Args:
            inputs: Image tensor shaped ``(B, 3, H, W)``.

        Returns:
            Logits shaped ``(B, 2)``.
        """
        features = self.features(inputs)
        return self.classifier(self.pool(features).flatten(1))


def test_grad_cam_returns_finite_normalized_maps() -> None:
    """Grad-CAM should preserve batch/spatial shape and normalize each map."""
    torch.manual_seed(7)
    model = TinyClassifier()
    inputs = torch.randn(2, 3, 16, 20)

    with GradCAM(model, model.features) as explainer:
        maps, logits = explainer(inputs, target_class=1)

    assert maps.shape == (2, 16, 20)
    assert logits.shape == (2, 2)
    assert torch.isfinite(maps).all()
    assert float(maps.min()) >= 0.0
    assert float(maps.max()) <= 1.0


def test_grad_cam_accepts_per_sample_targets() -> None:
    """A batch may request a different explained class for each sample."""
    model = TinyClassifier()
    inputs = torch.randn(2, 3, 10, 10)

    with GradCAM(model, model.features) as explainer:
        maps, _ = explainer(inputs, target_class=torch.tensor([0, 1]))

    assert maps.shape == (2, 10, 10)
