"""Smoke tests for the transform pipelines.

These tests do not require the actual dataset; they only check that a
dummy grayscale image passes through the transform pipeline with the
expected output shape.
"""

import torch
from PIL import Image

from src.transforms import get_eval_transforms, get_train_transforms


def _dummy_image() -> Image.Image:
    """Create a small dummy grayscale image for shape-checking tests.

    Returns:
        A 256x256 single-channel PIL Image.
    """
    return Image.new("L", (256, 256), color=128)


def test_train_transforms_output_shape() -> None:
    """Train transforms should produce a 3x224x224 tensor by default."""
    transform = get_train_transforms(image_size=224)
    output = transform(_dummy_image())
    assert output.shape == torch.Size([3, 224, 224])


def test_eval_transforms_output_shape() -> None:
    """Eval transforms should produce a 3x224x224 tensor by default."""
    transform = get_eval_transforms(image_size=224)
    output = transform(_dummy_image())
    assert output.shape == torch.Size([3, 224, 224])
