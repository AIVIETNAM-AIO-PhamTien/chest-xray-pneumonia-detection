"""Tests for transform pipelines and dataset discovery helpers.

These tests do not require the real dataset; they build tiny ImageFolder
trees on the fly.
"""

from pathlib import Path
from typing import Dict

import pytest
import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from src.dataset import compute_class_weights, find_data_root
from src.transforms import get_eval_transforms, get_train_transforms


def _dummy_image() -> Image.Image:
    """Create a small dummy grayscale image for shape-checking tests.

    Returns:
        A 256x256 single-channel PIL Image.
    """
    return Image.new("L", (256, 256), color=128)


def _make_tree(root: Path, counts: Dict[str, int]) -> None:
    """Write a minimal ImageFolder-style dataset tree.

    Args:
        root: Directory that will hold the train/val/test splits.
        counts: Mapping of class name to number of images per split.
    """
    for split in ("train", "val", "test"):
        for cls, n in counts.items():
            split_dir = root / split / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                _dummy_image().save(split_dir / f"{cls}_{i}.jpeg")


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


def test_eval_transforms_are_deterministic() -> None:
    """Eval transforms must not augment, or validation metrics become noise."""
    transform = get_eval_transforms(image_size=64)
    image = _dummy_image()
    assert torch.equal(transform(image), transform(image))


def test_find_data_root_ignores_nested_duplicate(tmp_path: Path) -> None:
    """The published dataset nests a second copy; the outer tree must win.

    A recursive loader that picked the nested chest_xray/chest_xray/ tree
    would silently double-count every image.
    """
    outer = tmp_path / "chest_xray"
    _make_tree(outer, {"NORMAL": 1, "PNEUMONIA": 1})
    _make_tree(outer / "chest_xray", {"NORMAL": 1, "PNEUMONIA": 1})

    assert find_data_root(tmp_path) == outer


def test_find_data_root_raises_when_missing(tmp_path: Path) -> None:
    """A missing dataset should fail loudly, not return a wrong path."""
    with pytest.raises(FileNotFoundError, match="No chest_xray dataset root"):
        find_data_root(tmp_path)


def test_compute_class_weights_is_inverse_frequency(tmp_path: Path) -> None:
    """Rarer classes get proportionally larger weights."""
    root = tmp_path / "chest_xray"
    _make_tree(root, {"NORMAL": 1, "PNEUMONIA": 3})
    weights = compute_class_weights(ImageFolder(root / "train"))

    # 1 NORMAL vs 3 PNEUMONIA -> 4/(2*1)=2.0 and 4/(2*3)=0.667
    assert weights.tolist() == pytest.approx([2.0, 2.0 / 3.0])


def test_compute_class_weights_preserves_loss_scale(tmp_path: Path) -> None:
    """The frequency-weighted average weight is 1.0.

    This is what keeps a weighted run's loss comparable to an unweighted
    one, so the two experiments can sit in the same results table.
    """
    root = tmp_path / "chest_xray"
    _make_tree(root, {"NORMAL": 1, "PNEUMONIA": 3})
    dataset = ImageFolder(root / "train")
    weights = compute_class_weights(dataset)

    per_sample = torch.tensor([weights[t] for t in dataset.targets])
    assert per_sample.mean().item() == pytest.approx(1.0)
