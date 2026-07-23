"""Dataset and DataLoader construction for the chest X-ray pneumonia dataset."""

from pathlib import Path
from typing import Dict, Tuple, Union

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.transforms import get_eval_transforms, get_train_transforms


def build_dataloaders(
    root_dir: Union[str, Path],
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    """Build train/val/test DataLoaders from the Kaggle chest_xray folder layout.

    Expects root_dir to contain train/, val/, and test/ subfolders, each with
    NORMAL/ and PNEUMONIA/ class folders (the layout used by the "Chest X-Ray
    Images (Pneumonia)" Kaggle dataset). Note that the dataset's original
    val/ split only has 16 images; re-split train/ yourself if a larger,
    more reliable validation set is needed.

    Args:
        root_dir: Path to the chest_xray dataset root.
        image_size: Target square size (in pixels) for resized images.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.

    Returns:
        A tuple of (loaders, class_to_idx) where loaders maps split name
        ("train", "val", "test") to a DataLoader, and class_to_idx maps
        class name to label index.
    """
    root_dir = Path(root_dir)
    train_set = ImageFolder(
        root_dir / "train", transform=get_train_transforms(image_size)
    )
    val_set = ImageFolder(root_dir / "val", transform=get_eval_transforms(image_size))
    test_set = ImageFolder(root_dir / "test", transform=get_eval_transforms(image_size))

    loaders = {
        "train": DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "val": DataLoader(
            val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
        "test": DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    }
    return loaders, train_set.class_to_idx
