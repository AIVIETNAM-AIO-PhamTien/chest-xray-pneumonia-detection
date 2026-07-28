"""Dataset and DataLoader construction for the chest X-ray pneumonia dataset."""

from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from src.transforms import get_eval_transforms, get_train_transforms


class ManifestDataset(Dataset):
    """Dataset backed by an explicit (path, label) list rather than folders.

    ImageFolder derives labels from directory layout, which cannot express a
    split that cuts across the published folders. Driving the dataset from a
    manifest keeps the exact file list for a run inspectable and reproducible.

    Attributes:
        rows: The (image path, class id) pairs this dataset serves.
        transform: Transform applied to each loaded image.
        targets: Class id per sample, exposed for compute_class_weights.
    """

    def __init__(
        self,
        rows: Sequence[Tuple[str, int]],
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            rows: Sequence of (image path, class id) pairs.
            transform: Transform applied to each PIL image.
            class_to_idx: Class name to id mapping; defaults to
                {"NORMAL": 0, "PNEUMONIA": 1}.
        """
        self.rows = list(rows)
        self.transform = transform
        self.class_to_idx = class_to_idx or {"NORMAL": 0, "PNEUMONIA": 1}
        self.targets = [label for _, label in self.rows]

    def __len__(self) -> int:
        """Return the number of samples.

        Returns:
            Sample count.
        """
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Load and transform one sample.

        Args:
            index: Position in the manifest.

        Returns:
            A (image tensor, class id) tuple.
        """
        path, label = self.rows[index]
        # 283 of the 5,856 published files are stored as RGB rather than
        # grayscale, so the mode is normalised here as well as in transforms.
        image = Image.open(path).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def find_data_root(search_paths: Union[str, Path, List[Union[str, Path]]]) -> Path:
    """Locate the directory that directly contains train/NORMAL and train/PNEUMONIA.

    Kaggle, Colab and local checkouts all mount this dataset at different
    depths, and the published copy contains a nested ``chest_xray/chest_xray/``
    tree holding a second copy of every image. Searching for the marker
    folders instead of hardcoding a path avoids both problems: the shallowest
    match wins, so the nested duplicate is never selected.

    Args:
        search_paths: One or more directories to search under.

    Returns:
        Path to the dataset root (the parent of ``train/``).

    Raises:
        FileNotFoundError: If no candidate directory contains the expected
            train/NORMAL and train/PNEUMONIA layout.
    """
    if isinstance(search_paths, (str, Path)):
        search_paths = [search_paths]

    candidates = []
    for base in search_paths:
        base = Path(base)
        if not base.exists():
            continue
        for train_dir in base.rglob("train"):
            if (train_dir / "NORMAL").is_dir() and (train_dir / "PNEUMONIA").is_dir():
                candidates.append(train_dir.parent)

    if not candidates:
        searched = [str(p) for p in search_paths]
        raise FileNotFoundError(
            f"No chest_xray dataset root found under: {searched}. "
            "Expected a directory containing train/NORMAL/ and train/PNEUMONIA/."
        )

    # Shallowest path wins -> picks chest_xray/ over chest_xray/chest_xray/.
    return min(candidates, key=lambda p: len(p.parts))


def compute_class_weights(dataset: Dataset) -> torch.Tensor:
    """Compute inverse-frequency class weights from a labelled dataset.

    Args:
        dataset: Dataset exposing class ids through either ``targets`` (the
            ImageFolder convention) or ``labels`` (the CXRDataset convention).

    Returns:
        A float tensor of per-class weights indexed by class id. Weights are
        normalised so their sample-frequency-weighted average is 1.0, which
        keeps the overall loss on the same scale as the unweighted run and
        makes the two directly comparable.
    """
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = getattr(dataset, "labels", None)
    if targets is None:
        raise TypeError(
            "Dataset must expose class ids through a 'targets' or 'labels' "
            "attribute."
        )
    targets = [int(label) for label in targets]
    counts = Counter(targets)
    num_classes = max(targets, default=-1) + 1
    if num_classes < 1 or any(counts[index] == 0 for index in range(num_classes)):
        raise ValueError("Class ids must be contiguous and every class must occur.")
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (num_classes * counts[i]) for i in range(num_classes)],
        dtype=torch.float,
    )
    return weights


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


def build_dataloaders_from_manifest(
    manifest,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    """Build DataLoaders from a manifest carrying an explicit "split" column.

    Use this instead of build_dataloaders when the train/val boundary comes
    from a split protocol rather than the published folder layout (see
    src/splits.py).

    Args:
        manifest: DataFrame from src.splits.make_splits.
        image_size: Target square size (in pixels) for resized images.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.

    Returns:
        A tuple of (loaders, class_to_idx), matching build_dataloaders.
    """
    from src.splits import manifest_rows

    class_to_idx = {"NORMAL": 0, "PNEUMONIA": 1}
    transforms_by_split = {
        "train": get_train_transforms(image_size),
        "val": get_eval_transforms(image_size),
        "test": get_eval_transforms(image_size),
    }

    loaders = {}
    for split, transform in transforms_by_split.items():
        dataset = ManifestDataset(
            manifest_rows(manifest, split),
            transform=transform,
            class_to_idx=class_to_idx,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
    return loaders, class_to_idx
