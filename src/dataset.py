"""Dataset and DataLoader construction for the chest X-ray pneumonia dataset.

This module (plus src/transforms.py and src/splits.py) is the pipeline used
by the canonical research notebook (chest_xray_research_complete.ipynb) and
the audit scripts to produce the frozen, hash-verified final numbers. Do not
change any code path reachable from build_dataloaders() or the transforms in
a way that alters its numeric output -- that would invalidate the frozen
checkpoints and manifests under artifacts/final/.

src/train.py's a_paper_compatible/b_patient_grouped protocols instead use a
separate, deliberately-kept pipeline in src/data/ (CXRDataset, build_loaders,
its own transforms/imbalance handling) for the historical single-holdout CLI
baseline -- see CLAUDE.md ("src.train vẫn là CLI baseline/single-holdout;
không dùng nó để tạo số cuối trong report 5-fold"). The two are intentionally
separate pipelines for two different purposes, not accidental duplication.
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from src.transforms import get_eval_transforms, get_train_transforms


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
