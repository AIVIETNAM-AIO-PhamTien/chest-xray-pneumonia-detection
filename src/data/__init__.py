"""Data pipeline for src/train.py's single-holdout CLI baseline.

This is a separate, deliberately-kept pipeline from src/dataset.py +
src/transforms.py + src/splits.py, which is what the canonical research
notebook and report scripts use to produce the frozen final numbers. Per
CLAUDE.md, src.train "vẫn là CLI baseline/single-holdout; không dùng nó để
tạo số cuối trong report 5-fold" (it is not used to produce the final 5-fold
report numbers), so changes here are lower-risk than changes to the
notebook's pipeline.
"""

from .build_loaders import DataLoaderConfig, LoaderBundle, build_dataloaders
from .cxr_dataset import CXRDataset
from .imbalance import FocalLoss
from .transforms import get_eval_transforms, get_train_transforms

__all__ = [
    "CXRDataset",
    "DataLoaderConfig",
    "FocalLoss",
    "LoaderBundle",
    "build_dataloaders",
    "get_eval_transforms",
    "get_train_transforms",
]
