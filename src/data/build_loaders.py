from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .cxr_dataset import CXRDataset
from .imbalance import (
    build_classification_loss,
    count_classes,
    make_weighted_sampler,
)
from .transforms import get_eval_transforms, get_train_transforms


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 32
    img_size: int = 128
    channels: int = 3
    augmentation_policy: str = "advanced"
    balance_strategy: str = "weighted_ce"
    num_classes: int = 2
    num_workers: int = 4
    seed: int = 42
    pretrained: bool = False
    pin_memory: Optional[bool] = None
    verify_paths: bool = False
    mean: Optional[Sequence[float]] = None
    std: Optional[Sequence[float]] = None
    focal_gamma: float = 2.0
    effective_beta: float = 0.999


@dataclass
class LoaderBundle:
    train: DataLoader
    val: Optional[DataLoader]
    test: Optional[DataLoader]
    criterion: nn.Module
    class_counts: torch.Tensor
    dataframes: Dict[str, pd.DataFrame]


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader_kwargs(config: DataLoaderConfig) -> dict:
    pin_memory = (
        torch.cuda.is_available() if config.pin_memory is None else config.pin_memory
    )
    kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": config.num_workers > 0,
    }
    if config.num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return kwargs


def _make_optional_loader(
    df: pd.DataFrame,
    transform,
    root_dir: Optional[Path],
    config: DataLoaderConfig,
) -> Optional[DataLoader]:
    if df.empty:
        return None
    dataset = CXRDataset(
        df=df,
        transform=transform,
        root_dir=root_dir,
        verify_paths=config.verify_paths,
    )
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        shuffle=False,
        generator=generator,
        **_loader_kwargs(config),
    )


def build_dataloaders(
    split_csv: Path,
    root_dir: Optional[Path] = None,
    config: Optional[DataLoaderConfig] = None,
) -> LoaderBundle:
    config = config or DataLoaderConfig()
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    frame = pd.read_csv(split_csv)
    required = {"path", "label", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Split CSV is missing {sorted(missing)}. Run split.py first."
        )

    frame["split"] = frame["split"].astype(str).str.lower()
    dataframes = {
        name: frame[frame["split"].eq(name)].reset_index(drop=True)
        for name in ("train", "val", "test")
    }
    if dataframes["train"].empty:
        raise ValueError("Training split is empty.")

    train_transform = get_train_transforms(
        img_size=config.img_size,
        channels=config.channels,
        policy=config.augmentation_policy,
        pretrained=config.pretrained,
        mean=config.mean,
        std=config.std,
    )
    eval_transform = get_eval_transforms(
        img_size=config.img_size,
        channels=config.channels,
        pretrained=config.pretrained,
        mean=config.mean,
        std=config.std,
    )

    train_dataset = CXRDataset(
        df=dataframes["train"],
        transform=train_transform,
        root_dir=root_dir,
        verify_paths=config.verify_paths,
    )
    labels = train_dataset.labels
    class_counts = count_classes(labels, num_classes=config.num_classes)

    balance = config.balance_strategy.lower()
    sampler = None
    if balance == "sampler":
        sampler, _ = make_weighted_sampler(
            labels,
            num_classes=config.num_classes,
            seed=config.seed,
        )
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = build_classification_loss(
            strategy=balance,
            labels=labels,
            num_classes=config.num_classes,
            gamma=config.focal_gamma,
            beta=config.effective_beta,
        )

    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator,
        **_loader_kwargs(config),
    )

    val_loader = _make_optional_loader(
        dataframes["val"],
        eval_transform,
        root_dir,
        config,
    )
    test_loader = _make_optional_loader(
        dataframes["test"],
        eval_transform,
        root_dir,
        config,
    )

    return LoaderBundle(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        criterion=criterion,
        class_counts=class_counts,
        dataframes=dataframes,
    )


def print_bundle_summary(bundle: LoaderBundle) -> None:
    print(f"Train samples: {len(bundle.train.dataset)}")
    print(f"Validation samples: {len(bundle.val.dataset) if bundle.val else 0}")
    print(f"Test samples: {len(bundle.test.dataset) if bundle.test else 0}")
    print(f"Class counts: {bundle.class_counts.tolist()}")
    images, labels = next(iter(bundle.train))
    print(f"One batch image shape: {tuple(images.shape)}")
    print(f"One batch label shape: {tuple(labels.shape)}")
    print(f"Criterion: {bundle.criterion.__class__.__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test CXR DataLoaders.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--root-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--channels", type=int, choices=[1, 3], default=3)
    parser.add_argument(
        "--augmentation",
        choices=["none", "paper", "advanced"],
        default="advanced",
    )
    parser.add_argument(
        "--balance",
        choices=[
            "none",
            "sampler",
            "weighted_ce",
            "effective_ce",
            "focal",
            "effective_focal",
        ],
        default="weighted_ce",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--verify-paths", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DataLoaderConfig(
        batch_size=args.batch_size,
        img_size=args.img_size,
        channels=args.channels,
        augmentation_policy=args.augmentation,
        balance_strategy=args.balance,
        num_workers=args.num_workers,
        seed=args.seed,
        pretrained=args.pretrained,
        verify_paths=args.verify_paths,
    )
    bundle = build_dataloaders(
        split_csv=args.csv,
        root_dir=args.root_dir,
        config=config,
    )
    print_bundle_summary(bundle)


if __name__ == "__main__":
    main()
