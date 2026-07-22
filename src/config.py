"""Configuration loading utilities for training runs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import yaml


@dataclass
class DataConfig:
    """Settings related to dataset location and loading.

    Attributes:
        root_dir: Path to the chest_xray dataset root (with train/val/test).
        image_size: Target square size (in pixels) images are resized to.
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.
    """

    root_dir: str = "data/raw/chest_xray"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2


@dataclass
class ModelConfig:
    """Settings related to model architecture.

    Attributes:
        backbone: Name of the torchvision backbone (e.g. "resnet18").
        pretrained: Whether to load ImageNet-pretrained weights.
        num_classes: Number of output classes.
        freeze_backbone: Whether to freeze all layers except the final head.
    """

    backbone: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 2
    freeze_backbone: bool = False


@dataclass
class TrainConfig:
    """Settings related to the optimization loop.

    Attributes:
        epochs: Number of training epochs.
        lr: Learning rate for the optimizer.
        weight_decay: Weight decay (L2 regularization) coefficient.
        optimizer: Name of the optimizer to use.
        device: Preferred device ("cuda" or "cpu").
    """

    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    device: str = "cuda"


@dataclass
class OutputConfig:
    """Settings related to checkpoint and log locations.

    Attributes:
        checkpoint_dir: Directory where model checkpoints are saved.
        log_dir: Directory where training logs are saved.
    """

    checkpoint_dir: str = "models"
    log_dir: str = "outputs/logs"


@dataclass
class Config:
    """Top-level experiment configuration.

    Attributes:
        seed: Random seed for reproducibility.
        data: Dataset-related settings.
        model: Model architecture settings.
        train: Optimization loop settings.
        output: Checkpoint/log output settings.
    """

    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(config_path: Union[str, Path]) -> Config:
    """Load a YAML config file into a Config object.

    Args:
        config_path: Path to a YAML file matching the Config schema (see
            configs/baseline.yaml for an example).

    Returns:
        A populated Config instance.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(
        seed=raw.get("seed", 42),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
        output=OutputConfig(**raw.get("output", {})),
    )
