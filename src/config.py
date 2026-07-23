"""Configuration loading utilities for training runs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

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
        name: Registered model name (see `src/models/`), e.g. "resnet18"
            or "simple_cnn".
        pretrained: Whether to load pretrained weights.
        num_classes: Number of output classes.
        freeze_backbone: Whether to freeze all layers except the final head.
        params: Architecture-specific keyword arguments forwarded to the
            model builder (e.g. {"img_size": 224} for a ViT).
    """

    name: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 2
    freeze_backbone: bool = False
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    """Settings related to the optimization loop.

    Attributes:
        epochs: Number of training epochs.
        lr: Learning rate for the optimizer.
        weight_decay: Weight decay (L2 regularization) coefficient.
        optimizer: Name of the optimizer to use ("adam", "adamw", or "sgd").
        device: Preferred device ("cuda" or "cpu").
        scheduler: Name of the LR scheduler to use ("none", "step", "cosine",
            or "plateau").
        scheduler_params: Extra keyword arguments forwarded to the scheduler
            constructor (e.g. {"step_size": 5, "gamma": 0.1} for "step").
    """

    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    device: str = "cuda"
    scheduler: str = "none"
    scheduler_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputConfig:
    """Settings related to checkpoint, log, and experiment-tracking locations.

    Attributes:
        checkpoint_dir: Directory where model checkpoints are saved.
        log_dir: Directory where training logs are saved.
        run_name: Name for this experiment run, used for the checkpoint
            filename, log filename, and W&B run name. Defaults to the config
            file's stem (e.g. "exp_alice_clahe") when not set.
        wandb_project: W&B project name experiments are logged under.
        wandb_entity: W&B entity (team/user) to log under, if any.
        wandb_mode: W&B run mode ("online", "offline", or "disabled").
    """

    checkpoint_dir: str = "models"
    log_dir: str = "outputs/logs"
    run_name: Optional[str] = None
    wandb_project: str = "chest-xray-pneumonia"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"


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
