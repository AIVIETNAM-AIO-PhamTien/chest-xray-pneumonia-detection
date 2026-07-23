"""Training entrypoint for the pneumonia detection baseline.

Runs standalone from the command line, or can be imported and called from a
Colab/Kaggle notebook (see notebooks/train_baseline.ipynb).
"""

import argparse
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from src.config import Config, load_config
from src.dataset import build_dataloaders
from src.evaluate import evaluate
from src.models import build_model
from src.utils import EarlyStopping, save_checkpoint, set_seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

_OPTIMIZER_REGISTRY = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": lambda params, lr, weight_decay: torch.optim.SGD(
        params, lr=lr, weight_decay=weight_decay, momentum=0.9
    ),
}


def build_optimizer(
    name: str, params: Any, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    """Build an optimizer by name.

    Args:
        name: Optimizer name ("adam", "adamw", or "sgd").
        params: Model parameters to optimize.
        lr: Learning rate.
        weight_decay: Weight decay (L2 regularization) coefficient.

    Returns:
        A configured optimizer instance.

    Raises:
        ValueError: If an unsupported optimizer name is given.
    """
    if name not in _OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unsupported optimizer: {name}. Available: {sorted(_OPTIMIZER_REGISTRY)}"
        )
    return _OPTIMIZER_REGISTRY[name](params, lr=lr, weight_decay=weight_decay)


def build_scheduler(
    name: str, optimizer: torch.optim.Optimizer, params: Dict[str, Any]
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Build an LR scheduler by name.

    Args:
        name: Scheduler name ("none", "step", "cosine", or "plateau").
        optimizer: Optimizer the scheduler will adjust.
        params: Extra keyword arguments forwarded to the scheduler
            constructor.

    Returns:
        A configured scheduler instance, or None if name is "none".

    Raises:
        ValueError: If an unsupported scheduler name is given.
    """
    if name == "none":
        return None
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **params)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
    raise ValueError(f"Unsupported scheduler: {name}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run a single training epoch.

    Args:
        model: Model being trained.
        loader: Training DataLoader.
        optimizer: Optimizer used to update model weights.
        criterion: Loss function.
        device: Device to run computation on.

    Returns:
        The average training loss over the epoch.
    """
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def _attach_file_logger(log_dir: Path, run_name: str) -> None:
    """Attach a per-run file handler to the module logger, replacing any prior one.

    Args:
        log_dir: Directory where the run's log file is written.
        run_name: Name used for the log file (``<run_name>.log``).
    """
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{run_name}.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)


def _run(cfg: Config, run_name: str) -> None:
    """Run the full training loop for an already-loaded config.

    Args:
        cfg: Fully populated experiment configuration.
        run_name: Name identifying this run (checkpoint filename, log
            filename, and W&B run name).
    """
    set_seed(cfg.seed)
    _attach_file_logger(Path(cfg.output.log_dir), run_name)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    wandb.init(
        project=cfg.output.wandb_project,
        entity=cfg.output.wandb_entity,
        name=run_name,
        mode=cfg.output.wandb_mode,
        config=asdict(cfg),
    )

    loaders, class_to_idx = build_dataloaders(
        root_dir=cfg.data.root_dir,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    logger.info("Classes: %s", class_to_idx)

    model = build_model(
        name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        freeze_backbone=cfg.model.freeze_backbone,
        **cfg.model.params,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        cfg.train.optimizer,
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = build_scheduler(
        cfg.train.scheduler, optimizer, cfg.train.scheduler_params
    )
    early_stopping = EarlyStopping(patience=5)

    best_f1 = 0.0
    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device
        )
        val_metrics = evaluate(model, loaders["val"], device)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics["f1"])
        elif scheduler is not None:
            scheduler.step()

        logger.info(
            "Epoch %d/%d - train_loss: %.4f - val_acc: %.4f - val_f1: %.4f",
            epoch,
            cfg.train.epochs,
            train_loss,
            val_metrics["accuracy"],
            val_metrics["f1"],
        )
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "lr": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_checkpoint(
                model, cfg.output.checkpoint_dir, filename=f"{run_name}_best.pth"
            )

        early_stopping.step(val_metrics["f1"])
        if early_stopping.should_stop:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    test_metrics = evaluate(model, loaders["test"], device)
    logger.info("Test metrics: %s", test_metrics)
    wandb.log(
        {f"test_{k}": v for k, v in test_metrics.items() if k != "confusion_matrix"}
    )
    wandb.finish()


def run_training(config_path: str) -> None:
    """Run the full training loop described by a config file.

    Args:
        config_path: Path to a YAML config (see configs/baseline.yaml).
    """
    cfg = load_config(config_path)
    run_name = cfg.output.run_name or Path(config_path).stem
    _run(cfg, run_name)


def main() -> None:
    """Parse command-line arguments and launch training."""
    parser = argparse.ArgumentParser(
        description="Train the pneumonia detection baseline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--root-dir", type=str, default=None, help="Override cfg.data.root_dir."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override cfg.train.device."
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="Override cfg.output.wandb_mode.",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Override cfg.model.name."
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override cfg.train.epochs."
    )
    parser.add_argument("--lr", type=float, default=None, help="Override cfg.train.lr.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.root_dir is not None:
        cfg.data.root_dir = args.root_dir
    if args.device is not None:
        cfg.train.device = args.device
    if args.wandb_mode is not None:
        cfg.output.wandb_mode = args.wandb_mode
    if args.model is not None:
        cfg.model.name = args.model
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.lr is not None:
        cfg.train.lr = args.lr

    run_name = cfg.output.run_name or Path(args.config).stem
    _run(cfg, run_name)


if __name__ == "__main__":
    main()
