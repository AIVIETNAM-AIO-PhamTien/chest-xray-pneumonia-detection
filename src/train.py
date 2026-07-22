"""Training entrypoint for the pneumonia detection baseline.

Runs standalone from the command line, or can be imported and called from a
Colab/Kaggle notebook (see notebooks/train_baseline.ipynb).
"""

import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import load_config
from src.dataset import build_dataloaders
from src.evaluate import evaluate
from src.model import build_model
from src.utils import EarlyStopping, save_checkpoint, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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


def run_training(config_path: str) -> None:
    """Run the full training loop described by a config file.

    Args:
        config_path: Path to a YAML config (see configs/baseline.yaml).
    """
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    loaders, class_to_idx = build_dataloaders(
        root_dir=cfg.data.root_dir,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    logger.info("Classes: %s", class_to_idx)

    model = build_model(
        backbone=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        freeze_backbone=cfg.model.freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    early_stopping = EarlyStopping(patience=5)

    best_f1 = 0.0
    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_metrics = evaluate(model, loaders["val"], device)

        logger.info(
            "Epoch %d/%d - train_loss: %.4f - val_acc: %.4f - val_f1: %.4f",
            epoch,
            cfg.train.epochs,
            train_loss,
            val_metrics["accuracy"],
            val_metrics["f1"],
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_checkpoint(model, cfg.output.checkpoint_dir, filename="baseline_best.pth")

        early_stopping.step(val_metrics["f1"])
        if early_stopping.should_stop:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    test_metrics = evaluate(model, loaders["test"], device)
    logger.info("Test metrics: %s", test_metrics)


def main() -> None:
    """Parse command-line arguments and launch training."""
    parser = argparse.ArgumentParser(description="Train the pneumonia detection baseline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
