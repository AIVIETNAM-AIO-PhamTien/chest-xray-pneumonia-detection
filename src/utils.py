"""Utility functions shared across training and evaluation scripts."""

import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Seed value applied to all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(preferred: str = "cuda") -> torch.device:
    """Resolve a configured device name to a device that actually exists.

    Falls back to CPU when the requested accelerator is unavailable, so a
    config written for a GPU box still runs on a laptop. Unlike a plain
    ``cuda``-only check, an explicit "mps" request is honoured on Apple
    silicon instead of silently degrading to CPU.

    Args:
        preferred: Device name from the config ("cuda", "mps", or "cpu").

    Returns:
        A torch.device that is available on this machine.
    """
    if preferred.startswith("cuda") and torch.cuda.is_available():
        return torch.device(preferred)
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device() -> torch.device:
    """Return the best available device (CUDA, then MPS, then CPU).

    Returns:
        A torch.device instance representing the selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir: Union[str, Path],
    filename: str = "baseline_best.pth",
) -> Path:
    """Save model weights to disk.

    Args:
        model: Trained PyTorch model whose state_dict will be saved.
        checkpoint_dir: Directory where the checkpoint file is written.
        filename: Name of the checkpoint file.

    Returns:
        Path to the saved checkpoint file.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / filename
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


class EarlyStopping:
    """Stop training when a monitored metric has stopped improving.

    Attributes:
        patience: Number of epochs to wait after the last improvement.
        min_delta: Minimum change to qualify as an improvement.
        best_score: Best metric value observed so far.
        counter: Number of consecutive epochs without improvement.
        should_stop: Whether training should be stopped.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        """Initialize the early stopping tracker.

        Args:
            patience: Number of epochs to wait after the last improvement.
            min_delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> None:
        """Update state given the latest monitored score (higher is better).

        Args:
            score: Latest value of the monitored metric (e.g. validation F1).
        """
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            self.should_stop = self.counter >= self.patience
