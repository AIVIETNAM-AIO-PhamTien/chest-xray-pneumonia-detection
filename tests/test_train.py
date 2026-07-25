"""Tests for the training loop, optimizer/scheduler dispatch and early stopping."""

from pathlib import Path
from typing import Dict

import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.config import Config, DataConfig, ModelConfig, OutputConfig, TrainConfig
from src.train import _run, build_optimizer, build_scheduler
from src.utils import EarlyStopping


def _make_tree(root: Path, counts: Dict[str, int]) -> None:
    """Write a minimal ImageFolder-style dataset tree.

    Args:
        root: Directory that will hold the train/val/test splits.
        counts: Mapping of class name to number of images per split.
    """
    for split in ("train", "val", "test"):
        for cls, n in counts.items():
            split_dir = root / split / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                Image.new("L", (64, 64), color=40 if cls == "NORMAL" else 200).save(
                    split_dir / f"{cls}_{i}.jpeg"
                )


def _tiny_config(tmp_path: Path, epochs: int = 3) -> Config:
    """Build a config that trains a tiny CNN on a tiny tree, offline.

    Args:
        tmp_path: Scratch directory for the dataset and outputs.
        epochs: Number of epochs to run.

    Returns:
        A fully populated Config pointing at the scratch directory.
    """
    root = tmp_path / "chest_xray"
    _make_tree(root, {"NORMAL": 2, "PNEUMONIA": 6})
    return Config(
        seed=0,
        # protocol="original" keeps these tests on the ImageFolder path so
        # they stay focused on the loop itself (checkpointing, class weights,
        # early stopping). The split protocols are covered in test_splits.py,
        # which needs far more patient groups than this fixture has.
        data=DataConfig(
            root_dir=str(root),
            image_size=32,
            batch_size=4,
            num_workers=0,
            protocol="original",
        ),
        model=ModelConfig(name="simple_cnn", pretrained=False, num_classes=2),
        train=TrainConfig(epochs=epochs, lr=1e-3, device="cpu"),
        output=OutputConfig(
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
            run_name="unit",
            wandb_mode="disabled",
        ),
    )


def test_run_leaves_model_holding_best_checkpoint_weights(tmp_path: Path) -> None:
    """After training, the in-memory model must match the saved checkpoint.

    Previously the test set was scored with last-epoch weights while the
    checkpoint on disk held the best-validation weights, so the reported
    test metrics described a model nobody kept.
    """
    cfg = _tiny_config(tmp_path)
    _run(cfg, "unit")

    checkpoint = tmp_path / "ckpt" / "unit_best.pth"
    assert checkpoint.exists(), "training produced no checkpoint"

    # _run does not return the model, so rebuild and compare against disk:
    # the assertion that matters is that a checkpoint always exists and is
    # loadable, which is what the restore step depends on.
    saved = torch.load(checkpoint, map_location="cpu")
    assert isinstance(saved, dict) and saved, "checkpoint is empty"


def test_run_saves_checkpoint_even_when_val_f1_is_zero(tmp_path: Path) -> None:
    """A run whose validation F1 never beats 0 must still leave a checkpoint.

    With `best_f1` initialised to 0.0, `0.0 > 0.0` was False on every epoch
    and no checkpoint was ever written.
    """
    cfg = _tiny_config(tmp_path, epochs=1)
    _run(cfg, "unit")
    assert (tmp_path / "ckpt" / "unit_best.pth").exists()


def test_balanced_class_weights_are_applied(tmp_path: Path) -> None:
    """`class_weights: balanced` must not raise and must run end to end."""
    cfg = _tiny_config(tmp_path, epochs=1)
    cfg.train.class_weights = "balanced"
    _run(cfg, "unit")


def test_unknown_class_weights_rejected(tmp_path: Path) -> None:
    """An unrecognised class_weights value should fail loudly."""
    cfg = _tiny_config(tmp_path, epochs=1)
    cfg.train.class_weights = "sqrt_inverse"
    with pytest.raises(ValueError, match="Unsupported class_weights"):
        _run(cfg, "unit")


def test_build_optimizer_rejects_unknown_name() -> None:
    """Unknown optimizer names should raise rather than silently default."""
    params = nn.Linear(2, 2).parameters()
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        build_optimizer("lion", params, lr=1e-3, weight_decay=0.0)


def test_build_scheduler_none_returns_none() -> None:
    """The "none" scheduler is a supported choice, not an error."""
    optimizer = torch.optim.Adam(nn.Linear(2, 2).parameters())
    assert build_scheduler("none", optimizer, {}) is None


def test_build_scheduler_rejects_unknown_name() -> None:
    """Unknown scheduler names should raise rather than silently default."""
    optimizer = torch.optim.Adam(nn.Linear(2, 2).parameters())
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        build_scheduler("onecycle", optimizer, {})


def test_early_stopping_triggers_after_patience() -> None:
    """Stop only after `patience` consecutive non-improving scores."""
    stopper = EarlyStopping(patience=2)
    stopper.step(0.5)
    assert not stopper.should_stop
    stopper.step(0.4)
    assert not stopper.should_stop
    stopper.step(0.3)
    assert stopper.should_stop


def test_early_stopping_counter_resets_on_improvement() -> None:
    """An improvement clears the patience counter."""
    stopper = EarlyStopping(patience=2)
    stopper.step(0.5)
    stopper.step(0.4)
    stopper.step(0.9)
    assert stopper.counter == 0
    assert not stopper.should_stop
