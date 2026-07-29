"""Tests for the class-imbalance loss used by src/train.py's CLI baseline."""

import torch

from src.data.imbalance import FocalLoss


def test_focal_loss_is_finite() -> None:
    criterion = FocalLoss(alpha=[1.2, 0.8], gamma=2.0)
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]], requires_grad=True)
    targets = torch.tensor([0, 1])
    loss = criterion(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
