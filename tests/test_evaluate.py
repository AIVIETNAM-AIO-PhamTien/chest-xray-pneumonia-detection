"""Tests for the evaluate() metrics function."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import evaluate


class _IdentityLogits(nn.Module):
    """Returns its input unchanged, so tests can dictate predictions directly."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _loader(logits: list, labels: list, batch_size: int = 4) -> DataLoader:
    """Build a DataLoader that yields the given logits as "images".

    Args:
        logits: Per-sample logit rows fed straight into an identity model.
        labels: Per-sample class ids.
        batch_size: DataLoader batch size.

    Returns:
        A DataLoader over the (logits, labels) pairs.
    """
    images = torch.tensor(logits, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(images, targets), batch_size=batch_size)


def test_evaluate_perfect_predictions() -> None:
    """All predictions match labels -> every metric is 1.0."""
    logits = [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    labels = [1, 0, 1, 0]
    metrics = evaluate(_IdentityLogits(), _loader(logits, labels), torch.device("cpu"))

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_evaluate_all_predictions_wrong_uses_zero_division_guard() -> None:
    """Model always predicts class 0 while every label is 1.

    precision_score/recall_score would otherwise divide by zero; evaluate()
    must return 0.0 rather than raising.
    """
    logits = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    labels = [1, 1, 1]
    metrics = evaluate(_IdentityLogits(), _loader(logits, labels), torch.device("cpu"))

    assert metrics["accuracy"] == pytest.approx(0.0)
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)
    assert metrics["f1"] == pytest.approx(0.0)


def test_evaluate_empty_loader_raises() -> None:
    """An empty DataLoader must fail loudly instead of raising ZeroDivisionError."""
    empty_loader = _loader([], [])

    with pytest.raises(ValueError, match="empty DataLoader"):
        evaluate(_IdentityLogits(), empty_loader, torch.device("cpu"))
